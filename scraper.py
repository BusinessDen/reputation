#!/usr/bin/env python3
"""
Reputation - Scraper
1. Scrapes third-party mentions of BusinessDen (Google News RSS, WordPress API,
   Reddit RSS, Medium RSS) — last 48 hours only.
2. Ingests BusinessDen's own articles via RSS (since March 1, 2026).
3. Generates AI summaries via Claude API:
   - Daily summary of new mentions (every run with new mentions)
   - Weekly summary (Mondays)
   - Monthly summary (1st of month)
   - Per-article reception summary (2 days, 7 days, 30 days after publish)
   - Byline analysis (per reporter)

Data is append-only for historical memory.
"""

import feedparser
import requests
import json
import os
import hashlib
import re
import html
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = "mentions-data.json"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

EXCLUDED_DOMAINS = ["businessden.com", "www.businessden.com"]
SPAM_DOMAINS = [
    "merriam-webster.com", "statmuse.com", "alltrails.com", "untappd.com",
    "yelp.com", "tiktok.com", "lamamapasteis.com.br", "tomasiusa.com",
    "univet.ee", "wisap.de", "facestudio.eu", "ingebjorg.no",
    "rectificadosindustriales.com", "fernandopando.com", "cmarinho.com.br",
    "prwe.com", "sigra.com", "educacioninnovadora.es", "lan-portal.uob.edu.ly",
    "weterynarzjarocin.pl",
]

GNEWS_QUERIES = [
    '"BusinessDen" -site:businessden.com when:2d',
    '"BusinessDen" blog -site:businessden.com when:2d',
    '"Thomas Gounley" Denver -site:businessden.com when:2d',
    '"Justin Wingerter" Denver -site:businessden.com when:2d',
    '"Matt Geiger" Denver business -site:businessden.com when:2d',
    '"Aaron Kremer" -site:businessden.com when:2d',
]

REPORTERS = ["Thomas Gounley", "Justin Wingerter", "Matt Geiger", "Aaron Kremer"]
BD_ARTICLES_SINCE = "2026-03-01T00:00:00+00:00"
USER_AGENT = "ReputationTracker/1.0 (BusinessDen)"

RELEVANCE_KEYWORDS = [
    "businessden", "business den",
    "thomas gounley", "justin wingerter", "matt geiger", "aaron kremer",
    "denver", "colorado", "real estate", "commercial", "office",
    "restaurant", "brewery", "development", "developer",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_id(url: str) -> str:
    normalized = url.lower().strip().rstrip("/")
    normalized = re.sub(r'[?&](utm_\w+|fbclid|gclid|ref|oc|source|medium|campaign)=[^&]*', '', normalized)
    return hashlib.sha256(normalized.rstrip("?&").encode()).hexdigest()[:16]

def is_excluded(url: str) -> bool:
    try:
        d = urlparse(url).netloc.lower()
        if d.startswith("www."): d = d[4:]
        for excl in EXCLUDED_DOMAINS + SPAM_DOMAINS:
            if d == excl or d.endswith("." + excl): return True
    except: pass
    return False

def is_relevant(title: str, snippet: str) -> bool:
    text = (title + " " + snippet).lower()
    return any(kw in text for kw in RELEVANCE_KEYWORDS)

def parse_rss_date(date_str: str) -> str | None:
    if not date_str: return None
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str).isoformat()
    except: return date_str

def extract_domain(url: str) -> str:
    try:
        d = urlparse(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except: return "unknown"

def strip_html(text: str) -> str:
    return html.unescape(re.sub(r'<[^>]+>', '', text or "")).strip()

def truncate(text: str, length: int = 500) -> str:
    if not text: return ""
    return text[:length - 3] + "..." if len(text) > length else text

def load_data() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {
        "mentions": [],
        "bd_articles": [],
        "summaries": {"daily": [], "weekly": [], "monthly": []},
        "byline": [],
        "article_reception": {},
        "last_updated": None,
        "run_log": [],
    }

def save_data(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------

def call_claude(prompt: str, max_tokens: int = 2000) -> str | None:
    if not ANTHROPIC_API_KEY:
        print("    [AI] No ANTHROPIC_API_KEY set — skipping")
        return None
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]
    except Exception as e:
        print(f"    [AI] Error: {e}")
        return None


# ---------------------------------------------------------------------------
# Source 1: Google News RSS
# ---------------------------------------------------------------------------

def scrape_gnews_rss() -> list[dict]:
    results = []
    seen_titles = set()
    for query in GNEWS_QUERIES:
        encoded = requests.utils.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        print(f"  [GNEWS] {query}")
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                src_title = entry.get("source", {}).get("title", "Unknown")
                src_href = entry.get("source", {}).get("href", "")
                glink = entry.get("link", "")
                if (src_href and is_excluded(src_href)) or is_excluded(glink): continue
                if src_title.lower().strip() == "businessden": continue
                title_raw = entry.get("title", "")
                tk = title_raw.lower().strip()
                if tk in seen_titles: continue
                seen_titles.add(tk)
                title = title_raw.rsplit(" - ", 1)[0].strip() if " - " in title_raw else title_raw.strip()
                snippet = truncate(strip_html(entry.get("summary", "")))
                if not is_relevant(title, snippet): continue
                results.append({
                    "title": title, "title_raw": title_raw, "url": glink,
                    "snippet": snippet, "source": src_title, "source_url": src_href,
                    "source_domain": extract_domain(src_href) if src_href else src_title.lower(),
                    "published": parse_rss_date(entry.get("published", "")),
                    "found_via": "google_news_rss", "query_used": query,
                    "entry_id": entry.get("id", ""),
                })
                count += 1
            print(f"         → {count} relevant (of {len(feed.entries)} raw)")
        except Exception as e:
            print(f"         → Error: {e}")
    return results


# ---------------------------------------------------------------------------
# Source 2: WordPress.com Public API
# ---------------------------------------------------------------------------

def scrape_wordpress() -> list[dict]:
    results = []
    for q in ["BusinessDen", '"Aaron Kremer"']:
        print(f"  [WP] Searching: {q}")
        try:
            resp = requests.get(
                "https://public-api.wordpress.com/rest/v1.1/read/search",
                params={"q": q, "number": 20},
                headers={"User-Agent": USER_AGENT}, timeout=15,
            )
            resp.raise_for_status()
            posts = resp.json().get("posts", [])
            count = 0
            for p in posts:
                if not isinstance(p, dict): continue
                url = p.get("URL", "")
                if not url or is_excluded(url): continue
                content_lower = (str(p.get("content", "")) + str(p.get("excerpt", ""))).lower()
                if "businessden" not in content_lower and "aaron kremer" not in content_lower: continue
                raw_excerpt = strip_html(str(p.get("excerpt", "")) or str(p.get("content", "")))
                context = ""
                for s in re.split(r'(?<=[.!?])\s+', raw_excerpt):
                    if "businessden" in s.lower() or "aaron kremer" in s.lower():
                        context = s.strip(); break
                author_data = p.get("author")
                author_name = author_data.get("name", "") if isinstance(author_data, dict) else ""
                results.append({
                    "title": strip_html(str(p.get("title", ""))),
                    "title_raw": str(p.get("title", "")), "url": url,
                    "snippet": context or truncate(raw_excerpt),
                    "source": str(p.get("site_name", "")) or extract_domain(url),
                    "source_url": str(p.get("site_URL", "")),
                    "source_domain": extract_domain(str(p.get("site_URL", url))),
                    "published": p.get("date"), "found_via": "wordpress_api", "query_used": q,
                    "wp_site_id": p.get("site_ID"), "wp_post_id": p.get("ID"),
                    "wp_like_count": p.get("like_count", 0),
                    "wp_comment_count": p.get("comment_count", 0),
                    "author": author_name,
                    "tags": list((p.get("tags") or {}).keys())[:10],
                    "categories": list((p.get("categories") or {}).keys())[:10],
                })
                count += 1
            print(f"         → {count} mentions (of {len(posts)} results)")
        except Exception as e:
            print(f"         → Error: {e}")
    return results


# ---------------------------------------------------------------------------
# Source 3: Reddit RSS
# ---------------------------------------------------------------------------

def scrape_reddit() -> list[dict]:
    results = []
    for q in ['"BusinessDen"']:
        url = f"https://www.reddit.com/search.rss?q={requests.utils.quote(q)}&sort=new&t=week&limit=25"
        print(f"  [REDDIT] Searching: {q}")
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
            if resp.status_code != 200:
                print(f"           → HTTP {resp.status_code} (may be blocked from this IP)"); continue
            feed = feedparser.parse(resp.text)
            count = 0
            for entry in feed.entries:
                link = entry.get("link", "")
                if is_excluded(link): continue
                title = strip_html(entry.get("title", ""))
                content = strip_html(entry.get("summary", ""))
                sub_match = re.search(r'reddit\.com/r/(\w+)', link)
                subreddit = sub_match.group(1) if sub_match else ""
                results.append({
                    "title": title, "title_raw": entry.get("title", ""), "url": link,
                    "snippet": truncate(content),
                    "source": f"r/{subreddit}" if subreddit else "Reddit",
                    "source_url": f"https://www.reddit.com/r/{subreddit}" if subreddit else "https://www.reddit.com",
                    "source_domain": "reddit.com",
                    "published": parse_rss_date(entry.get("published", "")),
                    "found_via": "reddit_rss", "query_used": q,
                    "author": entry.get("author", ""), "subreddit": subreddit,
                })
                count += 1
            print(f"           → {count} results")
        except Exception as e:
            print(f"           → Error: {e}")
    return results


# ---------------------------------------------------------------------------
# Source 4: Medium RSS
# ---------------------------------------------------------------------------

def scrape_medium() -> list[dict]:
    results = []
    for tag in ["businessden", "denver-business", "denver-real-estate"]:
        url = f"https://medium.com/feed/tag/{tag}"
        print(f"  [MEDIUM] Tag: {tag}")
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                link = entry.get("link", "")
                if is_excluded(link): continue
                title = strip_html(entry.get("title", ""))
                content = strip_html(entry.get("summary", ""))
                if tag != "businessden" and "businessden" not in (title + content).lower(): continue
                results.append({
                    "title": title, "title_raw": entry.get("title", ""), "url": link,
                    "snippet": truncate(content),
                    "source": "Medium", "source_url": "https://medium.com",
                    "source_domain": "medium.com",
                    "published": parse_rss_date(entry.get("published", "")),
                    "found_via": "medium_rss", "query_used": f"tag:{tag}",
                    "author": entry.get("author", ""),
                    "tags": [t.get("term", "") for t in entry.get("tags", [])][:10],
                })
                count += 1
            print(f"           → {count} results (of {len(feed.entries)} in feed)")
        except Exception as e:
            print(f"           → Error: {e}")
    return results


# ---------------------------------------------------------------------------
# BusinessDen article ingestion
# ---------------------------------------------------------------------------

def ingest_bd_articles(data: dict):
    """Fetch BusinessDen RSS and store articles since March 1, 2026."""
    print("\n5. BusinessDen article ingestion")
    existing_urls = {a["url"] for a in data["bd_articles"]}
    new_count = 0

    for page in range(1, 20):  # Up to 200 articles
        url = f"https://businessden.com/feed/?paged={page}"
        feed = feedparser.parse(url)
        if not feed.entries:
            break

        reached_cutoff = False
        for entry in feed.entries:
            pub = parse_rss_date(entry.get("published", ""))
            if pub and pub < BD_ARTICLES_SINCE:
                reached_cutoff = True
                break

            link = entry.get("link", "")
            if link in existing_urls:
                continue

            title = strip_html(entry.get("title", ""))
            author = entry.get("author", "Unknown")
            categories = [t.get("term", "") for t in entry.get("tags", [])]
            summary = strip_html(entry.get("summary", ""))
            content = ""
            if entry.get("content"):
                content = strip_html(entry["content"][0].get("value", ""))

            data["bd_articles"].append({
                "url": link,
                "title": title,
                "author": author,
                "published": pub,
                "categories": categories,
                "summary": summary[:300],
                "content_preview": content[:500],
                "ingested": datetime.now(timezone.utc).isoformat(),
            })
            existing_urls.add(link)
            new_count += 1

        if reached_cutoff:
            break

    # Sort newest first
    data["bd_articles"].sort(key=lambda a: a.get("published") or "0000", reverse=True)
    print(f"   {new_count} new articles ingested. {len(data['bd_articles'])} total since March 1.")


# ---------------------------------------------------------------------------
# AI: Match mentions to BD articles
# ---------------------------------------------------------------------------

def match_mentions_to_articles(data: dict) -> dict[str, list[dict]]:
    """For each BD article, find mentions that reference it."""
    matches = {}
    for article in data["bd_articles"]:
        art_url = article["url"]
        art_title = article["title"].lower()
        # Extract key phrases from title (3+ word chunks)
        title_words = re.findall(r'\w+', art_title)
        key_phrases = []
        if len(title_words) >= 4:
            key_phrases.append(" ".join(title_words[:4]))
            key_phrases.append(" ".join(title_words[-4:]))
        if len(title_words) >= 3:
            key_phrases.append(" ".join(title_words[:3]))

        article_mentions = []
        for mention in data["mentions"]:
            m_text = (mention.get("title", "") + " " + mention.get("snippet", "")).lower()
            # Check for URL reference, title overlap, or key phrase match
            if art_url.lower() in m_text:
                article_mentions.append(mention)
            elif any(phrase in m_text for phrase in key_phrases if len(phrase) > 12):
                article_mentions.append(mention)

        if article_mentions:
            matches[art_url] = article_mentions

    return matches


# ---------------------------------------------------------------------------
# AI: Daily summary
# ---------------------------------------------------------------------------

def generate_daily_summary(data: dict, new_mentions: list[dict], now: datetime):
    if not new_mentions:
        print("\n  [AI] No new mentions — skipping daily summary")
        return

    # Check if we already generated one today
    today = now.strftime("%Y-%m-%d")
    existing_dates = [s["date"] for s in data["summaries"]["daily"]]
    if today in existing_dates:
        print(f"\n  [AI] Daily summary already exists for {today}")
        return

    print(f"\n  [AI] Generating daily summary for {today}...")

    mentions_text = ""
    for m in new_mentions[:40]:  # Cap context size
        mentions_text += f"- [{m['source']}] \"{m['title']}\""
        if m.get("snippet"):
            mentions_text += f" — {m['snippet'][:150]}"
        mentions_text += f" (via {m['found_via']})\n"

    prompt = f"""You are an analyst for BusinessDen, a Denver business news publication.
Today is {today}. Below are {len(new_mentions)} new third-party mentions of BusinessDen found today.

Provide a detailed daily summary covering:
1. Which outlets mentioned BusinessDen today (name each one)
2. Which BusinessDen stories/topics got picked up and by whom
3. Which reporters' work was referenced (Thomas Gounley, Justin Wingerter, Matt Geiger, Aaron Kremer, or others)
4. Any story getting outsized attention (multiple outlets covering it)
5. Notable themes or patterns

Be specific — name outlets, article topics, and reporters. Write 2-4 paragraphs.

TODAY'S MENTIONS:
{mentions_text}"""

    result = call_claude(prompt, max_tokens=1500)
    if result:
        data["summaries"]["daily"].append({
            "date": today,
            "text": result,
            "mention_count": len(new_mentions),
            "generated": now.isoformat(),
        })
        print(f"    → Daily summary generated ({len(result)} chars)")


# ---------------------------------------------------------------------------
# AI: Weekly summary (Mondays)
# ---------------------------------------------------------------------------

def generate_weekly_summary(data: dict, now: datetime):
    if now.weekday() != 0:  # Monday = 0
        return

    week_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    week_key = f"week-{week_start}"

    existing_keys = [s["key"] for s in data["summaries"]["weekly"]]
    if week_key in existing_keys:
        print(f"\n  [AI] Weekly summary already exists for {week_key}")
        return

    # Gather last 7 days of mentions
    cutoff = (now - timedelta(days=7)).isoformat()
    week_mentions = [m for m in data["mentions"] if (m.get("published") or "") >= cutoff]
    if not week_mentions:
        print("\n  [AI] No mentions in past week — skipping weekly summary")
        return

    print(f"\n  [AI] Generating weekly summary ({len(week_mentions)} mentions)...")

    mentions_text = ""
    for m in week_mentions[:60]:
        mentions_text += f"- [{m['source']}] \"{m['title']}\" ({m.get('published','')[:10]})\n"

    # Also include daily summaries from the week
    dailies_text = ""
    for d in data["summaries"]["daily"]:
        if d["date"] >= week_start:
            dailies_text += f"\n--- {d['date']} ---\n{d['text']}\n"

    prompt = f"""You are an analyst for BusinessDen, a Denver business news publication.
Write a detailed WEEKLY summary for the week of {week_start} to {now.strftime('%Y-%m-%d')}.

Cover:
1. Total mentions and which outlets mentioned BusinessDen most frequently
2. The biggest stories that got picked up — what topics dominated?
3. Reporter performance — whose bylines appeared most in other outlets?
4. Any trends: is coverage growing/shrinking? New outlets picking up BusinessDen?
5. Notable individual mentions worth highlighting

Write 4-6 detailed paragraphs. Be specific with outlet names, story topics, and numbers.

DAILY SUMMARIES FROM THIS WEEK:
{dailies_text}

ALL MENTIONS THIS WEEK ({len(week_mentions)} total):
{mentions_text}"""

    result = call_claude(prompt, max_tokens=2500)
    if result:
        data["summaries"]["weekly"].append({
            "key": week_key,
            "date": now.strftime("%Y-%m-%d"),
            "week_start": week_start,
            "text": result,
            "mention_count": len(week_mentions),
            "generated": now.isoformat(),
        })
        print(f"    → Weekly summary generated ({len(result)} chars)")


# ---------------------------------------------------------------------------
# AI: Monthly summary (1st of month)
# ---------------------------------------------------------------------------

def generate_monthly_summary(data: dict, now: datetime):
    if now.day != 1:
        return

    last_month = (now - timedelta(days=1))
    month_key = last_month.strftime("%Y-%m")
    month_start = last_month.replace(day=1).isoformat()

    existing_keys = [s["key"] for s in data["summaries"]["monthly"]]
    if month_key in existing_keys:
        print(f"\n  [AI] Monthly summary already exists for {month_key}")
        return

    month_mentions = [m for m in data["mentions"] if (m.get("published") or "")[:7] == month_key]
    if not month_mentions:
        print(f"\n  [AI] No mentions for {month_key} — skipping monthly summary")
        return

    print(f"\n  [AI] Generating monthly summary for {month_key} ({len(month_mentions)} mentions)...")

    # Source breakdown
    source_counts = {}
    for m in month_mentions:
        source_counts[m["source"]] = source_counts.get(m["source"], 0) + 1
    source_text = "\n".join(f"  {s}: {c}" for s, c in sorted(source_counts.items(), key=lambda x: -x[1])[:20])

    # Weekly summaries from the month
    weeklies_text = ""
    for w in data["summaries"]["weekly"]:
        if w.get("week_start", "")[:7] == month_key:
            weeklies_text += f"\n--- Week of {w['week_start']} ---\n{w['text']}\n"

    prompt = f"""You are an analyst for BusinessDen, a Denver business news publication.
Write a comprehensive MONTHLY summary for {last_month.strftime('%B %Y')}.

Cover:
1. Overall mention volume and trend (compare to what you see in the data)
2. Top outlets by mention count and any new outlets that started covering BusinessDen
3. The month's biggest stories — which BusinessDen reporting got the most traction?
4. Reporter-by-reporter breakdown of whose work was picked up most
5. Themes and patterns across the month
6. Recommendations or observations about BusinessDen's media footprint

Write 5-8 detailed paragraphs with specific numbers, outlet names, and story references.

SOURCE BREAKDOWN:
{source_text}

WEEKLY SUMMARIES:
{weeklies_text}

TOTAL MENTIONS: {len(month_mentions)}"""

    result = call_claude(prompt, max_tokens=3500)
    if result:
        data["summaries"]["monthly"].append({
            "key": month_key,
            "date": now.strftime("%Y-%m-%d"),
            "month": last_month.strftime("%B %Y"),
            "text": result,
            "mention_count": len(month_mentions),
            "generated": now.isoformat(),
        })
        print(f"    → Monthly summary generated ({len(result)} chars)")


# ---------------------------------------------------------------------------
# AI: Per-article reception summary
# ---------------------------------------------------------------------------

def generate_article_reception(data: dict, now: datetime):
    """Generate reception summaries at 2-day, 7-day, and 30-day marks."""
    if "article_reception" not in data:
        data["article_reception"] = {}

    matches = match_mentions_to_articles(data)
    generated = 0

    for article in data["bd_articles"]:
        art_url = article["url"]
        pub = article.get("published")
        if not pub:
            continue

        try:
            pub_dt = datetime.fromisoformat(pub)
        except:
            continue

        age_days = (now - pub_dt).days
        article_mentions = matches.get(art_url, [])

        # Only generate if the article was picked up elsewhere
        if not article_mentions:
            continue

        # Determine which tier(s) to generate
        reception = data["article_reception"].get(art_url, {})
        tiers = []
        if age_days >= 2 and "2day" not in reception:
            tiers.append(("2day", "2-day"))
        if age_days >= 7 and "7day" not in reception:
            tiers.append(("7day", "7-day"))
        if age_days >= 30 and "30day" not in reception:
            tiers.append(("30day", "30-day"))

        for tier_key, tier_label in tiers:
            print(f"  [AI] Article reception ({tier_label}): {article['title'][:60]}...")

            mentions_text = ""
            for m in article_mentions[:20]:
                mentions_text += f"- [{m['source']}] \"{m['title']}\" ({m.get('published','')[:10]})"
                if m.get("snippet"):
                    mentions_text += f" — {m['snippet'][:120]}"
                mentions_text += "\n"

            prompt = f"""You are an analyst for BusinessDen, a Denver business news publication.

Below is a BusinessDen article and the third-party mentions it received within {tier_label} of publication.

ARTICLE:
Title: {article['title']}
Author: {article['author']}
Published: {article['published'][:10]}
Categories: {', '.join(article.get('categories', []))}
Summary: {article.get('summary', '')[:200]}

THIRD-PARTY MENTIONS ({len(article_mentions)} total):
{mentions_text}

Write a brief summary (2-4 sentences) of how this article was received. Name which outlets picked it up, whether the coverage was substantial or just a brief mention, and any notable framing differences."""

            result = call_claude(prompt, max_tokens=500)
            if result:
                if art_url not in data["article_reception"]:
                    data["article_reception"][art_url] = {}
                data["article_reception"][art_url][tier_key] = {
                    "text": result,
                    "mention_count": len(article_mentions),
                    "generated": now.isoformat(),
                }
                generated += 1

    if generated:
        print(f"    → {generated} article reception summaries generated")


# ---------------------------------------------------------------------------
# AI: Byline analysis
# ---------------------------------------------------------------------------

def generate_byline_analysis(data: dict, now: datetime):
    """Generate per-reporter analysis of article performance."""
    today = now.strftime("%Y-%m-%d")

    # Only generate once per day
    existing_dates = [b["date"] for b in data.get("byline", [])]
    if today in existing_dates:
        print(f"\n  [AI] Byline analysis already exists for {today}")
        return

    matches = match_mentions_to_articles(data)
    reporter_data = {}

    for article in data["bd_articles"]:
        author = article.get("author", "")
        if not any(r.lower() in author.lower() for r in REPORTERS):
            continue

        art_url = article["url"]
        article_mentions = matches.get(art_url, [])
        reception = data.get("article_reception", {}).get(art_url, {})

        # Get the most recent reception summary
        latest_reception = ""
        for tier in ["30day", "7day", "2day"]:
            if tier in reception:
                latest_reception = reception[tier]["text"]
                break

        if not reporter_data.get(author):
            reporter_data[author] = []

        reporter_data[author].append({
            "title": article["title"],
            "published": article.get("published", "")[:10],
            "mention_count": len(article_mentions),
            "reception": latest_reception,
            "outlets": list(set(m["source"] for m in article_mentions))[:10],
        })

    if not reporter_data:
        print("\n  [AI] No reporter data for byline analysis")
        return

    print(f"\n  [AI] Generating byline analysis for {len(reporter_data)} reporters...")

    reporter_text = ""
    for author, articles in reporter_data.items():
        picked_up = [a for a in articles if a["mention_count"] > 0]
        reporter_text += f"\n## {author}\n"
        reporter_text += f"Total articles: {len(articles)} | Picked up by others: {len(picked_up)}\n"
        for a in picked_up[:10]:
            reporter_text += f"- \"{a['title']}\" ({a['published']}) — {a['mention_count']} mentions"
            if a["outlets"]:
                reporter_text += f" from {', '.join(a['outlets'][:5])}"
            reporter_text += "\n"
            if a["reception"]:
                reporter_text += f"  Reception: {a['reception'][:200]}\n"

    prompt = f"""You are an analyst for BusinessDen, a Denver business news publication.
Today is {today}. Write a byline analysis for each reporter below.

For each reporter, summarize:
1. How many of their recent articles were picked up by other outlets
2. Which articles got the most traction and where
3. What themes or beats are generating the most external interest
4. Any notable patterns in how their work is being covered

Only discuss articles that were picked up by other outlets. Write 1-2 paragraphs per reporter.

REPORTER DATA:
{reporter_text}"""

    result = call_claude(prompt, max_tokens=2000)
    if result:
        data["byline"].append({
            "date": today,
            "text": result,
            "reporters": list(reporter_data.keys()),
            "generated": now.isoformat(),
        })
        print(f"    → Byline analysis generated ({len(result)} chars)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def scrape():
    data = load_data()
    existing_ids = {m["id"] for m in data["mentions"]}
    new_mentions = []
    now = datetime.now(timezone.utc)

    print(f"Reputation scraper — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Existing: {len(data['mentions'])} mentions, {len(data.get('bd_articles', []))} BD articles\n")

    # Ensure data structure
    if "summaries" not in data or isinstance(data["summaries"], list):
        data["summaries"] = {"daily": [], "weekly": [], "monthly": []}
    if "bd_articles" not in data:
        data["bd_articles"] = []
    if "article_reception" not in data:
        data["article_reception"] = {}
    if "byline" not in data:
        data["byline"] = []
    if "run_log" not in data:
        data["run_log"] = []

    # --- Scrape mentions ---
    print("1. Google News RSS")
    gnews = scrape_gnews_rss()
    print(f"   Subtotal: {len(gnews)}")

    print("\n2. WordPress.com API")
    wp = scrape_wordpress()
    print(f"   Subtotal: {len(wp)}")

    print("\n3. Reddit RSS")
    reddit = scrape_reddit()
    print(f"   Subtotal: {len(reddit)}")

    print("\n4. Medium RSS")
    medium = scrape_medium()
    print(f"   Subtotal: {len(medium)}")

    all_results = gnews + wp + reddit + medium
    print(f"\nTotal results after source filtering: {len(all_results)}")

    # 48-hour cutoff
    cutoff_48h = (now - timedelta(hours=48)).isoformat()
    time_filtered = []
    skipped_old = 0
    for item in all_results:
        pub = item.get("published")
        if pub and pub < cutoff_48h:
            skipped_old += 1
            continue
        time_filtered.append(item)
    if skipped_old:
        print(f"Skipped {skipped_old} results older than 48 hours")
    print(f"Results within 48-hour window: {len(time_filtered)}")

    # Dedup and store
    for item in time_filtered:
        article_id = make_id(item["url"])
        if article_id in existing_ids:
            continue
        mention = {
            "id": article_id, "title": item["title"],
            "title_raw": item.get("title_raw", item["title"]),
            "url": item["url"], "snippet": item["snippet"],
            "source": item["source"], "source_url": item.get("source_url", ""),
            "source_domain": item["source_domain"],
            "published": item["published"], "found_via": item["found_via"],
            "query_used": item.get("query_used", ""),
            "first_seen": now.isoformat(),
            "author": item.get("author", ""), "subreddit": item.get("subreddit", ""),
            "tags": item.get("tags", []), "categories": item.get("categories", []),
            "wp_site_id": item.get("wp_site_id"), "wp_post_id": item.get("wp_post_id"),
            "wp_like_count": item.get("wp_like_count"),
            "wp_comment_count": item.get("wp_comment_count"),
            "entry_id": item.get("entry_id", ""),
        }
        data["mentions"].append(mention)
        existing_ids.add(article_id)
        new_mentions.append(mention)
        tag = f"[{item['found_via']}]"
        print(f"  NEW {tag:22s} {item['title'][:60]}  ({item['source']})")

    data["mentions"].sort(key=lambda m: m.get("published") or "0000", reverse=True)

    # --- Ingest BD articles ---
    ingest_bd_articles(data)

    # --- AI summaries ---
    print("\n6. AI summaries")
    generate_daily_summary(data, new_mentions, now)
    generate_weekly_summary(data, now)
    generate_monthly_summary(data, now)

    print("\n7. Article reception summaries")
    generate_article_reception(data, now)

    print("\n8. Byline analysis")
    generate_byline_analysis(data, now)

    # --- Metadata ---
    data["last_updated"] = now.isoformat()
    data["new_this_run"] = len(new_mentions)

    data["run_log"].append({
        "timestamp": now.isoformat(),
        "new_mentions": len(new_mentions),
        "total_mentions": len(data["mentions"]),
        "total_bd_articles": len(data["bd_articles"]),
        "sources_checked": {
            "google_news_rss": len(gnews), "wordpress_api": len(wp),
            "reddit_rss": len(reddit), "medium_rss": len(medium),
        },
    })
    cutoff_log = (now - timedelta(days=90)).isoformat()
    data["run_log"] = [r for r in data["run_log"] if r["timestamp"] >= cutoff_log]

    save_data(data)
    print(f"\nDone. {len(new_mentions)} new mentions. {len(data['mentions'])} total. "
          f"{len(data['bd_articles'])} BD articles tracked.")


if __name__ == "__main__":
    scrape()
