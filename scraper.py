#!/usr/bin/env python3
"""
Reputation v2 - Scraper
Finds third-party mentions of BusinessDen and analyzes their impact.

Sources (all free, no API keys):
  1. Google News RSS (multiple queries, 48hr window)
  2. WordPress.com Public API
  3. Reddit RSS
  4. Medium RSS
  5. BusinessDen own RSS (article ingestion)

AI features (requires ANTHROPIC_API_KEY):
  - Claude-powered mention-to-article matching
  - Sentiment tagging per mention
  - Daily summary (pithy, bulleted, newspaper-style)
  - Weekly summary (detailed)
  - Monthly summary (comprehensive)
  - Per-article reception at 2d/7d/30d
  - Byline analysis per reporter

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
from difflib import SequenceMatcher
from zoneinfo import ZoneInfo

MT = ZoneInfo("America/Denver")

def mt_today(now: datetime) -> str:
    """Return today's date string in Mountain Time."""
    return now.astimezone(MT).strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = "mentions-data.json"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GA4_SERVICE_ACCOUNT_KEY = os.environ.get("GA4_SERVICE_ACCOUNT_KEY", "")
GA4_PROPERTY_ID = os.environ.get("GA4_PROPERTY_ID", "")

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

CORE_REPORTERS = ["Thomas Gounley", "Justin Wingerter", "Matt Geiger", "Aaron Kremer"]
BD_ARTICLES_SINCE = "2026-03-01T00:00:00+00:00"
USER_AGENT = "ReputationTracker/1.0 (BusinessDen)"

RELEVANCE_KEYWORDS = [
    "businessden", "business den",
    "thomas gounley", "justin wingerter", "matt geiger", "aaron kremer",
    "denver", "colorado", "real estate", "commercial", "office",
    "restaurant", "brewery", "development", "developer",
]

# Model selection: Haiku for cheap tasks, Sonnet for complex analysis
MODEL_CHEAP = "claude-haiku-4-5-20251001"    # $1/$5 per MTok
MODEL_SMART = "claude-sonnet-4-20250514"     # $3/$15 per MTok

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

def title_similar(a: str, b: str, threshold: float = 0.6) -> bool:
    """Fuzzy title comparison for cross-source dedup."""
    a_clean = re.sub(r'[^\w\s]', '', a.lower()).strip()
    b_clean = re.sub(r'[^\w\s]', '', b.lower()).strip()
    return SequenceMatcher(None, a_clean, b_clean).ratio() >= threshold

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
        "mentions": [], "bd_articles": [],
        "summaries": {"daily": [], "weekly": [], "monthly": []},
        "byline": [], "article_reception": {},
        "article_matches": {},
        "last_updated": None, "run_log": [],
    }

def save_data(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def ensure_structure(data: dict):
    """Ensure all expected keys exist."""
    defaults = {
        "mentions": [], "bd_articles": [],
        "summaries": {"daily": [], "weekly": [], "monthly": []},
        "byline": [], "article_reception": {},
        "article_matches": {},
        "last_updated": None, "run_log": [],
    }
    for k, v in defaults.items():
        if k not in data:
            data[k] = v
    if isinstance(data["summaries"], list):
        data["summaries"] = {"daily": [], "weekly": [], "monthly": []}


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------

def call_claude(prompt: str, model: str = None, max_tokens: int = 2000) -> str | None:
    if not ANTHROPIC_API_KEY:
        print("    [AI] No ANTHROPIC_API_KEY — skipping")
        return None
    if model is None:
        model = MODEL_SMART
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]
    except Exception as e:
        print(f"    [AI] Error: {e}")
        return None

def call_claude_json(prompt: str, model: str = None, max_tokens: int = 2000) -> dict | None:
    """Call Claude and parse JSON response."""
    result = call_claude(prompt, model=model, max_tokens=max_tokens)
    if not result:
        return None
    # Extract JSON from response (Claude sometimes wraps in markdown)
    json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
    if json_match:
        result = json_match.group(1)
    # Also try raw
    try:
        return json.loads(result)
    except:
        # Try finding the first { ... } block
        brace_match = re.search(r'\{.*\}', result, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except:
                pass
    print(f"    [AI] Failed to parse JSON from response")
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
                print(f"           → HTTP {resp.status_code} (may work from GitHub Actions)"); continue
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
# Cross-source dedup
# ---------------------------------------------------------------------------

def dedup_results(results: list[dict], existing_mentions: list[dict]) -> list[dict]:
    """Remove duplicates by URL hash AND fuzzy title similarity."""
    existing_ids = {m["id"] for m in existing_mentions}
    existing_titles = [m["title"].lower() for m in existing_mentions]

    deduped = []
    seen_ids = set()
    seen_titles = []

    for item in results:
        aid = make_id(item["url"])

        # Skip if URL already in database
        if aid in existing_ids or aid in seen_ids:
            continue

        # Skip if title is very similar to an existing mention
        item_title_lower = item["title"].lower()
        is_dup = False
        for et in existing_titles + seen_titles:
            if title_similar(item_title_lower, et, threshold=0.75):
                is_dup = True
                break
        if is_dup:
            continue

        seen_ids.add(aid)
        seen_titles.append(item_title_lower)
        deduped.append(item)

    return deduped


# ---------------------------------------------------------------------------
# BusinessDen article ingestion
# ---------------------------------------------------------------------------

def ingest_bd_articles(data: dict) -> list[str]:
    """Fetch BusinessDen RSS, return list of all known reporters."""
    print("\n5. BusinessDen article ingestion")
    existing_urls = {a["url"] for a in data["bd_articles"]}
    new_count = 0
    all_authors = set()

    for page in range(1, 30):
        url = f"https://businessden.com/feed/?paged={page}"
        feed = feedparser.parse(url)
        if not feed.entries: break

        reached_cutoff = False
        for entry in feed.entries:
            pub = parse_rss_date(entry.get("published", ""))
            if pub and pub < BD_ARTICLES_SINCE:
                reached_cutoff = True; break

            author = entry.get("author", "Unknown")
            all_authors.add(author)

            link = entry.get("link", "")
            if link in existing_urls: continue

            title = strip_html(entry.get("title", ""))
            categories = [t.get("term", "") for t in entry.get("tags", [])]
            summary = strip_html(entry.get("summary", ""))
            content = ""
            if entry.get("content"):
                content = strip_html(entry["content"][0].get("value", ""))

            data["bd_articles"].append({
                "url": link, "title": title, "author": author,
                "published": pub, "categories": categories,
                "summary": summary[:300], "content_preview": content[:500],
                "ingested": datetime.now(timezone.utc).isoformat(),
            })
            existing_urls.add(link)
            new_count += 1

        if reached_cutoff: break

    data["bd_articles"].sort(key=lambda a: a.get("published") or "0000", reverse=True)
    print(f"   {new_count} new articles. {len(data['bd_articles'])} total.")
    print(f"   Reporters detected: {', '.join(sorted(all_authors))}")
    return sorted(all_authors)


# ---------------------------------------------------------------------------
# AI: Claude-powered mention-to-article matching (#14)
# ---------------------------------------------------------------------------

def ai_match_mentions_to_articles(data: dict, new_mentions: list[dict]):
    """Use Claude to match new mentions to BD articles."""
    if not new_mentions or not data["bd_articles"]:
        return

    print("\n6. AI mention-to-article matching")

    # Build compact article list
    articles_text = ""
    for i, a in enumerate(data["bd_articles"][:50]):
        articles_text += f"{i}|{a['title']}|{a['author']}|{a.get('published','')[:10]}|{a['url']}\n"

    # Build compact mention list
    mentions_text = ""
    for m in new_mentions[:40]:
        mentions_text += f"{m['id']}|{m['title']}|{m['source']}|{m.get('snippet','')[:100]}\n"

    prompt = f"""Match third-party mentions to the BusinessDen articles they reference.

BUSINESSDEN ARTICLES (index|title|author|date|url):
{articles_text}

NEW MENTIONS (id|title|source|snippet):
{mentions_text}

For each mention that references a specific BD article, output a JSON object mapping mention_id to article_url. Only include matches where the mention clearly references or covers the same story as the BD article. Output ONLY valid JSON like:
{{"mention_id_1": "article_url_1", "mention_id_2": "article_url_2"}}

If no matches found, output: {{}}"""

    result = call_claude_json(prompt, model=MODEL_CHEAP, max_tokens=1000)
    if result and isinstance(result, dict):
        if "article_matches" not in data:
            data["article_matches"] = {}
        for mention_id, article_url in result.items():
            if article_url not in data["article_matches"]:
                data["article_matches"][article_url] = []
            if mention_id not in data["article_matches"][article_url]:
                data["article_matches"][article_url].append(mention_id)
        print(f"    → {len(result)} matches found")
    else:
        print(f"    → No matches returned")


# ---------------------------------------------------------------------------
# AI: Sentiment tagging (#9)
# ---------------------------------------------------------------------------

def ai_tag_sentiment(data: dict, new_mentions: list[dict]):
    """Tag new mentions with sentiment using Haiku."""
    if not new_mentions:
        return

    print("\n7. AI sentiment tagging")

    batch_text = ""
    ids = []
    for m in new_mentions[:40]:
        batch_text += f"{m['id']}|{m['title']}|{m.get('snippet','')[:120]}\n"
        ids.append(m["id"])

    prompt = f"""Classify the sentiment of each mention of BusinessDen below as "positive", "neutral", or "negative".
Positive = praise, citation as authoritative source, amplification of their reporting.
Neutral = factual reference, syndication, passing mention.
Negative = criticism, correction, dispute of their reporting.

MENTIONS (id|title|snippet):
{batch_text}

Output ONLY valid JSON mapping id to sentiment like:
{{"id1": "positive", "id2": "neutral"}}"""

    result = call_claude_json(prompt, model=MODEL_CHEAP, max_tokens=800)
    if result and isinstance(result, dict):
        count = 0
        for m in data["mentions"]:
            if m["id"] in result:
                m["sentiment"] = result[m["id"]]
                count += 1
        print(f"    → {count} mentions tagged")
    else:
        print(f"    → Sentiment tagging failed")


# ---------------------------------------------------------------------------
# AI: Daily summary — pithy, bulleted, newspaper-style (#13)
# ---------------------------------------------------------------------------

def generate_daily_summary(data: dict, new_mentions: list[dict], now: datetime):
    if not new_mentions:
        print("\n  [AI] No new mentions — skipping daily summary")
        return

    today = mt_today(now)
    if today in [s["date"] for s in data["summaries"]["daily"]]:
        print(f"\n  [AI] Daily summary exists for {today}")
        return

    print(f"\n8. AI daily summary for {today}...")

    mentions_text = ""
    for m in new_mentions[:40]:
        sentiment = m.get("sentiment", "")
        mentions_text += f"• [{m['source']}] {m['title']}"
        if m.get("snippet"): mentions_text += f" — {m['snippet'][:100]}"
        if sentiment: mentions_text += f" ({sentiment})"
        mentions_text += "\n"

    prompt = f"""You are a media monitor for BusinessDen, a Denver business publication.
Today is {today}. Write a pithy daily summary of today's {len(new_mentions)} third-party mentions.

FORMAT: Write this like a wire-service brief — short, punchy, no filler.
• Use bullet points (start each with •)
• Name every outlet plainly — do NOT wrap in ** or any markdown formatting
• Put BusinessDen story titles in "quotes"
• Flag any story getting outsized pickup (3+ outlets)
• Note the reporter whose work was cited, if identifiable
• If coverage is negative or critical, say so explicitly
• 8 bullet points maximum
• No preamble, no sign-off, no markdown formatting

TODAY'S MENTIONS:
{mentions_text}"""

    result = call_claude(prompt, model=MODEL_SMART, max_tokens=800)
    if result:
        data["summaries"]["daily"].append({
            "date": today, "text": result,
            "mention_count": len(new_mentions),
            "generated": now.isoformat(),
        })
        print(f"    → Generated ({len(result)} chars)")


# ---------------------------------------------------------------------------
# AI: Weekly summary (Mondays) — detailed
# ---------------------------------------------------------------------------

def generate_weekly_summary(data: dict, now: datetime):
    mt_now = now.astimezone(MT)
    if mt_now.weekday() != 0: return

    week_start = (mt_now - timedelta(days=7)).strftime("%Y-%m-%d")
    week_key = f"week-{week_start}"
    if week_key in [s["key"] for s in data["summaries"]["weekly"]]:
        print(f"\n  [AI] Weekly summary exists for {week_key}")
        return

    cutoff = (now - timedelta(days=7)).isoformat()
    week_mentions = [m for m in data["mentions"] if (m.get("published") or "") >= cutoff]
    if not week_mentions:
        print("\n  [AI] No mentions in past week")
        return

    print(f"\n9. AI weekly summary ({len(week_mentions)} mentions)...")

    mentions_text = ""
    for m in week_mentions[:60]:
        mentions_text += f"• [{m['source']}] {m['title']} ({m.get('published','')[:10]}) [{m.get('sentiment','')}]\n"

    dailies_text = ""
    for d in data["summaries"]["daily"]:
        if d["date"] >= week_start:
            dailies_text += f"\n--- {d['date']} ---\n{d['text']}\n"

    prompt = f"""You are a media analyst for BusinessDen, a Denver business publication.
Write a detailed WEEKLY summary for {week_start} to {mt_today(now)}.

Cover in 4-6 paragraphs:
1. Total mention volume and top outlets by count
2. Biggest stories that got picked up — topics that dominated
3. Reporter-by-reporter breakdown of whose work was referenced most
4. Sentiment breakdown — mostly positive/neutral/negative?
5. Any new outlets that started covering BusinessDen
6. Trends or patterns worth noting

Be specific with outlet names, story topics, and numbers.

DAILY SUMMARIES:
{dailies_text}

ALL MENTIONS ({len(week_mentions)}):
{mentions_text}"""

    result = call_claude(prompt, model=MODEL_SMART, max_tokens=2500)
    if result:
        data["summaries"]["weekly"].append({
            "key": week_key, "date": mt_today(now),
            "week_start": week_start, "text": result,
            "mention_count": len(week_mentions),
            "generated": now.isoformat(),
        })
        print(f"    → Generated ({len(result)} chars)")


# ---------------------------------------------------------------------------
# AI: Monthly summary (1st of month) — comprehensive
# ---------------------------------------------------------------------------

def generate_monthly_summary(data: dict, now: datetime):
    mt_now = now.astimezone(MT)
    if mt_now.day != 1: return

    last_month = mt_now - timedelta(days=1)
    month_key = last_month.strftime("%Y-%m")
    if month_key in [s["key"] for s in data["summaries"]["monthly"]]:
        print(f"\n  [AI] Monthly summary exists for {month_key}")
        return

    month_mentions = [m for m in data["mentions"] if (m.get("published") or "")[:7] == month_key]
    if not month_mentions:
        print(f"\n  [AI] No mentions for {month_key}")
        return

    print(f"\n10. AI monthly summary for {month_key} ({len(month_mentions)} mentions)...")

    source_counts = {}
    for m in month_mentions:
        source_counts[m["source"]] = source_counts.get(m["source"], 0) + 1
    source_text = "\n".join(f"  {s}: {c}" for s, c in sorted(source_counts.items(), key=lambda x: -x[1])[:20])

    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for m in month_mentions:
        s = m.get("sentiment", "neutral")
        if s in sentiment_counts: sentiment_counts[s] += 1

    weeklies_text = ""
    for w in data["summaries"]["weekly"]:
        if w.get("week_start", "")[:7] == month_key:
            weeklies_text += f"\n--- Week of {w['week_start']} ---\n{w['text']}\n"

    prompt = f"""You are a media analyst for BusinessDen, a Denver business publication.
Write a comprehensive MONTHLY analysis for {last_month.strftime('%B %Y')}.

Cover in 5-8 paragraphs:
1. Total mentions ({len(month_mentions)}) and trend analysis
2. Top outlets: {source_text}
3. Sentiment: {sentiment_counts}
4. Biggest stories and which BD reporting got the most traction
5. Reporter-by-reporter analysis
6. New outlets that picked up BD coverage
7. Themes and patterns
8. Observations about BusinessDen's media footprint

WEEKLY SUMMARIES:
{weeklies_text}"""

    result = call_claude(prompt, model=MODEL_SMART, max_tokens=3500)
    if result:
        data["summaries"]["monthly"].append({
            "key": month_key, "date": mt_today(now),
            "month": last_month.strftime("%B %Y"), "text": result,
            "mention_count": len(month_mentions),
            "generated": now.isoformat(),
        })
        print(f"    → Generated ({len(result)} chars)")


# ---------------------------------------------------------------------------
# AI: Per-article reception
# ---------------------------------------------------------------------------

def generate_article_reception(data: dict, now: datetime):
    """Generate at 2d/7d/30d marks for articles picked up elsewhere."""
    if "article_reception" not in data: data["article_reception"] = {}
    matches = data.get("article_matches", {})
    mention_lookup = {m["id"]: m for m in data["mentions"]}
    generated = 0

    for article in data["bd_articles"]:
        art_url = article["url"]
        pub = article.get("published")
        if not pub: continue
        try: pub_dt = datetime.fromisoformat(pub)
        except: continue

        age_days = (now - pub_dt).days

        # Use Claude matches if available, otherwise fall back to keyword matching
        matched_ids = matches.get(art_url, [])
        article_mentions = [mention_lookup[mid] for mid in matched_ids if mid in mention_lookup]

        if not article_mentions:
            title_words = re.findall(r'\w+', article["title"].lower())
            key_phrases = []
            if len(title_words) >= 4:
                key_phrases.append(" ".join(title_words[:4]))
            if len(title_words) >= 3:
                key_phrases.append(" ".join(title_words[:3]))
            for m in data["mentions"]:
                mt = (m.get("title", "") + " " + m.get("snippet", "")).lower()
                if any(p in mt for p in key_phrases if len(p) > 12):
                    article_mentions.append(m)

        if not article_mentions: continue

        reception = data["article_reception"].get(art_url, {})
        tiers = []
        if age_days >= 2 and "2day" not in reception: tiers.append(("2day", "2-day"))
        if age_days >= 7 and "7day" not in reception: tiers.append(("7day", "7-day"))
        if age_days >= 30 and "30day" not in reception: tiers.append(("30day", "30-day"))

        for tier_key, tier_label in tiers:
            print(f"  [AI] Reception ({tier_label}): {article['title'][:55]}...")

            mentions_text = ""
            for m in article_mentions[:20]:
                mentions_text += f"• [{m['source']}] {m['title']}"
                if m.get("snippet"): mentions_text += f" — {m['snippet'][:100]}"
                mentions_text += f" [{m.get('sentiment','')}]\n"

            prompt = f"""Summarize the {tier_label} reception of this BusinessDen article in 2-4 sentences.
Name which outlets picked it up, whether coverage was substantial or a brief mention, and note any framing differences.

ARTICLE: "{article['title']}" by {article['author']} ({article['published'][:10]})
Categories: {', '.join(article.get('categories', []))}

THIRD-PARTY MENTIONS ({len(article_mentions)}):
{mentions_text}"""

            result = call_claude(prompt, model=MODEL_CHEAP, max_tokens=400)
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
        print(f"    → {generated} reception summaries generated")


# ---------------------------------------------------------------------------
# AI: Byline analysis (#5, #12)
# ---------------------------------------------------------------------------

def generate_byline_analysis(data: dict, now: datetime, all_reporters: list[str]):
    """Per-reporter analysis. Uses all detected reporters, not just hardcoded."""
    today = mt_today(now)
    if today in [b["date"] for b in data.get("byline", [])]:
        print(f"\n  [AI] Byline analysis exists for {today}")
        return

    matches = data.get("article_matches", {})
    mention_lookup = {m["id"]: m for m in data["mentions"]}
    reporter_data = {}

    for article in data["bd_articles"]:
        author = article.get("author", "")
        if not author: continue

        art_url = article["url"]

        # Use Claude matches if available, otherwise fall back to keyword matching
        matched_ids = matches.get(art_url, [])
        article_mentions = [mention_lookup[mid] for mid in matched_ids if mid in mention_lookup]

        # Fallback: keyword matching if no Claude matches exist
        if not article_mentions and not matches:
            title_words = re.findall(r'\w+', article["title"].lower())
            key_phrases = []
            if len(title_words) >= 4:
                key_phrases.append(" ".join(title_words[:4]))
            if len(title_words) >= 3:
                key_phrases.append(" ".join(title_words[:3]))
            for m in data["mentions"]:
                mt = (m.get("title", "") + " " + m.get("snippet", "")).lower()
                if any(p in mt for p in key_phrases if len(p) > 12):
                    article_mentions.append(m)

        if author not in reporter_data:
            reporter_data[author] = {"total": 0, "picked_up": 0, "articles": []}
        reporter_data[author]["total"] += 1

        if article_mentions:
            reporter_data[author]["picked_up"] += 1
            reception = data.get("article_reception", {}).get(art_url, {})
            latest_reception = ""
            for tier in ["30day", "7day", "2day"]:
                if tier in reception: latest_reception = reception[tier]["text"]; break

            reporter_data[author]["articles"].append({
                "title": article["title"],
                "published": article.get("published", "")[:10],
                "mention_count": len(article_mentions),
                "outlets": list(set(m["source"] for m in article_mentions))[:8],
                "reception": latest_reception,
            })

    # Only include reporters with at least one pickup
    active_reporters = {k: v for k, v in reporter_data.items() if v["picked_up"] > 0}
    if not active_reporters:
        print("\n  [AI] No reporters with pickups — skipping byline")
        return

    print(f"\n12. AI byline analysis for {len(active_reporters)} reporters...")

    reporter_text = ""
    for author, rd in sorted(active_reporters.items()):
        reporter_text += f"\n## {author}\n"
        reporter_text += f"Articles: {rd['total']} total, {rd['picked_up']} picked up by others\n"
        for a in rd["articles"][:8]:
            reporter_text += f"• \"{a['title']}\" ({a['published']}) — {a['mention_count']} mentions from {', '.join(a['outlets'][:4])}\n"
            if a["reception"]: reporter_text += f"  Reception: {a['reception'][:200]}\n"

    prompt = f"""You are a media analyst for BusinessDen, a Denver business publication.
Today is {today}. Write a byline analysis for each reporter below.

For each reporter with pickups:
1. How many articles were picked up vs. total published
2. Which articles got the most traction and where
3. What beats/themes generate the most external interest
4. Notable patterns

Write 1-2 focused paragraphs per reporter. Only discuss articles picked up by other outlets.

{reporter_text}"""

    result = call_claude(prompt, model=MODEL_SMART, max_tokens=2000)
    if result:
        data["byline"].append({
            "date": today, "text": result,
            "reporters": list(active_reporters.keys()),
            "generated": now.isoformat(),
        })
        print(f"    → Generated ({len(result)} chars)")


# ---------------------------------------------------------------------------
# GA4: Article pageviews and subscription attribution
# ---------------------------------------------------------------------------

def fetch_ga4_data(data: dict, now: datetime):
    """Pull per-article pageviews, subscriptions, and daily breakdowns from GA4."""
    if not GA4_SERVICE_ACCOUNT_KEY or not GA4_PROPERTY_ID:
        print("\n  [GA4] No GA4 credentials — skipping")
        return

    print("\n13. GA4 article analytics")

    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request
    except ImportError:
        print("    [GA4] google-auth not installed — skipping")
        return

    try:
        key_data = json.loads(GA4_SERVICE_ACCOUNT_KEY)
        credentials = service_account.Credentials.from_service_account_info(
            key_data, scopes=["https://www.googleapis.com/auth/analytics.readonly"]
        )
        credentials.refresh(Request())
        token = credentials.token
        url = f"https://analyticsdata.googleapis.com/v1beta/properties/{GA4_PROPERTY_ID}:runReport"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    except Exception as e:
        print(f"    [GA4] Auth error: {e}")
        return

    def ga4_query(payload: dict) -> dict | None:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"    [GA4] Query error: {e}")
            return None

    # --- Query 1: Per-article pageviews (last 30 days, aggregate) ---
    pageviews = {}
    result = ga4_query({
        "dateRanges": [{"startDate": "30daysAgo", "endDate": "today"}],
        "dimensions": [{"name": "pagePath"}],
        "metrics": [
            {"name": "screenPageViews"},
            {"name": "totalUsers"},
            {"name": "sessions"},
        ],
        "dimensionFilter": {
            "filter": {"fieldName": "pagePath", "stringFilter": {"matchType": "CONTAINS", "value": "/20"}}
        },
        "orderBys": [{"metric": {"metricName": "screenPageViews"}, "desc": True}],
        "limit": 500
    })
    if result:
        for row in result.get("rows", []):
            path = row["dimensionValues"][0]["value"]
            pageviews[path] = {
                "views": int(row["metricValues"][0]["value"]),
                "users": int(row["metricValues"][1]["value"]),
                "sessions": int(row["metricValues"][2]["value"]),
            }
        print(f"    → {len(pageviews)} article paths with pageview data")

    # --- Query 2: Subscription events with landing page (aggregate) ---
    subscriptions = {}
    result = ga4_query({
        "dateRanges": [{"startDate": "2026-03-01", "endDate": "today"}],
        "dimensions": [{"name": "landingPage"}],
        "metrics": [{"name": "eventCount"}],
        "dimensionFilter": {
            "filter": {"fieldName": "eventName", "stringFilter": {"matchType": "EXACT", "value": "subscription"}}
        },
        "limit": 500
    })
    if result:
        for row in result.get("rows", []):
            path = row["dimensionValues"][0]["value"]
            subscriptions[path] = int(row["metricValues"][0]["value"])
        print(f"    → {len(subscriptions)} landing pages with subscription events")
        print(f"    → {sum(subscriptions.values())} total subscription events")

    # --- Query 3: Daily subscriptions per landing page (time series) ---
    daily_subs = {}  # {path: {date: count}}
    result = ga4_query({
        "dateRanges": [{"startDate": "2026-03-01", "endDate": "today"}],
        "dimensions": [{"name": "date"}, {"name": "landingPage"}],
        "metrics": [{"name": "eventCount"}],
        "dimensionFilter": {
            "filter": {"fieldName": "eventName", "stringFilter": {"matchType": "EXACT", "value": "subscription"}}
        },
        "limit": 5000
    })
    if result:
        rows = result.get("rows", [])
        print(f"    → Query 3 returned {len(rows)} rows")
        for row in rows:
            d = row["dimensionValues"][0]["value"]  # YYYYMMDD
            date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            path = row["dimensionValues"][1]["value"]
            count = int(row["metricValues"][0]["value"])
            if path not in daily_subs:
                daily_subs[path] = {}
            daily_subs[path][date_str] = daily_subs[path].get(date_str, 0) + count
        print(f"    → {len(daily_subs)} paths with daily subscription data")
    else:
        print("    → Query 3 returned no result")

    # --- Query 4: Daily pageviews per article (time series) ---
    daily_views = {}  # {path: {date: count}}
    result = ga4_query({
        "dateRanges": [{"startDate": "2026-03-01", "endDate": "today"}],
        "dimensions": [{"name": "date"}, {"name": "pagePath"}],
        "metrics": [{"name": "screenPageViews"}],
        "dimensionFilter": {
            "filter": {"fieldName": "pagePath", "stringFilter": {"matchType": "CONTAINS", "value": "/20"}}
        },
        "limit": 10000
    })
    if result:
        rows = result.get("rows", [])
        print(f"    → Query 4 returned {len(rows)} rows")
        for row in rows:
            d = row["dimensionValues"][0]["value"]
            date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            path = row["dimensionValues"][1]["value"]
            count = int(row["metricValues"][0]["value"])
            if path not in daily_views:
                daily_views[path] = {}
            daily_views[path][date_str] = daily_views[path].get(date_str, 0) + count
        print(f"    → {len(daily_views)} paths with daily pageview data")
    else:
        print("    → Query 4 returned no result")

    # --- Query 5: Daily total pageviews (all pages) for traffic breakdown chart ---
    daily_totals = {}  # {date: total}
    result = ga4_query({
        "dateRanges": [{"startDate": "30daysAgo", "endDate": "today"}],
        "dimensions": [{"name": "date"}],
        "metrics": [{"name": "screenPageViews"}],
        "limit": 50
    })
    if result:
        for row in result.get("rows", []):
            d = row["dimensionValues"][0]["value"]
            date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            daily_totals[date_str] = int(row["metricValues"][0]["value"])
        print(f"    → {len(daily_totals)} days with total pageview data")

    # --- Query 6: Daily pageviews for homepage/landing pages ---
    daily_home = {}  # {date: count}
    result = ga4_query({
        "dateRanges": [{"startDate": "30daysAgo", "endDate": "today"}],
        "dimensions": [{"name": "date"}],
        "metrics": [{"name": "screenPageViews"}],
        "dimensionFilter": {
            "filter": {"fieldName": "pagePath", "stringFilter": {"matchType": "EXACT", "value": "/"}}
        },
        "limit": 50
    })
    if result:
        for row in result.get("rows", []):
            d = row["dimensionValues"][0]["value"]
            date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            daily_home[date_str] = int(row["metricValues"][0]["value"])
        print(f"    → {len(daily_home)} days with homepage pageview data")

    # --- Query 7: Top pages by pageviews (all pages, for "other" breakdown) ---
    all_pages = {}  # {path: views}
    result = ga4_query({
        "dateRanges": [{"startDate": "30daysAgo", "endDate": "today"}],
        "dimensions": [{"name": "pagePath"}, {"name": "pageTitle"}],
        "metrics": [{"name": "screenPageViews"}, {"name": "totalUsers"}],
        "orderBys": [{"metric": {"metricName": "screenPageViews"}, "desc": True}],
        "limit": 500
    })
    if result:
        for row in result.get("rows", []):
            path = row["dimensionValues"][0]["value"]
            title = row["dimensionValues"][1]["value"]
            views = int(row["metricValues"][0]["value"])
            users = int(row["metricValues"][1]["value"])
            all_pages[path] = {"title": title, "views": views, "users": users}
        print(f"    → {len(all_pages)} total pages with pageview data")

    # --- Query 8: Daily unique article readers (totalUsers on article pages only) ---
    daily_readers = {}  # {date: users}
    result = ga4_query({
        "dateRanges": [{"startDate": "60daysAgo", "endDate": "today"}],
        "dimensions": [{"name": "date"}],
        "metrics": [{"name": "totalUsers"}],
        "dimensionFilter": {
            "filter": {"fieldName": "pagePath", "stringFilter": {"matchType": "PARTIAL_REGEXP", "value": "^/[0-9]{4}/[0-9]{2}/[0-9]{2}/"}}
        },
        "limit": 70
    })
    if result:
        for row in result.get("rows", []):
            d = row["dimensionValues"][0]["value"]
            date_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            daily_readers[date_str] = int(row["metricValues"][0]["value"])
        print(f"    → {len(daily_readers)} days with daily article reader data")

    # --- Fetch newsletter subscriber history from subscriber dashboard ---
    newsletter_history = {}
    try:
        snap_resp = requests.get("https://businessden.github.io/subscriber/data/snapshots.json", timeout=15)
        snap_resp.raise_for_status()
        snap_data = snap_resp.json()
        for snap in snap_data.get("snapshots", []):
            if snap.get("date") and snap.get("active_free") is not None:
                newsletter_history[snap["date"]] = snap["active_free"]
        print(f"    → {len(newsletter_history)} days of newsletter subscriber history")
    except Exception as e:
        print(f"    [Newsletter] Error fetching subscriber data: {e}")

    # --- Match GA4 data to BD articles ---
    if "ga4" not in data:
        data["ga4"] = {}

    data["ga4"]["last_fetched"] = now.isoformat()
    data["ga4"]["article_stats"] = {}

    def match_path(art_url: str, lookup: dict) -> dict | None:
        try:
            path = urlparse(art_url).path.rstrip("/")
        except:
            return None
        for p in [path + "/", path]:
            if p in lookup:
                return lookup[p]
        return None

    matched = 0
    for article in data["bd_articles"]:
        art_url = article["url"]
        stats = {}

        pv = match_path(art_url, pageviews)
        if pv:
            stats.update(pv)

        try:
            path = urlparse(art_url).path.rstrip("/")
        except:
            continue

        for sub_path, count in subscriptions.items():
            if path == sub_path.rstrip("/"):
                stats["subscriptions"] = count

        # Daily time series
        ds = match_path(art_url, daily_subs)
        if ds:
            stats["daily_subs"] = ds

        dv = match_path(art_url, daily_views)
        if dv:
            stats["daily_views"] = dv

        if stats:
            data["ga4"]["article_stats"][art_url] = stats
            matched += 1

    print(f"    → {matched} BD articles matched with GA4 data")

    # --- Build daily traffic breakdown for stacked area chart ---
    # Classify article pageviews by article age relative to each day
    article_pub_dates = {}
    for article in data["bd_articles"]:
        try:
            path = urlparse(article["url"]).path.rstrip("/")
            pub = (article.get("published") or "")[:10]
            if pub:
                article_pub_dates[path] = pub
                article_pub_dates[path + "/"] = pub
        except:
            pass

    # Compute aggregate browse and newsletter totals from all_pages for proportional split
    browse_total = 0
    newsletter_total = 0
    other_non_article_total = 0
    for path, info in all_pages.items():
        if path == "/":
            continue
        if re.match(r'^/\d{4}/\d{2}/\d{2}/', path):
            continue  # article
        if path in article_pub_dates or path.rstrip("/") in article_pub_dates:
            continue
        v = info["views"]
        if any(x in path for x in ["/category/", "/tag/", "/author/", "/page/", "/recent-stories"]):
            browse_total += v
        elif any(x in path.lower() for x in ["newsletter", "sign-up", "signup"]):
            newsletter_total += v
        else:
            other_non_article_total += v
    non_article_total = browse_total + newsletter_total + other_non_article_total

    traffic_breakdown = {}
    for date_str in sorted(daily_totals.keys()):
        total = daily_totals.get(date_str, 0)
        home = daily_home.get(date_str, 0)

        today_views = 0
        week_views = 0
        month_views = 0
        older_views = 0

        # Sum article views by age bucket
        for path, date_counts in daily_views.items():
            day_count = date_counts.get(date_str, 0)
            if not day_count:
                continue
            pub = article_pub_dates.get(path)
            # If not a tracked article, try to extract date from URL pattern /YYYY/MM/DD/
            if not pub:
                url_date_match = re.match(r'^/(\d{4})/(\d{2})/(\d{2})/', path)
                if url_date_match:
                    pub = f"{url_date_match.group(1)}-{url_date_match.group(2)}-{url_date_match.group(3)}"
                else:
                    continue  # not an article-pattern URL
            # Calculate article age on this day
            try:
                from datetime import date as date_type
                view_date = date_type.fromisoformat(date_str)
                pub_date = date_type.fromisoformat(pub)
                age = (view_date - pub_date).days
            except:
                continue

            if age == 0:
                today_views += day_count
            elif age <= 7:
                week_views += day_count
            elif age <= 30:
                month_views += day_count
            else:
                older_views += day_count

        article_total = today_views + week_views + month_views + older_views
        remainder = max(0, total - home - article_total)

        # Split remainder proportionally into browse, newsletter, other
        if non_article_total > 0:
            browse_views = round(remainder * browse_total / non_article_total)
            newsletter_views = round(remainder * newsletter_total / non_article_total)
            other_views = max(0, remainder - browse_views - newsletter_views)
        else:
            browse_views = 0
            newsletter_views = 0
            other_views = remainder

        traffic_breakdown[date_str] = {
            "total": total,
            "today": today_views,
            "week": week_views,
            "month": month_views,
            "older": older_views,
            "home": home,
            "browse": browse_views,
            "newsletter": newsletter_views,
            "other": other_views,
        }

    data["ga4"]["traffic_breakdown"] = traffic_breakdown
    data["ga4"]["daily_readers"] = daily_readers

    # Compute articles per reader per day from traffic breakdown and daily readers
    articles_per_reader = {}
    for date_str, tb in traffic_breakdown.items():
        article_views = tb.get("today", 0) + tb.get("week", 0) + tb.get("month", 0) + tb.get("older", 0)
        readers = daily_readers.get(date_str, 0)
        if readers > 0:
            articles_per_reader[date_str] = round(article_views / readers, 2)
    data["ga4"]["articles_per_reader"] = articles_per_reader
    if articles_per_reader:
        recent = sorted(articles_per_reader.items())[-1]
        print(f"    → Articles per reader: {recent[1]} ({recent[0]})")

    # Store newsletter subscriber history
    data["ga4"]["newsletter_history"] = newsletter_history
    if traffic_breakdown:
        print(f"    → {len(traffic_breakdown)} days of traffic breakdown data")

    # --- Classify all pages into categories for "other" breakdown ---
    article_paths = set()
    for article in data["bd_articles"]:
        try:
            p = urlparse(article["url"]).path.rstrip("/")
            article_paths.add(p)
            article_paths.add(p + "/")
        except:
            pass

    non_article_pages = []
    for path, info in all_pages.items():
        if path == "/" or path in article_paths:
            continue
        # Skip article-pattern URLs (these are news content, just not tracked)
        if re.match(r'^/\d{4}/\d{2}/\d{2}/', path):
            continue
        # Classify the page type
        ptype = "other"
        if "/category/" in path:
            ptype = "category"
        elif "/tag/" in path:
            ptype = "tag"
        elif "/author/" in path:
            ptype = "author"
        elif "/page/" in path:
            ptype = "pagination"
        elif path in ("/feed/", "/feed", "/rss/", "/rss"):
            ptype = "feed"
        elif any(x in path for x in ["/wp-", "/login", "/my-account", "/account"]):
            ptype = "account"
        elif any(x in path for x in ["/subscribe", "/membership", "/pricing", "/checkout"]):
            ptype = "subscribe"
        elif any(x in path for x in ["/about", "/contact", "/advertise", "/privacy", "/terms"]):
            ptype = "static"
        elif any(x in path for x in ["/search", "?s="]):
            ptype = "search"

        non_article_pages.append({
            "path": path,
            "title": info["title"],
            "views": info["views"],
            "users": info["users"],
            "type": ptype,
        })

    non_article_pages.sort(key=lambda x: -x["views"])
    data["ga4"]["non_article_pages"] = non_article_pages[:100]

    # Aggregate by type
    type_totals = {}
    for p in non_article_pages:
        t = p["type"]
        type_totals[t] = type_totals.get(t, 0) + p["views"]
    data["ga4"]["other_breakdown"] = dict(sorted(type_totals.items(), key=lambda x: -x[1]))

    if non_article_pages:
        print(f"    → {len(non_article_pages)} non-article pages classified")
        for t, v in sorted(type_totals.items(), key=lambda x: -x[1]):
            print(f"       {t}: {v:,} views")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def scrape():
    data = load_data()
    ensure_structure(data)
    new_mentions = []
    now = datetime.now(timezone.utc)

    print(f"Reputation v2 — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Existing: {len(data['mentions'])} mentions, {len(data['bd_articles'])} BD articles\n")

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
    print(f"\nTotal raw: {len(all_results)}")

    # 48-hour cutoff
    cutoff_48h = (now - timedelta(hours=48)).isoformat()
    time_filtered = [item for item in all_results if not item.get("published") or item["published"] >= cutoff_48h]
    skipped = len(all_results) - len(time_filtered)
    if skipped: print(f"Skipped {skipped} older than 48 hours")

    # Cross-source dedup (#2)
    deduped = dedup_results(time_filtered, data["mentions"])
    print(f"After dedup: {len(deduped)} new mentions")

    # Store new mentions
    for item in deduped:
        mention = {
            "id": make_id(item["url"]),
            "title": item["title"], "title_raw": item.get("title_raw", item["title"]),
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
            "sentiment": "",  # Will be filled by AI
        }
        data["mentions"].append(mention)
        new_mentions.append(mention)
        tag = f"[{item['found_via']}]"
        print(f"  NEW {tag:22s} {item['title'][:60]}  ({item['source']})")

    data["mentions"].sort(key=lambda m: m.get("published") or "0000", reverse=True)

    # --- Ingest BD articles ---
    all_reporters = ingest_bd_articles(data)

    # --- AI pipeline ---
    ai_match_mentions_to_articles(data, new_mentions)   # #14
    ai_tag_sentiment(data, new_mentions)                 # #9
    generate_daily_summary(data, new_mentions, now)      # #13
    generate_weekly_summary(data, now)
    generate_monthly_summary(data, now)

    print("\n11. Article reception summaries")
    # generate_article_reception removed — Our Stories uses stat cards instead

    # generate_byline_analysis removed — byline tab uses data directly

    # --- GA4 Analytics ---
    fetch_ga4_data(data, now)

    # --- Metadata ---
    data["last_updated"] = now.isoformat()
    data["new_this_run"] = len(new_mentions)

    data["run_log"].append({
        "timestamp": now.isoformat(),
        "new_mentions": len(new_mentions),
        "total_mentions": len(data["mentions"]),
        "total_bd_articles": len(data["bd_articles"]),
        "sources": {
            "google_news_rss": len(gnews), "wordpress_api": len(wp),
            "reddit_rss": len(reddit), "medium_rss": len(medium),
        },
    })
    cutoff_log = (now - timedelta(days=90)).isoformat()
    data["run_log"] = [r for r in data["run_log"] if r["timestamp"] >= cutoff_log]

    save_data(data)
    print(f"\nDone. {len(new_mentions)} new. {len(data['mentions'])} total. "
          f"{len(data['bd_articles'])} BD articles. "
          f"{len(data.get('article_matches', {}))} articles with matches.")


if __name__ == "__main__":
    scrape()
