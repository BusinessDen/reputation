#!/usr/bin/env python3
"""
Reputation - Scraper
Finds third-party mentions of BusinessDen across the web using free sources:
  1. Google News RSS (multiple query strategies, last 48 hours)
  2. WordPress.com Public API (searches across all WP-powered sites)
  3. Reddit RSS (search feed)
  4. Medium RSS (tag feed)

All results hosted on businessden.com are excluded.
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

EXCLUDED_DOMAINS = [
    "businessden.com",
    "www.businessden.com",
]

# Spam domains that show up in broad Google News RSS results
SPAM_DOMAINS = [
    "merriam-webster.com", "statmuse.com", "alltrails.com", "untappd.com",
    "yelp.com", "tiktok.com", "lamamapasteis.com.br", "tomasiusa.com",
    "univet.ee", "wisap.de", "facestudio.eu", "ingebjorg.no",
    "rectificadosindustriales.com", "fernandopando.com", "cmarinho.com.br",
    "prwe.com", "sigra.com", "educacioninnovadora.es", "lan-portal.uob.edu.ly",
    "weterynarzjarocin.pl",
]

# Google News RSS queries — each filtered to last 48 hours.
GNEWS_QUERIES = [
    # Direct exact-match mentions
    '"BusinessDen" -site:businessden.com when:2d',
    # Blog-focused results (surfaces different sources)
    '"BusinessDen" blog -site:businessden.com when:2d',
    # Reporter bylines — qualified to avoid false matches
    # (Matt Geiger qualified with Denver to avoid NBA player)
    '"Thomas Gounley" Denver -site:businessden.com when:2d',
    '"Justin Wingerter" Denver -site:businessden.com when:2d',
    '"Matt Geiger" Denver business -site:businessden.com when:2d',
    '"Aaron Kremer" -site:businessden.com when:2d',
]

USER_AGENT = "ReputationTracker/1.0 (BusinessDen; github.com/businessden/Reputation)"

# Relevance keywords — at least one must appear in title or snippet
# for results from broad queries to be kept
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
    """Create a stable unique ID from the canonical URL."""
    normalized = url.lower().strip().rstrip("/")
    normalized = re.sub(r'[?&](utm_\w+|fbclid|gclid|ref|oc|source|medium|campaign)=[^&]*', '', normalized)
    normalized = normalized.rstrip("?&")
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def is_excluded(url: str) -> bool:
    """Check if URL is on an excluded or spam domain."""
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        for excl in EXCLUDED_DOMAINS + SPAM_DOMAINS:
            if domain == excl or domain.endswith("." + excl):
                return True
    except Exception:
        pass
    return False


def is_relevant(title: str, snippet: str) -> bool:
    """Check if content is relevant (has at least one relevance keyword)."""
    text = (title + " " + snippet).lower()
    return any(kw in text for kw in RELEVANCE_KEYWORDS)


def parse_rss_date(date_str: str) -> str | None:
    if not date_str:
        return None
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str).isoformat()
    except Exception:
        return date_str


def extract_domain(url: str) -> str:
    try:
        d = urlparse(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return "unknown"


def strip_html(text: str) -> str:
    clean = re.sub(r'<[^>]+>', '', text or "")
    return html.unescape(clean).strip()


def truncate(text: str, length: int = 500) -> str:
    if not text:
        return ""
    return text[:length - 3] + "..." if len(text) > length else text


def load_data() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"mentions": [], "last_updated": None, "summaries": [], "run_log": []}


def save_data(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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
                source_title = entry.get("source", {}).get("title", "Unknown")
                source_href = entry.get("source", {}).get("href", "")
                google_link = entry.get("link", "")

                if source_href and is_excluded(source_href):
                    continue
                if google_link and is_excluded(google_link):
                    continue
                if source_title.lower().strip() == "businessden":
                    continue

                title_raw = entry.get("title", "")
                title_key = title_raw.lower().strip()
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)

                title = title_raw.rsplit(" - ", 1)[0].strip() if " - " in title_raw else title_raw.strip()
                snippet = truncate(strip_html(entry.get("summary", "")))

                # Relevance check
                if not is_relevant(title, snippet):
                    continue

                results.append({
                    "title": title,
                    "title_raw": title_raw,
                    "url": google_link,
                    "snippet": snippet,
                    "source": source_title,
                    "source_url": source_href,
                    "source_domain": extract_domain(source_href) if source_href else source_title.lower(),
                    "published": parse_rss_date(entry.get("published", "")),
                    "found_via": "google_news_rss",
                    "query_used": query,
                    "entry_id": entry.get("id", ""),
                })
                count += 1

            print(f"         → {count} relevant entries (of {len(feed.entries)} raw)")
        except Exception as e:
            print(f"         → Error: {e}")

    return results


# ---------------------------------------------------------------------------
# Source 2: WordPress.com Public API
# ---------------------------------------------------------------------------

def scrape_wordpress() -> list[dict]:
    results = []
    queries = ["BusinessDen", '"Aaron Kremer"']

    for q in queries:
        print(f"  [WP] Searching: {q}")
        try:
            resp = requests.get(
                "https://public-api.wordpress.com/rest/v1.1/read/search",
                params={"q": q, "number": 20},
                headers={"User-Agent": USER_AGENT},
                timeout=15,
            )
            resp.raise_for_status()
            posts = resp.json().get("posts", [])
            count = 0

            for p in posts:
                if not p or not isinstance(p, dict):
                    continue

                url = p.get("URL", "")
                if not url or is_excluded(url):
                    continue

                content_lower = (str(p.get("content", "")) + str(p.get("excerpt", ""))).lower()
                if "businessden" not in content_lower and "aaron kremer" not in content_lower:
                    continue

                raw_excerpt = strip_html(str(p.get("excerpt", "")) or str(p.get("content", "")))
                snippet = truncate(raw_excerpt)

                # Find context sentence
                context = ""
                for s in re.split(r'(?<=[.!?])\s+', raw_excerpt):
                    if "businessden" in s.lower() or "aaron kremer" in s.lower():
                        context = s.strip()
                        break

                author_data = p.get("author")
                author_name = ""
                if isinstance(author_data, dict):
                    author_name = author_data.get("name", "")

                results.append({
                    "title": strip_html(str(p.get("title", ""))),
                    "title_raw": str(p.get("title", "")),
                    "url": url,
                    "snippet": context or snippet,
                    "source": str(p.get("site_name", "")) or extract_domain(url),
                    "source_url": str(p.get("site_URL", "")),
                    "source_domain": extract_domain(str(p.get("site_URL", url))),
                    "published": p.get("date"),
                    "found_via": "wordpress_api",
                    "query_used": q,
                    "wp_site_id": p.get("site_ID"),
                    "wp_post_id": p.get("ID"),
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
    queries = ["BusinessDen"]

    for q in queries:
        url = f"https://www.reddit.com/search.rss?q={q}&sort=new&t=week"
        print(f"  [REDDIT] Searching: {q}")

        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
            if resp.status_code != 200:
                print(f"           → HTTP {resp.status_code} (may be blocked from this IP — will work from GitHub Actions)")
                continue

            feed = feedparser.parse(resp.text)
            count = 0

            for entry in feed.entries:
                link = entry.get("link", "")
                if is_excluded(link):
                    continue

                title = strip_html(entry.get("title", ""))
                content = strip_html(entry.get("summary", ""))
                snippet = truncate(content)

                sub_match = re.search(r'reddit\.com/r/(\w+)', link)
                subreddit = sub_match.group(1) if sub_match else ""

                results.append({
                    "title": title,
                    "title_raw": entry.get("title", ""),
                    "url": link,
                    "snippet": snippet,
                    "source": f"r/{subreddit}" if subreddit else "Reddit",
                    "source_url": f"https://www.reddit.com/r/{subreddit}" if subreddit else "https://www.reddit.com",
                    "source_domain": "reddit.com",
                    "published": parse_rss_date(entry.get("published", "")),
                    "found_via": "reddit_rss",
                    "query_used": q,
                    "author": entry.get("author", ""),
                    "subreddit": subreddit,
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
    tags = ["businessden", "denver-business", "denver-real-estate"]

    for tag in tags:
        url = f"https://medium.com/feed/tag/{tag}"
        print(f"  [MEDIUM] Tag: {tag}")

        try:
            feed = feedparser.parse(url)
            count = 0

            for entry in feed.entries:
                link = entry.get("link", "")
                if is_excluded(link):
                    continue

                title = strip_html(entry.get("title", ""))
                content = strip_html(entry.get("summary", ""))

                if tag != "businessden" and "businessden" not in (title + content).lower():
                    continue

                results.append({
                    "title": title,
                    "title_raw": entry.get("title", ""),
                    "url": link,
                    "snippet": truncate(content),
                    "source": "Medium",
                    "source_url": "https://medium.com",
                    "source_domain": "medium.com",
                    "published": parse_rss_date(entry.get("published", "")),
                    "found_via": "medium_rss",
                    "query_used": f"tag:{tag}",
                    "author": entry.get("author", ""),
                    "tags": [t.get("term", "") for t in entry.get("tags", [])][:10],
                })
                count += 1

            print(f"           → {count} results (of {len(feed.entries)} in feed)")
        except Exception as e:
            print(f"           → Error: {e}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def scrape():
    data = load_data()
    existing_ids = {m["id"] for m in data["mentions"]}
    new_mentions = []
    now = datetime.now(timezone.utc)

    print(f"Reputation scraper — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Existing mentions in database: {len(data['mentions'])}\n")

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
    print(f"\nTotal results after filtering: {len(all_results)}")

    for item in all_results:
        article_id = make_id(item["url"])
        if article_id in existing_ids:
            continue

        mention = {
            "id": article_id,
            "title": item["title"],
            "title_raw": item.get("title_raw", item["title"]),
            "url": item["url"],
            "snippet": item["snippet"],
            "source": item["source"],
            "source_url": item.get("source_url", ""),
            "source_domain": item["source_domain"],
            "published": item["published"],
            "found_via": item["found_via"],
            "query_used": item.get("query_used", ""),
            "first_seen": now.isoformat(),
            "author": item.get("author", ""),
            "subreddit": item.get("subreddit", ""),
            "tags": item.get("tags", []),
            "categories": item.get("categories", []),
            "wp_site_id": item.get("wp_site_id"),
            "wp_post_id": item.get("wp_post_id"),
            "wp_like_count": item.get("wp_like_count"),
            "wp_comment_count": item.get("wp_comment_count"),
            "entry_id": item.get("entry_id", ""),
        }

        data["mentions"].append(mention)
        existing_ids.add(article_id)
        new_mentions.append(mention)
        tag = f"[{item['found_via']}]"
        print(f"  NEW {tag:22s} {item['title'][:60]}  ({item['source']})")

    # Sort newest first
    data["mentions"].sort(key=lambda m: m.get("published") or "0000", reverse=True)

    data["last_updated"] = now.isoformat()
    data["new_this_run"] = len(new_mentions)

    # Run log
    if "run_log" not in data:
        data["run_log"] = []
    data["run_log"].append({
        "timestamp": now.isoformat(),
        "new_mentions": len(new_mentions),
        "total_mentions": len(data["mentions"]),
        "sources_checked": {
            "google_news_rss": len(gnews),
            "wordpress_api": len(wp),
            "reddit_rss": len(reddit),
            "medium_rss": len(medium),
        },
    })
    cutoff = (now - timedelta(days=90)).isoformat()
    data["run_log"] = [r for r in data["run_log"] if r["timestamp"] >= cutoff]

    save_data(data)
    print(f"\nDone. {len(new_mentions)} new mentions. {len(data['mentions'])} total in database.")


if __name__ == "__main__":
    scrape()
