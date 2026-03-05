#!/usr/bin/env python3
"""
Reputation - Scraper
Finds third-party mentions of BusinessDen using two sources:
  1. Google News RSS (free, no key required)
  2. Serper.dev Google Search API (SERPER_API_KEY required)

All results hosted on businessden.com are excluded.
Only third-party references to BusinessDen are tracked.
"""

import feedparser
import requests
import json
import os
import hashlib
import re
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = "mentions-data.json"
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

EXCLUDED_DOMAINS = [
    "businessden.com",
    "www.businessden.com",
]

# Google News RSS queries (free, no auth)
RSS_QUERIES = [
    '"BusinessDen" -site:businessden.com',
]

# Serper.dev queries (uses 1 credit each)
SERPER_QUERIES = [
    '"BusinessDen" -site:businessden.com',
    'BusinessDen Denver -site:businessden.com',
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_id(url: str) -> str:
    """Create a stable unique ID from the canonical URL."""
    # Normalize URL for dedup
    normalized = url.lower().strip().rstrip("/")
    # Remove tracking params
    normalized = re.sub(r'[?&](utm_\w+|fbclid|gclid|ref)=[^&]*', '', normalized)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def is_excluded(url: str) -> bool:
    """Check if a URL is on an excluded domain."""
    try:
        domain = urlparse(url).netloc.lower()
        for excluded in EXCLUDED_DOMAINS:
            if domain == excluded or domain.endswith("." + excluded):
                return True
    except Exception:
        pass
    return False


def parse_date(date_str: str) -> str | None:
    """Parse various date formats into ISO 8601."""
    if not date_str:
        return None
    # RSS date format
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(date_str)
        return dt.isoformat()
    except Exception:
        pass
    # ISO format already
    if "T" in date_str or len(date_str) == 10:
        return date_str
    # Serper sometimes returns "X days ago", "X hours ago", etc.
    try:
        now = datetime.now(timezone.utc)
        if "hour" in date_str:
            hours = int(re.search(r'(\d+)', date_str).group(1))
            return (now - timedelta(hours=hours)).isoformat()
        elif "day" in date_str:
            days = int(re.search(r'(\d+)', date_str).group(1))
            return (now - timedelta(days=days)).isoformat()
        elif "week" in date_str:
            weeks = int(re.search(r'(\d+)', date_str).group(1))
            return (now - timedelta(weeks=weeks)).isoformat()
        elif "month" in date_str:
            months = int(re.search(r'(\d+)', date_str).group(1))
            return (now - timedelta(days=months * 30)).isoformat()
        elif "year" in date_str:
            years = int(re.search(r'(\d+)', date_str).group(1))
            return (now - timedelta(days=years * 365)).isoformat()
    except Exception:
        pass
    return date_str


def extract_domain(url: str) -> str:
    """Extract a clean domain name from URL."""
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return "unknown"


def load_existing() -> dict:
    """Load existing data file."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"mentions": [], "last_updated": None, "summaries": []}


def save_data(data: dict):
    """Save data file."""
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Source 1: Google News RSS
# ---------------------------------------------------------------------------

def scrape_google_news_rss() -> list[dict]:
    """Fetch mentions from Google News RSS feeds."""
    results = []

    for query in RSS_QUERIES:
        encoded = requests.utils.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        print(f"  [RSS] Fetching: {query}")

        feed = feedparser.parse(url)
        print(f"  [RSS] Got {len(feed.entries)} entries")

        for entry in feed.entries:
            title_raw = entry.get("title", "")
            source_title = entry.get("source", {}).get("title", "Unknown")
            source_href = entry.get("source", {}).get("href", "")
            published = entry.get("published", "")
            google_link = entry.get("link", "")
            summary = entry.get("summary", "")

            # Skip if source is BusinessDen itself
            if source_href and is_excluded(source_href):
                continue
            if source_title.lower().strip() == "businessden":
                continue

            # Clean title: remove " - Source Name" suffix
            title = title_raw
            if " - " in title:
                title = title.rsplit(" - ", 1)[0].strip()

            # Extract snippet from summary (RSS summary is HTML)
            snippet = re.sub(r'<[^>]+>', '', summary).strip()
            if len(snippet) > 400:
                snippet = snippet[:397] + "..."

            results.append({
                "title": title,
                "url": google_link,
                "snippet": snippet,
                "source": source_title,
                "source_domain": extract_domain(source_href) if source_href else source_title.lower(),
                "published": parse_date(published),
                "found_via": "google_news_rss",
            })

    return results


# ---------------------------------------------------------------------------
# Source 2: Serper.dev Google Search
# ---------------------------------------------------------------------------

def scrape_serper() -> list[dict]:
    """Fetch mentions from Serper.dev Google Search API."""
    if not SERPER_API_KEY:
        print("  [SERPER] No API key set — skipping Serper search")
        return []

    results = []
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }

    for query in SERPER_QUERIES:
        print(f"  [SERPER] Searching: {query}")
        try:
            # Web search
            resp = requests.post(
                "https://google.serper.dev/search",
                headers=headers,
                json={"q": query, "num": 20, "gl": "us", "hl": "en"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            organic = data.get("organic", [])
            print(f"  [SERPER] Got {len(organic)} organic results")

            for item in organic:
                url = item.get("link", "")
                if is_excluded(url):
                    continue

                results.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "snippet": item.get("snippet", ""),
                    "source": item.get("source", extract_domain(url)),
                    "source_domain": extract_domain(url),
                    "published": parse_date(item.get("date", "")),
                    "found_via": "serper",
                })

            # Also search Google News via Serper
            print(f"  [SERPER] Searching news: {query}")
            resp_news = requests.post(
                "https://google.serper.dev/news",
                headers=headers,
                json={"q": query, "num": 20, "gl": "us", "hl": "en"},
                timeout=15,
            )
            resp_news.raise_for_status()
            news_data = resp_news.json()

            news_items = news_data.get("news", [])
            print(f"  [SERPER] Got {len(news_items)} news results")

            for item in news_items:
                url = item.get("link", "")
                if is_excluded(url):
                    continue

                results.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "snippet": item.get("snippet", ""),
                    "source": item.get("source", extract_domain(url)),
                    "source_domain": extract_domain(url),
                    "published": parse_date(item.get("date", "")),
                    "found_via": "serper_news",
                })

        except requests.exceptions.RequestException as e:
            print(f"  [SERPER] Error: {e}")

    return results


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

def scrape():
    data = load_existing()
    existing_ids = {m["id"] for m in data["mentions"]}
    new_mentions = []

    # Collect from both sources
    print("Scraping Google News RSS...")
    rss_results = scrape_google_news_rss()

    print("Scraping Serper.dev...")
    serper_results = scrape_serper()

    all_results = rss_results + serper_results
    print(f"\nTotal raw results: {len(all_results)}")

    # Deduplicate and add new mentions
    for item in all_results:
        article_id = make_id(item["url"])
        if article_id in existing_ids:
            continue

        mention = {
            "id": article_id,
            "title": item["title"],
            "url": item["url"],
            "snippet": item["snippet"],
            "source": item["source"],
            "source_domain": item["source_domain"],
            "published": item["published"],
            "found_via": item["found_via"],
            "first_seen": datetime.now(timezone.utc).isoformat(),
        }

        data["mentions"].append(mention)
        existing_ids.add(article_id)
        new_mentions.append(mention)
        print(f"  NEW [{item['found_via']:18s}] {item['title'][:65]}  ({item['source']})")

    # Sort by published date (newest first), nulls last
    data["mentions"].sort(
        key=lambda m: m.get("published") or "0000",
        reverse=True,
    )

    # Update metadata
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    data["new_this_run"] = len(new_mentions)

    save_data(data)
    print(f"\nDone. {len(new_mentions)} new mentions. {len(data['mentions'])} total.")

    return new_mentions


if __name__ == "__main__":
    scrape()
