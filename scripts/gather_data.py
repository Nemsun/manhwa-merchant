import re
import csv
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

URL: str = 'https://graphql.anilist.co'
OUTPUT_FILE: str = 'manhwa_data_complete.csv'

PER_PAGE: int = 50          # Must match the perPage value in the query string
MAX_RETRIES: int = 5        # Max consecutive retry attempts before aborting
RATE_LIMIT_SLEEP: int = 60  # Seconds to wait after a 429 response
REQUEST_TIMEOUT: tuple[int, int] = (10, 30)  # (connect timeout, read timeout)

QUERY: str = f'''
query ($page: Int) {{
  Page(page: $page, perPage: {PER_PAGE}) {{
    pageInfo {{ hasNextPage }}
    media(type: MANGA, countryOfOrigin: "KR", sort: POPULARITY_DESC) {{
      id
      title {{ english romaji }}
      averageScore
      popularity
      genres
      tags {{ name rank }}
      description
      characters(role: MAIN) {{
        edges {{ node {{ name {{ full }} }} }}
      }}
    }}
  }}
}}
'''

HEADERS: dict[str, str] = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'User-Agent': 'Manhwa-Merchant/1.0'
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_html(text: str | None) -> str | None:
    """
    Remove HTML tags from AniList description fields.
    Raw descriptions contain <br>, <i>, <b>, etc. which pollute
    text features (TF-IDF, embeddings) used by the recommender.
    """
    if not text:
        return None
    return re.sub(r'<[^>]+>', ' ', text).strip()


def parse_tags(tags_list: list[dict], rank_threshold: int = 0) -> str:
    """
    Return a pipe-delimited string of 'name:rank' pairs for tags whose
    rank exceeds the threshold.  Preserving the rank allows the recommender
    to weight tags rather than treating them as binary features.

    The default threshold is 0 (store every ranked tag). Rank-based
    filtering and tag-block down-weighting are deferred to the notebook so
    the cutoff can be tuned without re-hitting the API on every experiment.

    Example output: "Action:85|Fantasy:92|Romance:78"
    """
    filtered = [
        f"{t['name']}:{t['rank']}"
        for t in tags_list
        if t.get('rank', 0) > rank_threshold and t.get('name')
    ]
    return "|".join(filtered)


def parse_main_character(char_block: dict | None) -> str | None:
    """
    Safely extract the full name of the first main character.
    Returns None if any level of the nested structure is missing or malformed.
    """
    if not char_block:
        return None
    edges = char_block.get('edges') or []
    if not edges:
        return None

    # edges[0] exists but could still have a malformed/null node
    node = (edges[0] or {}).get('node') or {}
    name_dict = node.get('name') or {}
    return name_dict.get('full')  # None if absent — intentional, not a silent bug


def flatten_media_item(item: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw AniList media object into a flat CSV-ready dict."""
    title_info = item.get('title') or {}
    genres_list = item.get('genres') or []
    tags_list   = item.get('tags') or []

    return {
        'id':          item.get('id'),
        'title':       title_info.get('english') or title_info.get('romaji') or "Unknown Title",
        'score':       item.get('averageScore'),
        'popularity':  item.get('popularity'),
        'genres':      "|".join(genres_list),
        # Tags stored unfiltered as name:rank pairs; rank filtering happens
        # downstream in the notebook so the threshold stays tunable.
        'tags':        parse_tags(tags_list),
        'main_char':   parse_main_character(item.get('characters')),
        # HTML stripped so text features aren't polluted with markup tokens
        'description': strip_html(item.get('description')),
    }


# ---------------------------------------------------------------------------
# Core fetch logic
# ---------------------------------------------------------------------------

def fetch_page(page: int) -> requests.Response:
    """
    POST a single GraphQL page request.
    Raises requests.exceptions.RequestException on network-level failures
    so the caller can decide whether to retry.
    """
    return requests.post(
        URL,
        json={'query': QUERY, 'variables': {'page': page}},
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,   # Prevents infinite hangs on stalled connections
    )


def fetch_manhwa_data(max_pages: int | None = None) -> pd.DataFrame:
    """
    Paginate through the AniList GraphQL API and write results to CSV
    incrementally so a mid-run crash doesn't lose collected data.

    Returns a DataFrame of everything collected (read back from the CSV).
    """
    output_path = Path(OUTPUT_FILE)
    page        = 1
    total_rows  = 0
    write_header = True

    with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer: csv.DictWriter | None = None  # Initialised on first successful page

        while True:
            if max_pages is not None and page > max_pages:
                print(f"Reached requested page limit ({max_pages}).")
                break

            print(f"Fetching page {page}...")
            retries = 0

            # --- Request + retry loop (handles 429s and transient errors) ---
            while retries < MAX_RETRIES:
                try:
                    response = fetch_page(page)
                except requests.exceptions.RequestException as exc:
                    retries += 1
                    wait = 2 ** retries  # Exponential backoff: 2, 4, 8, 16, 32 s
                    print(f"  [!] Network error (attempt {retries}/{MAX_RETRIES}): {exc}")
                    print(f"  [!] Retrying in {wait}s...")
                    time.sleep(wait)
                    continue

                if response.status_code == 429:
                    retries += 1
                    print(f"  [!] Rate limit hit (attempt {retries}/{MAX_RETRIES}). "
                          f"Sleeping {RATE_LIMIT_SLEEP}s...")
                    time.sleep(RATE_LIMIT_SLEEP)
                    continue

                if response.status_code != 200:
                    print(f"\n[!] API error — status {response.status_code}")
                    print(f"[!] Server payload:\n{response.text}\n")
                    return pd.read_csv(output_path) if total_rows else pd.DataFrame()

                break  # Successful response — exit retry loop

            else:
                # Exhausted all retries
                print(f"\n[!] Giving up after {MAX_RETRIES} failed attempts on page {page}.")
                break

            # --- Parse response ---
            try:
                payload    = response.json()
                page_data  = payload.get('data', {}).get('Page', {})
            except (ValueError, KeyError) as exc:
                print(f"\n[!] Failed to parse JSON on page {page}: {exc}")
                break

            # Check pagination *before* testing media list so we don't
            # misinterpret a legitimately empty final page as an error.
            page_info    = page_data.get('pageInfo') or {}
            has_next     = page_info.get('hasNextPage', False)

            media_list   = page_data.get('media') or []
            if not media_list:
                print("Empty media list — end of database.")
                break

            # --- Flatten and write page to CSV immediately ---
            flat_items = [flatten_media_item(item) for item in media_list]

            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=flat_items[0].keys())

            if write_header:
                writer.writeheader()
                write_header = False

            writer.writerows(flat_items)
            csv_file.flush()  # Force OS buffer to disk after every page

            total_rows += len(flat_items)
            print(f"  -> Page {page} written ({len(flat_items)} rows, {total_rows} total)")

            if not has_next:
                print("Reached the end of the database!")
                break

            page += 1
            time.sleep(1)  # Polite delay — placed after successful processing

    if total_rows == 0:
        print("Extraction failed. No data collected.")
        return pd.DataFrame()

    df = pd.read_csv(output_path)
    print(f"\nDone! {total_rows} manhwas saved to '{OUTPUT_FILE}'.")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Starting global database extraction...")
    df = fetch_manhwa_data()

    if not df.empty:
        print(f"\nDataFrame shape: {df.shape}")
        print(df.head())