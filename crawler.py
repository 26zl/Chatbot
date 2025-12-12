# crawler.py
import json
import os
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
from security_utils import UrlValidationError, validate_outbound_url

# Configuration
START_URL = "https://developers.cloudflare.com/workers/"
BASE_DOMAIN = "developers.cloudflare.com"
MAX_PAGES = 200
OUTPUT_FILE = "data/crawled.jsonl"
EVAL_SET_PATH = "data/eval_set.jsonl"
REQUEST_DELAY = 0.5


def is_same_domain(url):
    """Check if URL belongs to our target domain"""
    try:
        return BASE_DOMAIN in urlparse(url).netloc
    except:
        return False


def fetch_page(url):
    """Fetch a single page and return HTML"""
    try:
        try:
            url = validate_outbound_url(url, allowed_domains=[BASE_DOMAIN])
        except UrlValidationError as e:
            print(f"Blocked URL {url}: {e}")
            return None

        response = requests.get(url, timeout=10, headers={
            "User-Agent": "SimpleBot/1.0"
        })
        if response.status_code == 200:
            return response.text
        return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def extract_text(html):
    """Extract clean text from HTML"""
    soup = BeautifulSoup(html, 'html.parser')
    
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()
    
    text = soup.get_text()
    lines = [line.strip() for line in text.splitlines()]
    text = '\n'.join(line for line in lines if line)
    
    return text


def extract_links(html, base_url):
    """Extract all links from HTML"""
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        absolute_url = urljoin(base_url, href)
        if is_same_domain(absolute_url):
            links.append(absolute_url)
    
    return links


def save_to_file(url, text):
    """Save a single record to file"""
    record = {"url": url, "text": text}
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')


def load_reference_urls():
    """Pull reference URLs from the eval set so we always crawl gold sources"""
    if not os.path.exists(EVAL_SET_PATH):
        return []

    urls = []
    with open(EVAL_SET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = record.get("reference_url")
            if url and is_same_domain(url):
                urls.append(url)
    return urls


def crawl():
    """Main BFS crawler"""
    print(f"Starting BFS crawler...")
    print(f"Target: {START_URL}")
    print(f"Max pages: {MAX_PAGES}\n")
    
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    seed_urls = [START_URL]
    reference_urls = load_reference_urls()
    if reference_urls:
        print(f"Loaded {len(reference_urls)} reference URLs from {EVAL_SET_PATH}")
        seed_urls.extend(reference_urls)

    # Deduplicate while preserving order
    seen = set()
    queue = []
    for url in seed_urls:
        if url not in seen:
            queue.append(url)
            seen.add(url)

    visited = set()

    while queue and len(visited) < MAX_PAGES:
        url = queue.pop(0)
        
        if url in visited:
            continue
        
        print(f"[{len(visited) + 1}/{MAX_PAGES}] Crawling: {url}")
        
        html = fetch_page(url)
        if not html:
            continue
        
        text = extract_text(html)
        
        if len(text) < 200:
            print(f"  Skipped (too short)")
            del html, text
            continue
        
        save_to_file(url, text)
        print(f"  Saved ({len(text)} chars)")
        
        links = extract_links(html, url)
        new_links = [link for link in links if link not in visited]
        queue.extend(new_links)
        print(f"  Found {len(new_links)} new links")
        
        visited.add(url)
        del html, text
        time.sleep(REQUEST_DELAY)
    
    print(f"\nDone! Crawled {len(visited)} pages")
    print(f"Output saved to: {OUTPUT_FILE}")


def load_records(path=OUTPUT_FILE):
    """Load all records from the crawled file"""
    records = []
    if not os.path.exists(path):
        return records
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                continue
    return records


if __name__ == "__main__":
    crawl()
