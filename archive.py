import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sample = pd.read_csv("data/processed/otto_sample_data.csv", encoding="cp1252")
categories = sample["Product_Category"].dropna().drop_duplicates().tolist()
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36"}
session = requests.Session()
session.headers.update(headers)
session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))


def scrape_category(category, limit=10):
    url = f"https://www.otto.de/suche/{quote(category.lower())}/"
    response = session.get(url, timeout=(10, 30))
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    products = []
    cards = soup.select('[data-testid="product-tile"], article')
    for card in cards:
        link = card.select_one('a[href*="/p/"]')
        name = card.select_one('[data-testid*="title"], h2, h3')
        price = card.select_one('[data-testid*="price"], [class*="price"]')
        if link and name:
            products.append({
                "category": category,
                "product_name": name.get_text(" ", strip=True),
                "price": price.get_text(" ", strip=True) if price else None,
                "product_url": urljoin(url, link.get("href")),
            })
    return pd.DataFrame(products).drop_duplicates("product_url").head(limit)

frames = []
for category in categories:
    try:
        frames.append(scrape_category(category))
        print(category, "done")
    except requests.RequestException as error:
        print(category, "failed:", error)
    time.sleep(2)

scraped = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
scraped.to_csv("data/processed/otto_scraped_products.csv", index=False, encoding="utf-8-sig")
scraped.groupby("category").size() if not scraped.empty else "No products scraped"