import json
import re
import time
import pathlib
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager

# 1. Setup Edge Options
edge_options = EdgeOptions()
edge_options.add_argument("--headless")  # Run in background
edge_options.add_argument("--no-sandbox")
edge_options.add_argument("--disable-dev-shm-usage")
edge_options.add_argument("window-size=1920,1080") # Set a standard screen size
edge_options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

driver = webdriver.Edge(
    service=EdgeService(EdgeChromiumDriverManager().install()),
    options=edge_options,
)

URL = "https://www.otto.de/herren/mode/hosen/"

try:
    print(f"Opening Edge and loading: {URL} ...")
    driver.get(URL)
    
    # 2. Wait up to 15 seconds until at least one product article tag is present in the DOM
    print("Waiting for products to load...")
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "article"))
    )
    
    # 3. Scroll down the page gradually to trigger lazy-loading of images and products
    print("Scrolling page to load all items...")
    for _ in range(3):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 3 * %d);" % (_ + 1))
        time.sleep(2) # Wait for content to load after scroll

    # Parse the fully loaded page source
    soup = BeautifulSoup(driver.page_source, "html.parser")
    scraped_data = {}

    # Otto uses 'article' tags with class names containing 'product' or 'find_tile'
    product_tiles = soup.find_all("article")
    
    # Filter only articles that look like products (usually have a data-product-id)
    product_tiles = [tile for tile in product_tiles if tile.get("data-product-id") or "product" in "".join(tile.get("class", []))]

    print(f"Bypassed security and lazy-loading! Found {len(product_tiles)} products.")

    for index, tile in enumerate(product_tiles, start=1):
        try:
            product_id = (
                tile.get("data-product-id")
                or tile.get("id")
                or f"otto_pants_{index}"
            )

            # Extract Name (Otto uses specific class names for titles)
            name_tag = tile.find(class_=re.compile(r"title|name|p_brand-name"))
            if not name_tag:
                name_tag = tile.find("h2")
            product_name = name_tag.text.strip() if name_tag else "Unknown Pants"

            # Extract Brand
            brand_tag = tile.find(class_=re.compile(r"brand|manufacturer"))
            manufacturer = (
                brand_tag.text.strip() if brand_tag else "Otto"
            )

            # Extract Price
            price_tag = tile.find(class_=re.compile(r"price|selling-price|p_price"))
            price = price_tag.text.strip() if price_tag else "0,00 €"

            # Extract URL
            link_tag = tile.find("a", href=True)
            product_url = (
                f"https://www.otto.de{link_tag['href']}" if link_tag else ""
            )

            # Extract Image
            img_tag = tile.find("img", src=True)
            image_url = img_tag["src"] if img_tag else ""
            alt_text = img_tag.get("alt", product_name) if img_tag else product_name

            scraped_data[f"product_{index}"] = {
                "id": product_id,
                "product_name": product_name,
                "manufacturer": manufacturer,
                "price": price,
                "variants_info": "+ more variants" if "variant" in tile.text.lower() else "",
                "product_url": product_url,
                "category": "mens_pants",
                "image_url": image_url,
                "alt_text": alt_text,
            }
        except Exception as e:
            continue

    # Save to JSON
    output_file = pathlib.Path(__file__).resolve().parent /"web-scraping"/ "Scraped_data" / "webscraping_trial_pants.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(scraped_data)} products to {output_file}!")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    driver.quit()