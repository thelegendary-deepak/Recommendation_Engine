import json
import pathlib
import re
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.microsoft import EdgeChromiumDriverManager


def get_edge_driver():
    """Helper to initialize a stable, crash-resistant Edge browser instance."""
    edge_options = EdgeOptions()
    edge_options.add_argument("--headless=new")  # Use the modern, stable headless engine
    edge_options.add_argument("--disable-gpu")    # Prevents hardware acceleration crashes
    edge_options.add_argument("--no-sandbox")
    edge_options.add_argument("--disable-dev-shm-usage")
    edge_options.add_argument("--disable-blink-features=AutomationControlled") # Hides automation
    edge_options.add_argument("window-size=1920,1080")
    edge_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    
    # Block automation flags to prevent Otto from crashing the connection
    edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    edge_options.add_experimental_option('useAutomationExtension', False)
    
    return webdriver.Edge(
        service=EdgeService(EdgeChromiumDriverManager().install()),
        options=edge_options,
    )


DEPARTMENTS = [
    {"url": "https://www.otto.de/spielzeug/puzzle/", "dept": "Toys"},
        {"url": "https://www.otto.de/moebel/", "dept": "Furniture"},
    {"url": "https://www.otto.de/koerperpflege/", "dept": "Beauty"},
    {
        "url": "https://www.otto.de/technik/multimedia/?reduziert",
        "dept": "Electronics",
    },
    {"url": "https://www.otto.de/buecher/", "dept": "Media"},
    {"url": "https://www.otto.de/?selektion=(und.(ist.sortiment.fahrzeuge).(oder.(sind.kategorien.e-bikes).(sind.kategorien.fahrraeder)).(~.(v.1)))&variationIds=1824846155%2CS0QE4088T4NU",
     "dept": "Sports & Outdoor"},
    {
        "url": "https://www.otto.de/haushalt/haushaltsgeraete/",
        "dept": "Home & Living",
    },
    {
        "url": "https://www.otto.de/herren/schuhe/?reduziert",
        "dept": "Fashion",
        "gender": "Unisex",
    },
    {
        "url": "https://www.otto.de/?selektion=(und.(ist.aktion.super-sale).(ist.rabatt.30).(ist.zielgruppe.damen).(~.(v.1)))",
        "dept": "Fashion",
        "gender": "Female",
    },
 
]

scraped_data = {}
global_product_counter = 1

for target in DEPARTMENTS:
    url = target["url"]
    dept_name = target["dept"]
    gender = target.get("gender", None)

    print(f"--- Scraping Department: {dept_name} ({url}) ---")

    # Start a fresh browser instance for each department
    driver = get_edge_driver()

    try:
        driver.get(url)

        # Wait for products to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "article"))
        )

        # Scroll to lazy-load products
        for _ in range(4):
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight / 4 * %d);"
                % (_ + 1)
            )
            time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        articles = soup.find_all("article")
        product_tiles = [
            a
            for a in articles
            if a.get("data-product-id")
            or "product" in "".join(a.get("class", []))
        ][:120]

        print(f"Found {len(product_tiles)} products for {dept_name}.")

        for tile in product_tiles:
            try:
                product_id = (
                    tile.get("data-product-id")
                    or tile.get("id")
                    or f"otto_{global_product_counter}"
                )

                name_tag = tile.find(
                    class_=re.compile(r"title|name|p_brand-name")
                )
                if not name_tag:
                    name_tag = tile.find("h2")
                product_name = (
                    name_tag.text.strip() if name_tag else "Unknown Product"
                )
                if product_name == "Unknown Product" or not product_name:
                    continue

                brand_tag = tile.find(class_=re.compile(r"brand|manufacturer"))
                manufacturer = brand_tag.text.strip() if brand_tag else "Unknown"

                price_tag = tile.find(
                    class_=re.compile(r"price|selling-price|p_price")
                )
                price = price_tag.text.strip() if price_tag else "0,00 €"
                if price == "0,00 €" or not price:
                    continue

                link_tag = tile.find("a", href=True)
                product_url = (
                    f"https://www.otto.de{link_tag['href']}" if link_tag else ""
                )

                img_tag = tile.find("img", src=True)
                image_url = img_tag["src"] if img_tag else ""
                alt_text = (
                    img_tag.get("alt", product_name)
                    if img_tag
                    else product_name
                )

                product_entry = {
                    "id": product_id,
                    "product_name": product_name,
                    "manufacturer": manufacturer,
                    "price": price,
                    "variants_info": "+ more variants"
                    if "variant" in tile.text.lower()
                    else "",
                    "product_url": product_url,
                    "Department": dept_name,
                    "image_url": image_url,
                    "alt_text": alt_text,
                }

                if gender:
                    product_entry["Gender_Target"] = gender

                scraped_data[f"product_{global_product_counter}"] = (
                    product_entry
                )
                global_product_counter += 1

            except Exception as e:
                continue

    except Exception as e:
        print(f"Skipping department {dept_name} due to error: {e}")
    finally:
        # Safely shut down the browser instance
        driver.quit()

    # Polite delay to avoid bot detection between departments
    time.sleep(5)

# Save output
output_dir = pathlib.Path(__file__).resolve().parent / "Scraped_data"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "otto_departments_v2.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(scraped_data, f, ensure_ascii=False, indent=4)

print(f"Successfully saved {len(scraped_data)} products to {output_file}!")