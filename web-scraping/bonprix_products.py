import json
import pathlib
import re
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

DEPARTMENTS = [
    {"url": "https://www.bonprix.de/kategorie/damen-mode-tops/", 
     "dept": "Tops",
     "Gender_Target": "Female"},
    {"url": "https://www.bonprix.de/kategorie/herren-mode-t-shirts/",
     "dept": "Tops",
     "Gender_Target": "Unisex"},
    {"url":"https://www.bonprix.de/kategorie/herren-mode-sportbekleidung/",
     "dept": "Activewear",
     "Gender_Target": "Unisex"},
    {"url": "https://www.bonprix.de/kategorie/herren-mode-jeans/",
      "dept": "Bottoms",
      "Gender_Target": "Unisex"},
    { "url":"https://www.bonprix.de/kategorie/damen-mode-kleider/",
       "dept": "Dresses",
       "Gender_Target": "Female"},
    {"url":"https://www.bonprix.de/kategorie/damen-schuhe/",
     "dept": "Footwear",
     "Gender_Target": "Female"},
    {"url":"https://www.bonprix.de/kategorie/herren-schuhe/",
     "dept":"Footwear",
     "Gender_Target": "Unisex"},
    {"url": "https://www.bonprix.de/kategorie/damen-damenwaesche-homewear/",
     "dept": "Home Textiles",
     "Gender_Target": "Unisex"},
    {"url": "https://www.bonprix.de/kategorie/kinder-neuh eiten-maedchen/",
     "dept": "Kidswear",
     "Gender_Target": "Female"},
    {"url": "https://www.bonprix.de/kategorie/kinder-neuheiten-jungen/",
     "dept": "Kidswear",
     "Gender_Target": "Unisex"},
    {"url": "https://www.bonprix.de/kategorie/herren-mode-pullover-und-strickjacken/",
     "dept": "Knitwear",
     "Gender_Target": "Unisex"},
    {"url": "https://www.bonprix.de/kategorie/damen-waesche-dessous/",
     "dept":"Lingerie",
     "Gender_Target": "Female"},
    {"url": "https://www.bonprix.de/kategorie/herren-mode-anzuege/",
     "dept": "Menswear",
     "Gender_Target": "Male"},
    {"url": "https://www.bonprix.de/kategorie/herren-mode-jacken/",
     "dept": "Outerwear",
     "Gender_Target": "Unisex"},
    {"url": "https://www.bonprix.de/kategorie/damen-bademode-bikini/",
     "dept": "Swimwear",
     "Gender_Target": "Female"},
        
]

def get_chrome_driver():
    """Helper to initialize a stable, crash-resistant Chrome browser instance."""
    chrome_options = ChromeOptions()
    
    # Run in headless mode (very stable in Chrome)
    chrome_options.add_argument("--headless=new") 
    
    # Stability and crash-prevention flags
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-crash-reporter")  # Stops Windows shutdown crashes
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--log-level=3")
    
    # Anti-detection flags
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("window-size=1920,1080")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Selenium 4 automatically downloads and manages the correct chromedriver
    return webdriver.Chrome(options=chrome_options)


scraped_data = {}
global_product_counter = 1

for target in DEPARTMENTS:
    url = target["url"]
    dept_name = target["dept"]

    print(f"--- Scraping Department: {dept_name} ({url}) ---")

    # Start Chrome
    driver = get_chrome_driver()

    try:
        driver.get(url)
        print("Scrolling down slowly to trigger lazy-loading...")
        
        # Scroll down in steps to force Bonprix to load all images and links
        for _ in range(8):
            driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(1.5)  # Give the site time to render the elements

        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Find all product cards (Bonprix uses 'product' or 'tile' or 'style-tile')
        product_tiles = soup.find_all(
            ["div", "article"], 
            class_=re.compile(r"product|tile|style-tile|card", re.I)
        )

        print(f"Analyzing {len(product_tiles)} potential elements...")

        for tile in product_tiles:
            try:
                # 1. Extract Link (Skip if no link exists)
                link_tag = tile.find("a", href=True)
                if not link_tag or "kategorie" in link_tag['href'] and len(link_tag['href']) < 30:
                    continue  # Skip category banners or navigation links
                
                product_url = (
                    f"https://www.bonprix.de{link_tag['href']}" 
                    if link_tag['href'].startswith("/") 
                    else link_tag['href']
                )

                # 2. Extract Name (Skip if no name exists)
                name_tag = tile.find(class_=re.compile(r"title|name|brand", re.I)) or tile.find(["h2", "h3"])
                product_name = name_tag.text.strip() if name_tag else ""
                if not product_name or len(product_name) < 3 or "€" in product_name :
                    continue

                # 3. Extract Price
                price_tag = tile.find(class_=re.compile(r"price|promo|amount", re.I))
                price = price_tag.text.strip() if price_tag else "0,00 €"

                # 4. Extract Image (Handling srcset, data-src, and src)
                img_tag = tile.find("img")
                image_url = ""
                if img_tag:
                    # Check srcset first (highest quality / modern lazy-loading)
                    srcset = img_tag.get("srcset") or img_tag.get("data-srcset")
                    if srcset:
                        # Grab the first URL from the srcset list
                        image_url = srcset.split(",")[0].split(" ")[0].strip()
                    else:
                        image_url = (
                            img_tag.get("data-src") 
                            or img_tag.get("src") 
                            or ""
                        )
                
                # Fix relative image URLs
                if image_url.startswith("//"):
                    image_url = "https:" + image_url
                elif image_url.startswith("/"):
                    image_url = "https://www.bonprix.de" + image_url

                # Skip if we still don't have an image or product URL
                if not image_url or not product_url:
                    continue

                product_id = (
                    tile.get("data-product-id") 
                    or tile.get("id") 
                    or f"bonprix_{global_product_counter}"
                )

                # Prevent duplicates
                if any(p["product_url"] == product_url for p in scraped_data.values()):
                    continue

                product_entry = {
                    "id": product_id,
                    "product_name": product_name,
                    "price": price,
                    "product_url": product_url,
                    "Department": dept_name,
                    "image_url": image_url,
                }

                scraped_data[f"product_{global_product_counter}"] = product_entry
                global_product_counter += 1

            except Exception as e:
                continue

    except Exception as e:
        print(f"Skipping department {dept_name} due to error: {e}")

    finally:
        print("Closing Chrome...")
        driver.quit()

# Save output
output_dir = pathlib.Path(__file__).resolve().parent / "Scraped_data"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "bonprix_departments.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(scraped_data, f, ensure_ascii=False, indent=4)

print(f"Successfully saved {len(scraped_data)} products to {output_file}!")