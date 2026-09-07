import json
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. Loading exported JOBLIB model artifacts (Otto version)
artifacts = joblib.load("./app/models/otto_recommender_model.joblib")
transformer = artifacts["content_transformer"]
interactions = artifacts["interactions"]
cb_matrix = artifacts["content_matrix"]          
cb_id_to_idx = artifacts["product_id_to_idx"]    # Changed key from 'crate_barrel_product_id_to_idx' to 'product_id_to_idx'

# 2. Load your freshly scraped Otto JSON products
with open("./web-scraping/Scraped_data/otto_departments.json", "r", encoding="utf-8") as f:
    scraped_data = json.load(f)

# Convert JSON dict-of-dicts to DataFrame
json_df = pd.DataFrame(list(scraped_data.values()))

# Helper to clean price ("17,90 €" -> 17.90)
def clean_price(val):
    if pd.isna(val):
        return 0.0
    # Search for the first occurrence of a number (supports integers and decimals with commas/dots)
    match = re.search(r"(\d+(?:[\.,]\d+)?)", str(val))
    if match:
        cleaned = match.group(1).replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0

# 3. ALIGN: Map JSON fields to the Otto Excel schema columns
aligned_df = pd.DataFrame()
aligned_df["Product_ID"] = json_df["id"]
aligned_df["Product_Name"] = json_df["product_name"]
aligned_df["Department"] = json_df["Department"]  # Mapped to Otto's 'Department'
aligned_df["Brand_Name"] = json_df["manufacturer"]
aligned_df["Product_Price"] = json_df["price"].apply(clean_price)

# Safely map Gender_Target if it exists in the scraped JSON
if "Gender_Target" in json_df.columns:
    aligned_df["Gender_Target"] = json_df["Gender_Target"].fillna("Unknown")
else:
    aligned_df["Gender_Target"] = "Unknown"

# Fill columns that Otto Excel had but JSON doesn't with "Unknown"
missing_cols = [
    "Product_Category",
    "Product_Sub_Category",
    "Product_Type",
    "Color",
    "Material",
    "Style",
    "Season",
    "Occasion_Tag"
]
for col in missing_cols:
    aligned_df[col] = "Unknown"

# 4. TRANSFORM: Convert JSON products into the same mathematical format
json_matrix = transformer.transform(aligned_df)
json_matrix = (
    np.asarray(json_matrix.todense())
    if hasattr(json_matrix, "todense")
    else json_matrix
)

# 5. RECOMMEND: Find the best JSON product for an Otto Customer
def recommend_json_to_customer(customer_id: str, top_n: int = 3):
    # Get customer's positive Otto interactions
    cust_history = interactions[
        (interactions["Customer_ID"] == customer_id)
        & (interactions["score"] > 0)
    ]
    if cust_history.empty:
        return "New customer: No history to base recommendations on."

    # Build the Otto User Taste Profile
    # Try converting product IDs to string/int to avoid type mismatch
    idxs = (
        cust_history["Product_ID"]
        .map(cb_id_to_idx)
        .dropna()
        .astype(int)
        .to_numpy()
    )
    weights = cust_history.loc[
        cust_history["Product_ID"].isin(cb_id_to_idx), "score"
    ].to_numpy()
    weights = weights / weights.sum()

    user_profile = np.average(cb_matrix[idxs], axis=0, weights=weights).reshape(
        1, -1
    )

    # Compare User Profile against the JSON products matrix
    similarities = cosine_similarity(user_profile, json_matrix).ravel()

    # Get top N matches from the JSON data
    top_indices = np.argsort(similarities)[::-1][:top_n]

    results = json_df.iloc[top_indices].copy()
    results["similarity_score"] = similarities[top_indices]

    results["cleaned_price"] = results["price"].apply(clean_price)

    # Return using Otto's specific JSON columns
    return results[
        ["id", "product_name", "Department", "cleaned_price", "similarity_score"]
    ]

# --- RUN TEST ---
# Put any active Customer_ID from your Otto Excel dataset here:
sample_customer = interactions["Customer_ID"].iloc[0]
print(f"Top JSON recommendations for Customer {sample_customer}:")
print(recommend_json_to_customer(sample_customer, top_n=1))