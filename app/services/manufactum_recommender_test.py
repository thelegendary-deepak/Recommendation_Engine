import json
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load your exported JOBLIB model artifacts
artifacts = joblib.load("C:\\Users\\DRUPAKUL\\OneDrive - Otto Group\\Documents\\GitHub\\Own Repo\\Recommendation_Engine\\app\\models\\recommender_model.joblib")
transformer = artifacts["content_transformer"]
interactions = artifacts["interactions"]
cb_matrix = artifacts["content_matrix"]
cb_id_to_idx = artifacts["product_id_to_idx"]

# 2. Load your scraped JSON products
with open("./Scraped_data/product_metadata.json", "r", encoding="utf-8") as f:
    scraped_data = json.load(f)

# Convert JSON dict-of-dicts to DataFrame
json_df = pd.DataFrame(list(scraped_data.values()))


# Helper to clean price ("17,90 €" -> 17.90)
def clean_price(val):
    if pd.isna(val):
        return 0.0
    cleaned = re.sub(r"[^\d,.-]", "", str(val)).replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


# 3. ALIGN: Map JSON fields to the Excel schema columns
aligned_df = pd.DataFrame()
aligned_df["Product_ID"] = json_df["id"]
aligned_df["Product_Name"] = json_df["product_name"]
aligned_df["Product_Category"] = json_df["category"]
aligned_df["Brand_Name"] = json_df["manufacturer"]
aligned_df["Product_Price"] = json_df["price"].apply(clean_price)

# Fill columns that Excel had but JSON doesn't with "Unknown"
missing_cols = [
    "Product_Sub_Category",
    "Room_Department",
    "Product_Type",
    "Color",
    "Material",
    "Style",
    "Pattern",
    "Occasion_Tag",
    "Season",
    "Gender_Target",
    "Fit_Type",
    "Age_Group"
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


# 5. RECOMMEND: Find the best JSON product for a Crate & Barrel Customer
def recommend_json_to_customer(customer_id: str, top_n: int = 3):
    # Get customer's positive Crate & Barrel interactions
    cust_history = interactions[
        (interactions["Customer_ID"] == customer_id)
        & (interactions["score"] > 0)
    ]

    if cust_history.empty:
        return "New customer: No history to base recommendations on."

    # Build the Crate & Barrel User Taste Profile
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

    return results[
        ["id", "product_name", "category", "price", "similarity_score"]
    ]


# --- RUN TEST ---
# Put any active Customer_ID from your Excel dataset here:
sample_customer = interactions["Customer_ID"].iloc[0]
print(f"Top JSON recommendations for Customer {sample_customer}:")
print(recommend_json_to_customer(sample_customer, top_n=1))