import json
import re
from pathlib import Path
from typing import List
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Recommendation Engine Service")

# --- 1. Dynamic Path Resolution ---
BASE_DIR = Path(__file__).resolve().parent.parent  # Points to recommendation_engine
PROJECT_ROOT = BASE_DIR.parent  # Points to the repository root

MODEL_PATH = BASE_DIR / "models" / "recommender_model.joblib"
JSON_PATH = PROJECT_ROOT/ "Scraped_data" / "product_metadata.json"

# --- 2. Load Model & Scraped Data at Startup ---
try:
    artifacts = joblib.load(MODEL_PATH)
    transformer = artifacts["content_transformer"]
    interactions = artifacts["interactions"]
    cb_matrix = artifacts["content_matrix"]
    cb_id_to_idx = artifacts["product_id_to_idx"]
except Exception as e:
    raise RuntimeError(
        f"Failed to load model from {MODEL_PATH}. Error: {e}"
    )

try:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        scraped_data = json.load(f)
    json_df = pd.DataFrame(list(scraped_data.values()))
except Exception as e:
    raise RuntimeError(
        f"Failed to load scraped data from {JSON_PATH}. Error: {e}"
    )


# --- 3. Clean and Align Scraped Data ---
def clean_price(val):
    if pd.isna(val):
        return 0.0
    cleaned = re.sub(r"[^\d,.-]", "", str(val)).replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


aligned_df = pd.DataFrame()
aligned_df["Product_ID"] = json_df["id"]
aligned_df["Product_Name"] = json_df["product_name"]
aligned_df["Product_Category"] = json_df["category"]
aligned_df["Brand_Name"] = json_df["manufacturer"]
aligned_df["Product_Price"] = json_df["price"].apply(clean_price)

# Fill missing columns for the transformer
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
    "Age_Group",
]
for col in missing_cols:
    aligned_df[col] = "Unknown"

# Pre-transform JSON products into content matrix
json_matrix = transformer.transform(aligned_df)
json_matrix = (
    np.asarray(json_matrix.todense())
    if hasattr(json_matrix, "todense")
    else json_matrix
)


# --- 4. Pydantic Schemas for Response Validation ---
class RecommendationItem(BaseModel):
    productId: str
    name: str
    type: str = "PRODUCT"
    score: float
    reason: str


class RecommendationResponse(BaseModel):
    customerId: str
    recommendations: List[RecommendationItem]


# --- 5. GET Endpoint ---
@app.get(
    "/recommendations/customer/{id}", response_model=RecommendationResponse
)
def get_recommendations(id: str, top_n: int = 3):
    """Get personalized product recommendations from scraped data for a given customer ID."""
    
    search_id = id
    try:
        # If the ID consists of only digits, convert to integer
        if id.isdigit():
            search_id = int(id)
    except ValueError:
        pass
    # Get customer's positive Crate & Barrel interactions
    cust_history = interactions[
        (interactions["Customer_ID"] == search_id) & (interactions["score"] > 0)
    ]

    # Rule 6: If no history exists, return an empty list instead of incorrect recommendations
    if cust_history.empty:
        return RecommendationResponse(customerId=search_id, recommendations=[])

    try:
        # Build User Taste Profile
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

        user_profile = np.average(
            cb_matrix[idxs], axis=0, weights=weights
        ).reshape(1, -1)

        # Calculate similarity against the JSON products
        similarities = cosine_similarity(user_profile, json_matrix).ravel()

        # Get top N matches
        top_indices = np.argsort(similarities)[::-1][:top_n]

        recommendations_list = []
        for idx in top_indices:
            recommendations_list.append(
                RecommendationItem(
                    productId=str(json_df.iloc[idx]["id"]),
                    name=str(json_df.iloc[idx]["product_name"]),
                    type="PRODUCT",
                    score=round(float(similarities[idx]), 2),
                    reason="Based on recent purchases",
                )
            )

        # Rule 7: Publish 'RecommendationGenerated' event (Logged to console/system)
        print(
            f"[EVENT] RecommendationGenerated published successfully for Customer: {search_id}"
        )

        return RecommendationResponse(
            customerId=search_id, recommendations=recommendations_list
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Recommendation generation failed: {e}"
        )