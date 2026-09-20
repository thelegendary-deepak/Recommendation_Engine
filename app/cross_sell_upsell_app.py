"""Dedicated Cross-Sell and Upsell Recommendation API.

Standalone FastAPI service that returns exactly one cross-sell and one upsell
recommendation per customer. Run with:

    uvicorn app.cross_sell_upsell_app:app --port 8001
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from app.cross_sell_upsell import build_recommendations
from app.events.publisher import publish_safely
from app.events.schemas import RecommendationGenerated, RecommendationType

app = FastAPI(title="Cross-Sell and Upsell Recommendation Service")

# --- 1. Dynamic Path Resolution ---
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Otto Paths
MODEL_PATH_OTTO = BASE_DIR / "models" / "otto_recommender_model.joblib"
JSON_PATH_OTTO = PROJECT_ROOT / "web-scraping" / "Scraped_data" / "otto_departments.json"

# Bonprix Paths
MODEL_PATH_BONPRIX = BASE_DIR / "models" / "bonprix_recommender_model.joblib"
JSON_PATH_BONPRIX = PROJECT_ROOT / "web-scraping" / "Scraped_data" / "bonprix_departments.json"

# Crate & Barrel Paths
MODEL_PATH_CB = BASE_DIR / "models" / "crateandbarrel_recommender_model.joblib"
JSON_PATH_CB = PROJECT_ROOT / "web-scraping" / "Scraped_data" / "product_metadata.json"


# --- 2. Load Models & Data at Startup ---
try:
    # Load Otto
    otto_artifacts = joblib.load(MODEL_PATH_OTTO)
    otto_transformer = otto_artifacts["content_transformer"]
    otto_interactions = otto_artifacts["interactions"]
    otto_matrix = otto_artifacts["content_matrix"]
    otto_id_to_idx = otto_artifacts["product_id_to_idx"]

    with open(JSON_PATH_OTTO, "r", encoding="utf-8") as f:
        otto_json_df = pd.DataFrame(list(json.load(f).values()))

    # Load Bonprix
    bonprix_artifacts = joblib.load(MODEL_PATH_BONPRIX)
    bonprix_transformer = bonprix_artifacts["content_transformer"]
    bonprix_interactions = bonprix_artifacts["interactions"]
    bonprix_matrix = bonprix_artifacts["content_matrix"]
    bonprix_id_to_idx = bonprix_artifacts["product_id_to_idx"]

    with open(JSON_PATH_BONPRIX, "r", encoding="utf-8") as f:
        bonprix_json_df = pd.DataFrame(list(json.load(f).values()))

    # Load Crate & Barrel
    cb_artifacts = joblib.load(MODEL_PATH_CB)
    cb_transformer = cb_artifacts["content_transformer"]
    cb_interactions = cb_artifacts["interactions"]
    cb_matrix = cb_artifacts["content_matrix"]
    cb_id_to_idx = cb_artifacts["product_id_to_idx"]

    with open(JSON_PATH_CB, "r", encoding="utf-8") as f:
        cb_json_df = pd.DataFrame(list(json.load(f).values()))

except Exception as e:
    raise RuntimeError(f"Startup loading failed: {e}")


# --- 3. Price Cleaning ---
def clean_price_robust(val):
    r"""Robust price cleaning utilizing the regex `(\d+(?:[\.,]\d+)?)` to match any first decimal."""
    if pd.isna(val):
        return 0.0
    match = re.search(r"(\d+(?:[\.,]\d+)?)", str(val))
    if match:
        cleaned = match.group(1).replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


# --- 4. Align Catalogs ---
aligned_otto = pd.DataFrame()
aligned_otto["Product_ID"] = otto_json_df["id"]
aligned_otto["Product_Name"] = otto_json_df["product_name"]
aligned_otto["Department"] = otto_json_df["Department"]
aligned_otto["Product_Price"] = otto_json_df["price"].apply(clean_price_robust)

aligned_bonprix = pd.DataFrame()
aligned_bonprix["Product_ID"] = bonprix_json_df["id"]
aligned_bonprix["Product_Name"] = bonprix_json_df["product_name"]
aligned_bonprix["Department"] = bonprix_json_df["Department"]
aligned_bonprix["Product_Price"] = bonprix_json_df["price"].apply(clean_price_robust)

aligned_cb = pd.DataFrame()
aligned_cb["Product_ID"] = cb_json_df["id"]
aligned_cb["Product_Name"] = cb_json_df["product_name"]
aligned_cb["Product_Category"] = cb_json_df["category"]
aligned_cb["Product_Price"] = cb_json_df["price"].apply(clean_price_robust)

# Shared missing columns for alignment
all_missing_cols = [
    "Product_Category",
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
    "Department",
    "Brand_Name"
]

for col in all_missing_cols:
    if col not in aligned_otto.columns:
        aligned_otto[col] = "Unknown"
    if col not in aligned_bonprix.columns:
        aligned_bonprix[col] = "Unknown"
    if col not in aligned_cb.columns:
        aligned_cb[col] = "Unknown"

# Pre-transform matrices
otto_json_matrix = otto_transformer.transform(aligned_otto)
otto_json_matrix = (
    np.asarray(otto_json_matrix.todense())
    if hasattr(otto_json_matrix, "todense")
    else otto_json_matrix
)

bonprix_json_matrix = bonprix_transformer.transform(aligned_bonprix)
bonprix_json_matrix = (
    np.asarray(bonprix_json_matrix.todense())
    if hasattr(bonprix_json_matrix, "todense")
    else bonprix_json_matrix
)

cb_json_matrix = cb_transformer.transform(aligned_cb)
cb_json_matrix = (
    np.asarray(cb_json_matrix.todense())
    if hasattr(cb_json_matrix, "todense")
    else cb_json_matrix
)


# --- 5. Pydantic Schemas ---
class Recommendation(BaseModel):
    productId: str
    name: str
    url: str
    type: str
    score: float
    reason: str


class CrossSellUpsellResponse(BaseModel):
    customerId: str
    brand: str
    cross_sell: Recommendation
    upsell: Recommendation


# --- 6. Recommendation Endpoint ---
@app.get("/recommendations/customer/{customer_id}", response_model=CrossSellUpsellResponse)
def get_cross_sell_and_upsell(
    customer_id: str,
    brand: str = Query(
        "otto",
        description="Brand: 'otto', 'bonprix', or 'crate and barrel'",
    ),
):
    """Get exactly one cross-sell and one upsell recommendation for a customer.
    
    Args:
        customer_id: The customer identifier (e.g., CUST3141)
        brand: The brand catalog ('otto', 'bonprix', 'crate and barrel')
    
    Returns:
        One cross-sell and one upsell recommendation with product details.
    """
    brand = brand.lower().strip()

    # Route variables based on selected brand
    if brand == "otto":
        interactions = otto_interactions
        id_to_idx = otto_id_to_idx
        model_matrix = otto_matrix
        target_matrix = otto_json_matrix
        df_display = otto_json_df
        category_col = "Department"
    elif brand == "bonprix":
        interactions = bonprix_interactions
        id_to_idx = bonprix_id_to_idx
        model_matrix = bonprix_matrix
        target_matrix = bonprix_json_matrix
        df_display = bonprix_json_df
        category_col = "Department"
    elif brand == "crate and barrel":
        interactions = cb_interactions
        id_to_idx = cb_id_to_idx
        model_matrix = cb_matrix
        target_matrix = cb_json_matrix
        df_display = cb_json_df
        category_col = "Product_Category"
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid brand. Use 'otto', 'bonprix', or 'crate and barrel'.",
        )

    # Format customer ID
    search_id = int(customer_id) if customer_id.isdigit() else customer_id

    # Get customer history
    cust_history = interactions[
        (interactions["Customer_ID"] == search_id) & (interactions["score"] > 0)
    ]

    if cust_history.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No purchase history found for customer {customer_id}.",
        )

    try:
        # Build user profile
        idxs = (
            cust_history["Product_ID"]
            .map(id_to_idx)
            .dropna()
            .astype(int)
            .to_numpy()
        )
        weights = cust_history.loc[
            cust_history["Product_ID"].isin(id_to_idx), "score"
        ].to_numpy()
        weights = weights / weights.sum()

        user_profile = np.average(
            model_matrix[idxs], axis=0, weights=weights
        ).reshape(1, -1)

        # Calculate similarity
        similarities = cosine_similarity(user_profile, target_matrix).ravel()

        # Build candidates with URLs
        candidates = []
        for idx, product in df_display.iterrows():
            candidates.append(
                {
                    "id": product["id"],
                    "productId": product["id"],
                    "category": product.get(category_col, ""),
                    "price": product.get("price", 0.0),
                    "url": product.get("product_url", ""),
                    "score": round(float(similarities[idx]), 2),
                    "eligible_for_upsell": True,
                }
            )

        # Classify and rank
        policy_recommendations, _ = build_recommendations(
            str(customer_id),
            [],
            candidates,
            top_n=None,
            category_key="category",
        )

        # Split into cross-sell and upsell
        cross_sells = [item for item in policy_recommendations if item.type == "CROSS_SELL"]
        upsells = [item for item in policy_recommendations if item.type == "UPSELL"]

        if not cross_sells:
            raise HTTPException(
                status_code=404,
                detail=f"No cross-sell recommendations found for customer {customer_id}.",
            )
        if not upsells:
            raise HTTPException(
                status_code=404,
                detail=f"No upsell recommendations found for customer {customer_id}.",
            )

        # Take the top recommendation from each type
        top_cross_sell = cross_sells[0]
        top_upsell = upsells[0]

        # Build response with product details
        cross_sell_response = Recommendation(
            productId=top_cross_sell.productId,
            name=str(
                df_display.loc[
                    df_display["id"].astype(str) == top_cross_sell.productId,
                    "product_name",
                ].iloc[0]
            ),
            url=top_cross_sell.url,
            type=top_cross_sell.type,
            score=top_cross_sell.score,
            reason="Complementary product",
        )

        upsell_response = Recommendation(
            productId=top_upsell.productId,
            name=str(
                df_display.loc[
                    df_display["id"].astype(str) == top_upsell.productId,
                    "product_name",
                ].iloc[0]
            ),
            url=top_upsell.url,
            type=top_upsell.type,
            score=top_upsell.score,
            reason="Higher-priced option in the same category",
        )

        response = CrossSellUpsellResponse(
            customerId=str(customer_id),
            brand=brand,
            cross_sell=cross_sell_response,
            upsell=upsell_response,
        )
        publish_safely(
            RecommendationGenerated(
                customerId=response.customerId,
                recommendationId=response.cross_sell.productId,
                recommendationType=RecommendationType.CROSS_SELL,
                model="CosineSimilarity",
            )
        )
        publish_safely(
            RecommendationGenerated(
                customerId=response.customerId,
                recommendationId=response.upsell.productId,
                recommendationType=RecommendationType.UPSELL,
                model="CosineSimilarity",
            )
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Recommendation generation failed: {e}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
