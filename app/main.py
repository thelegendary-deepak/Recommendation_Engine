import json
import re
from pathlib import Path
from typing import List, Literal
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from app.events.publisher import publish_safely
from app.events.schemas import RecommendationGenerated, RecommendationType

app = FastAPI(title="Multi-Brand Recommendation Engine Service")

# --- 1. Dynamic Path Resolution ---
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # Points to the repository root

# Crate & Barrel Paths
MODEL_PATH_CB = BASE_DIR / "models" / "crateandbarrel_recommender_model.joblib"
JSON_PATH_CB = PROJECT_ROOT / "web-scraping" / "Scraped_data" / "product_metadata.json"

# Otto Paths
MODEL_PATH_OTTO = BASE_DIR / "models" / "otto_recommender_model.joblib"
JSON_PATH_OTTO = PROJECT_ROOT / "web-scraping" / "Scraped_data" / "otto_departments.json"

#Bonprix Paths
MODEL_PATH_BONPRIX = BASE_DIR / "models" / "bonprix_recommender_model.joblib"
JSON_PATH_BONPRIX = PROJECT_ROOT / "web-scraping" / "Scraped_data" / "bonprix_departments.json"


# --- 2. Load Models & Data at Startup ---
try:
    # Load Crate & Barrel (CB)
    cb_artifacts = joblib.load(MODEL_PATH_CB)
    cb_transformer = cb_artifacts["content_transformer"]
    cb_interactions = cb_artifacts["interactions"]
    cb_matrix = cb_artifacts["content_matrix"]
    cb_id_to_idx = cb_artifacts["product_id_to_idx"]

    with open(JSON_PATH_CB, "r", encoding="utf-8") as f:
        cb_json_df = pd.DataFrame(list(json.load(f).values()))

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

except Exception as e:
    raise RuntimeError(f"Startup loading failed: {e}")


# --- 3. Clean and Align Functions ---
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


# Setup Crate & Barrel Matrix
aligned_cb = pd.DataFrame()
aligned_cb["Product_ID"] = cb_json_df["id"]
aligned_cb["Product_Name"] = cb_json_df["product_name"]
aligned_cb["Product_Category"] = cb_json_df["category"]
aligned_cb["Brand_Name"] = cb_json_df["manufacturer"]
aligned_cb["Product_Price"] = cb_json_df["price"].apply(clean_price_robust)

# Setup Otto Matrix
aligned_otto = pd.DataFrame()
aligned_otto["Product_ID"] = otto_json_df["id"]
aligned_otto["Product_Name"] = otto_json_df["product_name"]
aligned_otto["Department"] = otto_json_df["Department"]
aligned_otto["Brand_Name"] = otto_json_df["manufacturer"]
aligned_otto["Product_Price"] = otto_json_df["price"].apply(clean_price_robust)
aligned_otto["Gender_Target"] = (
    otto_json_df["Gender_Target"].fillna("Unknown")
    if "Gender_Target" in otto_json_df.columns
    else "Unknown"
)
# Setup Bonprix Matrix
aligned_bonprix = pd.DataFrame()
aligned_bonprix["Product_ID"] = bonprix_json_df["id"]
aligned_bonprix["Product_Name"] = bonprix_json_df["product_name"]
aligned_bonprix["Department"] = bonprix_json_df["Department"]
aligned_bonprix["Product_Price"] = bonprix_json_df["price"].apply(clean_price_robust)
aligned_bonprix["Gender_Target"] = (
    bonprix_json_df["Gender_Target"].fillna("Unknown")
    if "Gender_Target" in bonprix_json_df.columns
    else "Unknown"
)


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
    if col not in aligned_cb.columns:
        aligned_cb[col] = "Unknown"
    if col not in aligned_otto.columns:
        aligned_otto[col] = "Unknown"
    if col not in aligned_bonprix.columns:
        aligned_bonprix[col] = "Unknown"

# Pre-transform both matrices
cb_json_matrix = cb_transformer.transform(aligned_cb)
cb_json_matrix = (
    np.asarray(cb_json_matrix.todense())
    if hasattr(cb_json_matrix, "todense")
    else cb_json_matrix
)

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

# --- 4. Pydantic Schemas ---
class RecommendationItem(BaseModel):
    productId: str
    name: str
    url: str
    type: Literal["PRODUCT"]
    score: float
    reason: str


class RecommendationResponse(BaseModel):
    brand: str
    customerId: str
    recommendations: List[RecommendationItem]


def classify_recommendation(
    candidate: pd.Series,
    owned_products: pd.DataFrame,
    category_col: str,
) -> str:
    """Classify an unseen candidate using the customer's known catalog items."""
    candidate_category = str(candidate.get(category_col, "Unknown"))
    if not candidate_category or candidate_category == "Unknown":
        return "CROSS_SELL"

    comparable = owned_products[
        owned_products[category_col].fillna("Unknown").astype(str)
        == candidate_category
    ]
    if comparable.empty:
        return "CROSS_SELL"

    candidate_price = clean_price_robust(candidate.get("price"))
    highest_owned_price = comparable["price"].apply(clean_price_robust).max()
    if candidate_price > highest_owned_price:
        return "UPSELL"
    return "CROSS_SELL"


# --- 5. Multi-Brand GET Endpoint ---
@app.get(
    "/recommendations/customer/{id}", response_model=RecommendationResponse
)
def get_recommendations(
    id: str,
    brand: str = Query("otto", description="Select brand: 'crate and barrel' or 'otto' or 'bonprix'"),
    top_n: int = 3,
):
    """Get personalized recommendations from scraped data. Choose brand using the 'brand' parameter."""
    brand = brand.lower().strip()

    # Route variables based on selected brand
    if brand == "crate and barrel":
        interactions = cb_interactions
        id_to_idx = cb_id_to_idx
        model_matrix = cb_matrix
        target_matrix = cb_json_matrix
        df_display = cb_json_df
        category_col = "category"

    elif brand == "otto":
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
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid brand choice. Use 'crate and barrel' or 'otto' or 'bonprix'.",
        )

    # Format ID type
    search_id = int(id) if id.isdigit() else id

    # Get customer history
    cust_history = interactions[
        (interactions["Customer_ID"] == search_id) & (interactions["score"] > 0)
    ]

    if cust_history.empty:
        return RecommendationResponse(
            brand=brand, customerId=str(id), recommendations=[]
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
        top_indices = np.argsort(similarities)[::-1][:top_n]

        recommendations_list = []
        for idx in top_indices:
            recommendations_list.append(
                RecommendationItem(
                    productId=str(df_display.iloc[idx]["id"]),
                    name=str(df_display.iloc[idx]["product_name"]),
                    url=str(df_display.iloc[idx].get("product_url", "")),
                    type="PRODUCT",
                    score=round(float(similarities[idx]), 2),
                    reason="Based on recent purchases",
                )
            )

        response = RecommendationResponse(
            brand=brand,
            customerId=str(id),
            recommendations=recommendations_list,
        )
        for recommendation in response.recommendations:
            publish_safely(
                RecommendationGenerated(
                    customerId=response.customerId,
                    recommendationId=recommendation.productId,
                    recommendationType=RecommendationType.PRODUCT,
                    model="CosineSimilarity",
                )
            )
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Recommendation generation failed: {e}"
        )