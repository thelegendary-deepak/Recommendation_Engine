"""Retention-offer recommendation API.

Run with:
    uvicorn retention_offers.retention_offer_app:app --port 8002
"""

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from app.events.publisher import publish_safely
from app.events.schemas import OfferRecommended


app = FastAPI(title="Retention Offer Recommendation Service")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "app" / "models" / "otto_recommender_model.joblib"
CATALOG_PATH = (
    PROJECT_ROOT / "web-scraping" / "Scraped_data" / "otto_departments.json"
)
CHURN_BATCH_URL = os.getenv(
    "CHURN_BATCH_URL",
    "http://127.0.0.1:8000/churn/score/batch",
)

artifacts = joblib.load(MODEL_PATH)
content_transformer = artifacts["content_transformer"]
interactions = artifacts["interactions"]
content_matrix = artifacts["content_matrix"]
product_id_to_idx = artifacts["product_id_to_idx"]

with CATALOG_PATH.open("r", encoding="utf-8") as catalog_file:
    catalog = pd.DataFrame(list(json.load(catalog_file).values()))

aligned_catalog = pd.DataFrame(
    {
        "Product_ID": catalog["id"],
        "Product_Name": catalog["product_name"],
        "Department": catalog["Department"],
        "Product_Price": catalog["price"].str.extract(
            r"(\d+(?:[.,]\d+)?)", expand=False
        ).str.replace(",", ".", regex=False).astype(float),
    }
)

for column in [
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
    "Brand_Name",
]:
    aligned_catalog[column] = "Unknown"

catalog_matrix = content_transformer.transform(aligned_catalog)
if hasattr(catalog_matrix, "todense"):
    catalog_matrix = np.asarray(catalog_matrix.todense())


class RetentionOffer(BaseModel):
    offerId: str
    offerType: str
    offerValue: str
    recommendationScore: float
    productName: str
    productUrl: str


class RetentionOfferResponse(BaseModel):
    customerId: str
    churnScore: int
    eligible: bool
    offer: RetentionOffer | None


class BatchRetentionOfferResponse(BaseModel):
    endDate: str
    recommendations: list[RetentionOfferResponse]


def offer_type_for_score(churn_score: int) -> str | None:
    if churn_score >= 71:
        return "Discount"
    if churn_score >= 51:
        return "Limited-Time Promotion"
    if churn_score >= 31:
        return "Quick Delivery"
    return None


def offer_mask(offer_type: str) -> pd.Series:
    price_text = catalog["price"].fillna("")

    if offer_type == "Discount":
        return price_text.str.contains(r"-\s*\d+\s*%", regex=True)
    if offer_type == "Limited-Time Promotion":
        return price_text.str.contains(
            r"nur diesen Monat|nur bis [A-Za-zÄÖÜäöüß]+|nur für kurze Zeit",
            case=False,
            regex=True,
        )
    if offer_type == "Quick Delivery":
        return price_text.str.contains(
            r"in\s+1\s*-\s*2\s+Werktagen\s+bei\s+dir",
            case=False,
            regex=True,
        )
    return pd.Series(False, index=catalog.index)


def fetch_churn_scores(end_date: str) -> list[dict[str, Any]]:
    request = Request(
        CHURN_BATCH_URL,
        data=json.dumps({"endDate": end_date}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            payload: dict[str, Any] = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"Churn API error: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="Churn API is unavailable") from exc

    return payload.get("scores", [])


def recommend_offer(customer_id: str, offer_type: str) -> RetentionOffer:
    history = interactions[
        interactions["Customer_ID"].astype(str).str.casefold().eq(customer_id.casefold())
        & (interactions["score"] > 0)
    ]
    if history.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No purchase history found for customer {customer_id}",
        )

    known_history = history[history["Product_ID"].isin(product_id_to_idx)].copy()
    if known_history.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No usable purchase history found for customer {customer_id}",
        )

    model_indices = known_history["Product_ID"].map(product_id_to_idx).astype(int)
    weights = known_history["score"].to_numpy(dtype=float)
    user_profile = np.average(
        content_matrix[model_indices.to_numpy()],
        axis=0,
        weights=weights / weights.sum(),
    ).reshape(1, -1)
    similarities = cosine_similarity(user_profile, catalog_matrix).ravel()

    eligible_indices = catalog.index[offer_mask(offer_type)].to_numpy()
    if eligible_indices.size == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No catalog products support offer type {offer_type}",
        )

    best_index = int(eligible_indices[np.argmax(similarities[eligible_indices])])
    product = catalog.iloc[best_index]
    return RetentionOffer(
        offerId=str(product["id"]),
        offerType=offer_type,
        offerValue=str(product["price"]),
        recommendationScore=round(float(similarities[best_index]), 4),
        productName=str(product["product_name"]),
        productUrl=str(product["product_url"]),
    )


@app.get(
    "/recommendations/offers",
    response_model=BatchRetentionOfferResponse,
)
def get_retention_offers(
    end_date: str = Query(..., description="Churn snapshot date in YYYY-MM-DD format"),
):
    recommendations = []

    for churn_result in fetch_churn_scores(end_date):
        customer_id = str(churn_result["customerId"])
        churn_score = int(churn_result["churnScore"])
        offer_type = offer_type_for_score(churn_score)
        offer = recommend_offer(customer_id, offer_type) if offer_type else None
        response = RetentionOfferResponse(
            customerId=customer_id,
            churnScore=churn_score,
            eligible=offer is not None,
            offer=offer,
        )
        recommendations.append(response)

        if response.eligible:
            publish_safely(
                OfferRecommended(
                    customerId=response.customerId,
                    recommendationId=response.offer.offerId,
                    model="RetentionRules+CosineSimilarity",
                )
            )

    return BatchRetentionOfferResponse(
        endDate=end_date,
        recommendations=recommendations,
    )


@app.get("/")
def health_check():
    return {"status": "ok"}