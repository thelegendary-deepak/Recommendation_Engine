"""Event contracts shared by recommendation services."""

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class RecommendationType(str, Enum):
    PRODUCT = "PRODUCT"
    CROSS_SELL = "CROSS_SELL"
    UPSELL = "UPSELL"
    RETENTION_OFFER = "RETENTION_OFFER"


class RecommendationEvent(BaseModel):
    eventId: str = Field(default_factory=lambda: f"EVT-{uuid4()}")
    eventType: Literal["RecommendationGenerated", "OfferRecommended"]
    customerId: str
    recommendationId: str
    recommendationType: RecommendationType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str | None = None


class RecommendationGenerated(RecommendationEvent):
    eventType: Literal["RecommendationGenerated"] = "RecommendationGenerated"


class OfferRecommended(RecommendationEvent):
    eventType: Literal["OfferRecommended"] = "OfferRecommended"
    recommendationType: Literal[
        RecommendationType.RETENTION_OFFER
    ] = RecommendationType.RETENTION_OFFER