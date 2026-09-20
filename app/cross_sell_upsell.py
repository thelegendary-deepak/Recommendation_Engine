"""Cross-sell and upsell recommendation policy.

This module is intentionally independent from model loading and web-framework code.
Call ``build_recommendations`` after a recommender has produced scored candidates.
"""

from dataclasses import asdict, dataclass
import re
from statistics import median
from typing import Any, Iterable, Mapping, Optional


CROSS_SELL = "CROSS_SELL"
UPSELL = "UPSELL"


@dataclass(frozen=True)
class Recommendation:
    """The public recommendation contract."""

    productId: str
    url: str
    type: str
    score: float


@dataclass(frozen=True)
class RecommendationGenerated:
    """Event emitted after recommendations have been generated successfully."""

    event: str
    customerId: str
    recommendations: tuple[Recommendation, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "customerId": self.customerId,
            "recommendations": [asdict(item) for item in self.recommendations],
        }


def build_recommendations(
    customer_id: str,
    owned_products: Iterable[Mapping[str, Any]],
    candidates: Iterable[Mapping[str, Any]],
    *,
    top_n: Optional[int] = None,
    explicitly_configured_ids: Iterable[Any] = (),
    category_key: str = "category",
) -> tuple[list[Recommendation], RecommendationGenerated]:
    """Classify, filter, and rank scored candidates for one customer.

    A candidate is an ``UPSELL`` when it is eligible and priced above the
    department benchmark. If owned-product metadata is available, the
    benchmark is the customer's highest price in that department. Otherwise,
    the benchmark falls back to the median candidate price in that department.
    This fallback supports catalogs whose purchase and scraped-product IDs do
    not match.

    Product IDs listed in ``explicitly_configured_ids`` are allowed even when
    already owned. Without that explicit configuration, owned products are
    always excluded. Ranking is descending by the original model score.
    """
    owned = list(owned_products)
    highest_owned_price: dict[str, float] = {}

    for product in owned:
        category = _category(product, category_key)
        price = _price(product)
        highest_owned_price[category] = max(
            highest_owned_price.get(category, 0.0), price
        )

    candidate_list = list(candidates)
    department_prices: dict[str, list[float]] = {}
    for candidate in candidate_list:
        department = _category(candidate, category_key)
        price = _price(candidate)
        if department and price > 0:
            department_prices.setdefault(department, []).append(price)

    department_benchmark = {
        department: median(prices)
        for department, prices in department_prices.items()
    }

    ranked: list[tuple[float, Recommendation]] = []
    seen_ids: set[str] = set()
    for candidate in candidate_list:
        product_id = _product_id(candidate)
        if not product_id or product_id in seen_ids:
            continue
        seen_ids.add(product_id)

        category = _category(candidate, category_key)
        is_eligible_for_upsell = bool(
            candidate.get("eligible_for_upsell", candidate.get("upsell_eligible", False))
        )
        benchmark = highest_owned_price.get(
            category, department_benchmark.get(category, 0.0)
        )
        is_upsell = (
            is_eligible_for_upsell
            and benchmark > 0
            and _price(candidate) > benchmark
        )
        score = _score(candidate)
        ranked.append(
            (
                score,
                Recommendation(
                    productId=product_id,
                    url=_url(candidate),
                    type=UPSELL if is_upsell else CROSS_SELL,
                    score=score,
                ),
            )
        )

    ranked.sort(key=lambda item: item[0], reverse=True)
    if top_n is not None:
        if top_n < 0:
            raise ValueError("top_n must be zero or greater")
        ranked = ranked[:top_n]

    recommendations = [recommendation for _, recommendation in ranked]
    event = RecommendationGenerated(
        event="RecommendationGenerated",
        customerId=str(customer_id),
        recommendations=tuple(recommendations),
    )
    return recommendations, event


def _product_id(product: Mapping[str, Any]) -> str:
    value = product.get("productId", product.get("product_id", product.get("id", "")))
    return str(value) if value is not None else ""


def _category(product: Mapping[str, Any], category_key: str) -> str:
    value = product.get(category_key)
    if value is None:
        value = product.get("Department", product.get("department", ""))
    return str(value).strip().casefold()


def _url(product: Mapping[str, Any]) -> str:
    value = product.get("url", product.get("product_url", ""))
    return str(value) if value is not None else ""


def _price(product: Mapping[str, Any]) -> float:
    value = product.get("price", product.get("Product_Price", 0.0))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        match = re.search(r"(\d+(?:[\.,]\d+)?)", str(value))
        if match is None:
            return 0.0
        return float(match.group(1).replace(",", "."))
    except (TypeError, ValueError):
        return 0.0


def _score(product: Mapping[str, Any]) -> float:
    value = product.get("score", product.get("similarity_score", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
