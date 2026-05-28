from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from sklearn.metrics.pairwise import cosine_similarity

from mmr import mmr_rerank


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "sample"
ORDERS_PATH = DATA_DIR / "orders_sample.csv"
PRODUCTS_PATH = DATA_DIR / "products_sample.csv"
MARGINS_PATH = DATA_DIR / "product_margins_sample.csv"


@dataclass
class RecommendationEngine:
    orders_path: Path = ORDERS_PATH
    products_path: Path = PRODUCTS_PATH
    margins_path: Path = MARGINS_PATH
    min_support: float = 0.08
    min_confidence: float = 0.25
    min_lift: float = 1.0

    def __post_init__(self) -> None:
        self.orders = pd.read_csv(self.orders_path)
        self.products = pd.read_csv(self.products_path)
        self.margins = pd.read_csv(self.margins_path)
        self.transactions = self._build_transactions()
        self.transaction_matrix = self._build_transaction_matrix()
        self.rules = self._build_association_rules()
        self.item_similarity = self._build_item_similarity()

    def _build_transactions(self) -> list[list[str]]:
        grouped = (
            self.orders.groupby("order_id")["product_name"]
            .apply(lambda values: sorted(set(str(v).strip().lower() for v in values)))
            .tolist()
        )
        return grouped

    def _build_transaction_matrix(self) -> pd.DataFrame:
        encoder = TransactionEncoder()
        encoded = encoder.fit(self.transactions).transform(self.transactions)
        return pd.DataFrame(encoded, columns=encoder.columns_)

    def _build_association_rules(self) -> pd.DataFrame:
        frequent_itemsets = fpgrowth(
            self.transaction_matrix,
            min_support=self.min_support,
            use_colnames=True,
        )

        if frequent_itemsets.empty:
            return pd.DataFrame(
                columns=["antecedents", "consequents", "support", "confidence", "lift"]
            )

        rules = association_rules(
    frequent_itemsets,
    num_itemsets=len(frequent_itemsets),
    metric="confidence",
    min_threshold=self.min_confidence
)
        rules = rules[rules["lift"] >= self.min_lift].copy()
        rules["antecedent_len"] = rules["antecedents"].apply(len)
        rules["consequent_len"] = rules["consequents"].apply(len)
        rules = rules[rules["consequent_len"] == 1].copy()
        return rules.sort_values(["lift", "confidence", "support"], ascending=False)

    def _build_item_similarity(self) -> dict[tuple[str, str], float]:
        item_order_matrix = self.transaction_matrix.T.astype(int)
        product_names = item_order_matrix.index.tolist()
        sim = cosine_similarity(item_order_matrix.values)

        similarities: dict[tuple[str, str], float] = {}
        for i, left in enumerate(product_names):
            for j, right in enumerate(product_names):
                if left != right:
                    similarities[(left, right)] = float(sim[i, j])
        return similarities

    def available_products(self) -> list[str]:
        return sorted(self.products["product_name"].str.lower().unique().tolist())

    def get_rules_for_cart(self, cart_products: list[str]) -> pd.DataFrame:
        cart = set(product.strip().lower() for product in cart_products)
        if not cart or self.rules.empty:
            return pd.DataFrame()

        def antecedent_matches(antecedents: frozenset) -> bool:
            return set(antecedents).issubset(cart)

        matched = self.rules[self.rules["antecedents"].apply(antecedent_matches)].copy()
        matched = matched[~matched["consequents"].apply(lambda x: list(x)[0] in cart)]
        return matched

    def _collaborative_score(self, candidate: str, cart_products: list[str]) -> float:
        if not cart_products:
            return 0.0
        scores = [
            self.item_similarity.get((candidate, cart_item), self.item_similarity.get((cart_item, candidate), 0.0))
            for cart_item in cart_products
        ]
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _normalize(values: pd.Series) -> pd.Series:
        if values.empty:
            return values
        min_value = values.min()
        max_value = values.max()
        if max_value == min_value:
            return pd.Series([1.0] * len(values), index=values.index)
        return (values - min_value) / (max_value - min_value)

    def recommend(
        self,
        cart_products: list[str],
        top_k: int = 5,
        lambda_mmr: float = 0.75,
    ) -> list[dict]:
        cart_products = [product.strip().lower() for product in cart_products if product]
        matched_rules = self.get_rules_for_cart(cart_products)

        candidates: dict[str, dict] = {}

        for _, row in matched_rules.iterrows():
            product = list(row["consequents"])[0]
            margin_row = self.margins[self.margins["product_name"].str.lower() == product]
            margin = float(margin_row["margin"].iloc[0]) if not margin_row.empty else 0.20
            collaborative = self._collaborative_score(product, cart_products)

            current = candidates.get(product)
            rule_payload = {
                "product_name": product,
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
                "margin": margin,
                "collaborative_score": collaborative,
                "antecedents": sorted(list(row["antecedents"])),
            }

            if current is None or rule_payload["confidence"] > current["confidence"]:
                candidates[product] = rule_payload

        # Fallback: item-item recommendations when no rule matches perfectly.
        if not candidates:
            for product in self.available_products():
                if product in cart_products:
                    continue
                collaborative = self._collaborative_score(product, cart_products)
                if collaborative <= 0:
                    continue
                margin_row = self.margins[self.margins["product_name"].str.lower() == product]
                margin = float(margin_row["margin"].iloc[0]) if not margin_row.empty else 0.20
                candidates[product] = {
                    "product_name": product,
                    "support": 0.0,
                    "confidence": 0.0,
                    "lift": 0.0,
                    "margin": margin,
                    "collaborative_score": collaborative,
                    "antecedents": cart_products,
                }

        if not candidates:
            return []

        df = pd.DataFrame(candidates.values())
        df["confidence_norm"] = self._normalize(df["confidence"])
        df["lift_norm"] = self._normalize(df["lift"])
        df["margin_norm"] = self._normalize(df["margin"])
        df["collaborative_norm"] = self._normalize(df["collaborative_score"])
        df["final_score"] = (
            0.35 * df["confidence_norm"]
            + 0.25 * df["lift_norm"]
            + 0.20 * df["margin_norm"]
            + 0.20 * df["collaborative_norm"]
        )

        df = df.sort_values("final_score", ascending=False)
        reranked = mmr_rerank(
            df.to_dict("records"),
            similarity_matrix=self.item_similarity,
            top_k=top_k,
            lambda_param=lambda_mmr,
        )
        return reranked

    @staticmethod
    def explanation(rec: dict) -> str:
        antecedents = " i ".join(rec.get("antecedents", []))
        product = rec["product_name"]
        confidence_pct = rec.get("confidence", 0.0) * 100
        if rec.get("confidence", 0.0) > 0:
            return (
                f"Kupci koji uzmu {antecedents} kupe {product} "
                f"u {confidence_pct:.1f}% slučajeva. Lift je {rec.get('lift', 0.0):.2f}, "
                f"što znači da je veza jača od slučajne kupovine."
            )
        return (
            f"{product} je preporučen zato što se često pojavljuje u korpama sličnim "
            f"trenutnoj korpi i ima solidan poslovni score."
        )
