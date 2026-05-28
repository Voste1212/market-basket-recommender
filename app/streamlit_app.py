from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from graph_utils import build_rules_graph
from recommender import RecommendationEngine


st.set_page_config(
    page_title="Market Basket Recommender",
    page_icon="🛒",
    layout="wide",
)


@st.cache_resource
def load_engine() -> RecommendationEngine:
    return RecommendationEngine()


engine = load_engine()

st.title("🛒 Sistem preporuka — Šta još da kupim?")
st.write(
    "Demo aplikacija za preporuke proizvoda u korpi. "
    "Kombinuje association rules, collaborative filtering signal, poslovnu marginu i MMR diversity."
)

with st.sidebar:
    st.header("Podešavanja")
    top_k = st.slider("Broj preporuka", min_value=1, max_value=10, value=5)
    lambda_mmr = st.slider(
        "MMR balans relevance/diversity",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
    )
    st.caption("Veća vrijednost daje prednost najjačem score-u; manja vrijednost većoj raznovrsnosti.")

products = engine.available_products()
cart = st.multiselect(
    "Izaberi proizvode koji su trenutno u korpi:",
    options=products,
    default=["banana", "milk"] if "banana" in products and "milk" in products else products[:2],
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Generiši preporuke", type="primary"):
        recommendations = engine.recommend(cart, top_k=top_k, lambda_mmr=lambda_mmr)
        st.session_state["recommendations"] = recommendations

with col2:
    st.metric("Broj pronađenih pravila", len(engine.rules))

recommendations = st.session_state.get("recommendations", [])

if recommendations:
    st.subheader("Preporučeni proizvodi")
    table = pd.DataFrame(recommendations)[
        [
            "product_name",
            "final_score",
            "support",
            "confidence",
            "lift",
            "margin",
            "collaborative_score",
        ]
    ]
    st.dataframe(table, use_container_width=True)

    st.subheader("Objašnjenja")
    for index, rec in enumerate(recommendations, start=1):
        with st.container(border=True):
            st.markdown(f"### {index}. {rec['product_name'].title()}")
            st.write(engine.explanation(rec))
            st.write(
                f"**Support:** {rec['support']:.3f} | "
                f"**Confidence:** {rec['confidence']:.3f} | "
                f"**Lift:** {rec['lift']:.3f} | "
                f"**Margin:** {rec['margin']:.2f}"
            )
else:
    st.info("Izaberi proizvode i klikni na dugme za generisanje preporuka.")

st.divider()
st.subheader("Interaktivni network graph pravila")

rules_for_graph = engine.rules.head(40).copy()
if not rules_for_graph.empty:
    graph_path = build_rules_graph(rules_for_graph, "data/processed/rules_graph.html")
    with open(graph_path, "r", encoding="utf-8") as file:
        html = file.read()
    components.html(html, height=700, scrolling=True)
else:
    st.warning("Nema dovoljno pravila za prikaz grafa.")

st.divider()
with st.expander("Pregled svih association rules"):
    display_rules = engine.rules.copy()
    if not display_rules.empty:
        display_rules["antecedents"] = display_rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
        display_rules["consequents"] = display_rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
        st.dataframe(
            display_rules[["antecedents", "consequents", "support", "confidence", "lift"]],
            use_container_width=True,
        )
    else:
        st.write("Nema pravila za prikaz.")
