# Notebooks

Predloženi redosljed rada:

1. `01_data_preparation.ipynb` — priprema Instacart podataka u format `order_id, product_name`.
2. `02_fp_growth_rules.ipynb` — treniranje FP-Growth pravila.
3. `03_collaborative_filtering.ipynb` — item-item ili ALS collaborative filtering.
4. `04_evaluation.ipynb` — analiza support/confidence/lift i ručna validacija preporuka.

U ovoj verziji glavna logika je implementirana direktno u `app/recommender.py`, da bi aplikacija radila odmah bez dodatnog pokretanja notebook-a.
