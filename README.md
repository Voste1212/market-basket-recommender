# Market Basket Recommender — “Šta još da kupim?”

## Live demo

Aplikacija je dostupna na:
https://market-basket-recommender-dhp6f3yrzdhja23cvcxjtp.streamlit.app

Projekat implementira sistem preporuka za online prodavnicu. Korisnik unosi proizvode koji su trenutno u korpi, a aplikacija vraća dodatne proizvode koje bi vjerovatno trebalo kupiti zajedno sa njima.

Rješenje kombinuje:

- **FP-Growth / association rules** za pravila tipa: `Ako kupac kupi X i Y, često kupi i Z`
- **Collaborative filtering pristup** preko ko-pojavljivanja proizvoda u korpama, kao praktična lokalna zamjena za ALS demo
- **Business margin scoring** za davanje prednosti profitabilnijim proizvodima
- **MMR diversity algoritam** za raznovrsnije preporuke
- **Network graph** za vizuelni pregled pravila: support, confidence i lift

> Napomena: projekat radi odmah sa sample podacima. Za veći rezultat zamijeniti fajlove u `data/sample/` pravim Instacart fajlovima.

---

## 1. Pokretanje lokalno

```bash
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
# .venv\Scripts\activate      # Windows

pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Aplikacija će se otvoriti u browseru.

---

## 2. Struktura projekta

```text
market-basket-recommender/
│
├── app/
│   ├── streamlit_app.py       # Streamlit aplikacija
│   ├── recommender.py         # Glavna logika preporuka
│   ├── mmr.py                 # MMR diversity algoritam
│   └── graph_utils.py         # Network graph pravila
│
├── data/
│   ├── sample/                # Mali demo dataset
│   ├── raw/                   # Ovdje idu originalni Instacart CSV fajlovi
│   └── processed/             # Obrađeni podaci
│
├── models/                    # Modeli / pravila / margine
├── notebooks/                 # Notebook fajlovi za objašnjenje rada
├── reports/                   # Prezentacija i izvještaji
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 3. Demo podaci

Sample dataset sadrži male korpe izmišljene online prodavnice. Fajlovi su:

- `data/sample/orders_sample.csv`
- `data/sample/products_sample.csv`
- `data/sample/product_margins_sample.csv`

Format `orders_sample.csv`:

```csv
order_id,product_name
1,banana
1,milk
1,cereal
2,banana
2,yogurt
```

Jedan `order_id` predstavlja jednu korpu.

---

## 4. Kako radi algoritam

### 4.1. FP-Growth / association rules

Iz korpi se formira transakciona matrica. Zatim se pronalaze česti skupovi proizvoda i pravila:

```text
antecedents -> consequents
```

Primjer:

```text
{banana, milk} -> {cereal}
```

Za svako pravilo računaju se:

- **support** — koliko često se kombinacija javlja u svim korpama
- **confidence** — vjerovatnoća da se kupi consequent ako je antecedent već kupljen
- **lift** — koliko je pravilo jače od slučajne kupovine

### 4.2. Business margin

Za svaki proizvod postoji procijenjena poslovna margina. Finalni score ne preporučuje samo ono što je statistički često, već i ono što ima poslovnu vrijednost.

### 4.3. Collaborative filtering

U ovoj verziji koristi se item-item ko-pojavljivanje proizvoda iz istorijskih korpi. To omogućava dodatne preporuke i onda kada nema savršenog association rule poklapanja.

U produkcionoj varijanti ovaj dio se može zamijeniti ALS modelom iz biblioteke `implicit` ili Spark MLlib ALS modelom.

### 4.4. MMR diversity

MMR smanjuje problem da korisnik dobije 5 skoro istih preporuka. Bira proizvode koji imaju dobar score, ali nijesu previše slični već izabranim preporukama.

---

## 5. Scoring preporuka

Finalni score se računa kombinacijom:

```text
final_score = 0.35 * confidence_score
            + 0.25 * lift_score
            + 0.20 * margin_score
            + 0.20 * collaborative_score
```

Zatim se primjenjuje MMR filtriranje.

---

## 6. Korišćenje sa Instacart datasetom

Za Instacart dataset očekivani fajlovi su:

```text
orders.csv
order_products__prior.csv
order_products__train.csv
products.csv
aisles.csv
departments.csv
```

Minimalna priprema:

1. Spojiti `order_products__prior.csv` sa `products.csv` preko `product_id`.
2. Dobiti tabelu oblika:

```csv
order_id,product_name
1,Banana
1,Organic Whole Milk
1,Organic Strawberries
```

3. Sačuvati fajl kao:

```text
data/sample/orders_sample.csv
```

ili promijeniti putanju u `app/recommender.py`.

---

## 7. Deploy

Najbrži deploy je preko Streamlit Community Cloud:

1. Napraviti GitHub repozitorijum.
2. Uploadovati sve fajlove iz ovog projekta.
3. Na Streamlit Cloud izabrati repo.
4. Kao main file navesti:

```text
app/streamlit_app.py
```

---

## 8. Šta prikazati na odbrani

1. Izabrati par proizvoda u korpi.
2. Kliknuti na “Generiši preporuke”.
3. Objasniti zašto je sistem preporučio konkretan proizvod.
4. Pokazati support, confidence i lift.
5. Pokazati network graph.
6. Objasniti zašto se koristi margina i MMR.

---

## 9. Ograničenja

- Sample dataset je mali, pa rezultati služe za demonstraciju.
- Za ozbiljne rezultate potreban je veliki dataset poput Instacart Market Basket Analysis.
- Collaborative filtering dio je pojednostavljena item-item verzija. ALS je opisan kao naredno proširenje.

---

## 10. Zaključak

Rješenje pokazuje kako online prodavnica može automatski predlagati dodatne proizvode u korpi. Kombinacija association rules, collaborative filtering signala, poslovne margine i diversity algoritma daje preporuke koje imaju i statistički i poslovni smisao.
