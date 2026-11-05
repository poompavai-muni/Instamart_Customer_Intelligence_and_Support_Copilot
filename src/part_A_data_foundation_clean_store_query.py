# !git clone https://github.com/poompavai-muni/Instamart_Customer_Intelligence_and_Support_Copilot.git

# import os
# List the contents of the cloned directory to verify its structure
# os.listdir('Instamart_Customer_Intelligence_and_Support_Copilot')

# Commented out IPython magic to ensure Python compatibility.
# %cd Instamart_Customer_Intelligence_and_Support_Copilot
# %pip install -r requirements.txt

# Generate the synthetic data once
# 
# !python generate_data.py

# 1. Define path candidates for automatic project folder detection

import os

possible_folders = [
    os.getcwd(),
    "/content/Instamart_Customer_Intelligence_and_Support_Copilot",
    "/content",
]

# 2. Traverse active and parent directories to locate 'generate_data.py'

ROOT = None
for folder in possible_folders:
    if os.path.isfile(os.path.join(folder, "generate_data.py")):
        ROOT = folder
        break

folder = os.getcwd()
while ROOT is None:
    parent = os.path.dirname(folder)
    if parent == folder:
        break
    if os.path.isfile(os.path.join(parent, "generate_data.py")):
        ROOT = parent
        break
    folder = parent

# 3. Enforce directory configuration or throw an error

if ROOT is None:
    raise FileNotFoundError(
        "Project folder not found. In Colab, clone the repository, then run this cell again. "
        "The folder must contain generate_data.py."
    )

os.chdir(ROOT)
print("Project folder:", ROOT)
print("Current working folder:", os.getcwd())

"""# Part A — Data Foundation: Clean, Store, Query

This notebook audits the deterministic synthetic exports, cleans the tables using the assignment's rules, creates `data/clean/*.csv` and `data/instamart.db`, runs seven SQL analyses, and asserts the main reference checkpoints.

**Prerequisites:** Python 3.11+, `python generate_data.py` already run from the repository root, and packages in `requirements.txt`. Do not modify `generate_data.py` or `mock_llm.py`.

## Part A — Data Foundation

**Goal:** inspect the raw files, clean customer and order data, save clean CSV files, build the SQLite database, run the seven required SQL queries, and check the reference numbers.

**Run order:** setup → inspect raw data → audit → clean customers → clean orders → save files/database → SQL queries → summaries → checkpoint validation.
"""

# 1. Initialize directory paths for raw and clean data

import pandas as pd
import numpy as np
import sqlite3
import os

RAW = os.path.join(ROOT, "data", "raw")
CLEAN = os.path.join(ROOT, "data", "clean")
os.makedirs(CLEAN, exist_ok=True)
DB_PATH = os.path.join(ROOT, "data", "instamart.db")

# 2. Validate the existence of all raw datasets

required = ["customers.csv", "products.csv", "orders.csv", "tickets.csv"]
missing = [f for f in required if not os.path.isfile(os.path.join(RAW, f))]
if missing:
    raise FileNotFoundError(f"Missing {missing}. Run `python generate_data.py` from the repository root first.")

# 3. Load raw exports into pandas dataframes

customers_raw = pd.read_csv(os.path.join(RAW, "customers.csv"), dtype=str, keep_default_na=False)
products_raw = pd.read_csv(os.path.join(RAW, "products.csv"))
orders_raw = pd.read_csv(os.path.join(RAW, "orders.csv"), dtype=str, keep_default_na=False)
tickets_raw = pd.read_csv(os.path.join(RAW, "tickets.csv"), dtype=str, keep_default_na=False)

# 4. Print loaded dataset shapes for audit verification

print("Raw shapes:", {"customers": customers_raw.shape, "products": products_raw.shape,
                      "orders": orders_raw.shape, "tickets": tickets_raw.shape})

"""## 1. Audit raw data
Missing values in the raw CSVs include blank strings, so the audit explicitly treats empty strings as missing.
"""

# 1. Helper function to audit column missingness and datatypes

def audit_table(df, name):
    missing_count = df.astype(str).apply(lambda col: col.str.strip().eq("").sum())
    audit = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": missing_count,
        "missing_percent": (missing_count / len(df) * 100).round(2)
    })
    print(f"\n{name}: shape={df.shape}\n")
    print(audit)

# 2. Run audit repots on the raw exports

audit_table(customers_raw, "customers_raw")
audit_table(products_raw, "products_raw")
audit_table(orders_raw, "orders_raw")

# 3. Prepare numeric series for range and domain auditing

order_amount_num = pd.to_numeric(orders_raw["order_amount"], errors="coerce")
delivery_num = pd.to_numeric(orders_raw["delivery_minutes"], errors="coerce")
age_num = pd.to_numeric(customers_raw["age"], errors="coerce")

# 4. Aggregate core pipeline audit checkpoints

audit_checkpoints = {
    "raw_order_rows": len(orders_raw),
    "exact_duplicate_rows": int(orders_raw.duplicated().sum()),
    "pending_rows": int(orders_raw["status"].eq("pending").sum()),
    "distinct_order_ids": int(orders_raw["order_id"].nunique()),
    "non_numeric_order_amount": int(order_amount_num.isna().sum()),
    "negative_delivery_minutes": int(delivery_num.lt(0).sum()),
    "blank_delivery_minutes": int(orders_raw["delivery_minutes"].str.strip().eq("").sum()),
    "blank_age": int(customers_raw["age"].str.strip().eq("").sum()),
    "ages_outside_10_100": int((age_num.notna() & ~age_num.between(10,100)).sum()),
    "distinct_raw_city_spellings": int(customers_raw["city"].nunique()),
    "distinct_nonblank_gender_spellings": int(customers_raw.loc[customers_raw.gender.str.strip().ne(""), "gender"].nunique()),
    "blank_gender": int(customers_raw["gender"].str.strip().eq("").sum()),
}
print("\n",pd.Series(audit_checkpoints))

"""## 2. Clean customers
City is stripped and title-cased; gender is mapped to `male`, `female`, or `unknown`; invalid ages become missing and are filled with the city median using `groupby(...).transform(...)`.
"""

# 1. Initialize customers dataframe and standardize city spellings

customers = customers_raw.copy()
customers["city"] = customers["city"].astype(str).str.strip().str.title()

# 2. Standardize gender labels using deterministic domain mapping

gender_map = {"male":"male", "m":"male", "female":"female", "f":"female"}
customers["gender"] = customers["gender"].astype(str).str.strip().str.lower().map(gender_map).fillna("unknown")

# 3. Parse age and impute invalid values using group medians

customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[~customers["age"].between(10, 100), "age"] = np.nan
customers["age"] = customers["age"].fillna(customers.groupby("city")["age"].transform("median"))
customers["age"] = customers["age"].astype(int)

# 4. Convert signup date and enforce quality assertion checks

customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="raise")
assert customers["city"].nunique() == 6, customers["city"].value_counts()
assert customers["age"].between(18, 60).all()
assert customers["age"].isna().sum() == 0

# 5. Print clean customer metrics for confirmation

print("Clean customer counts:")
print(customers["city"].value_counts())
print(customers["gender"].value_counts())
print("\nAge range:", customers.age.min(), customers.age.max(), "\nMissing ages:", customers.age.isna().sum())

"""## 3. Clean orders
Rules are applied in the order:  

*   parse timestamp
*   remove exact duplicates
*   retain latest timestamp per order ID
*   parse and impute amounts using product price
*   quantity and discount
*   make negative delivery durations missing
*   impute missing completed-order durations by product category
*   leave cancelled delivery times missing.
"""

# 1. Initialize orders and products datasets with datatype parsing

orders = orders_raw.copy()
products = products_raw.copy()
products["price_inr"] = pd.to_numeric(products["price_inr"], errors="raise")
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="raise")

# 2. De-duplicate orders keeping the latest record per order_id

orders = orders.drop_duplicates().copy()
orders = orders.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last").copy()

# 3. Convert metrics to numeric and impute null order amounts

for col in ["quantity", "discount_pct", "order_amount", "delivery_minutes"]:
    orders[col] = pd.to_numeric(orders[col], errors="coerce")

orders = orders.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left", validate="many_to_one")
orders["order_amount"] = orders["order_amount"].fillna(
    (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
).round(2)
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

# 4. Impute completed delivery time via category group medians

completed_mask = orders["status"].eq("completed")
category_medians = orders.loc[completed_mask].groupby("category")["delivery_minutes"].median()
missing_completed = completed_mask & orders["delivery_minutes"].isna()
orders.loc[missing_completed, "delivery_minutes"] = orders.loc[missing_completed, "category"].map(category_medians)
orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"] = np.nan

# 5. Convert columns and run data quality assertion checks

orders["quantity"] = orders["quantity"].astype(int)
orders["discount_pct"] = orders["discount_pct"].astype(int)
orders = orders.drop(columns=["price_inr"])
assert orders["order_id"].is_unique
assert not orders["status"].eq("pending").any()
assert orders["order_amount"].notna().all()
assert orders.loc[orders.status.eq("completed"), "delivery_minutes"].notna().all()
assert orders.loc[orders.status.eq("cancelled"), "delivery_minutes"].isna().all()

# 6. Print summary metrics and validation reports for audit

print("Clean orders:", orders.shape)
print("\n", orders["status"].value_counts())
print("\nCompleted revenue INR:", round(orders.loc[orders.status.eq("completed"), "order_amount"].sum(), 2))
print("\nMissing delivery values by status:")
print(orders.groupby("status")["delivery_minutes"].apply(lambda s: s.isna().sum()))

"""## 4. Save clean CSVs and SQLite database"""

# 1. Create backup copies and format signup dates and timestamps

customers_out = customers.copy()
customers_out["signup_date"] = customers_out["signup_date"].dt.strftime("%Y-%m-%d")
orders_out = orders.copy()
orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

# 2. Write clean datasets as portable independent CSV files

customers_out.to_csv(os.path.join(CLEAN, "customers.csv"), index=False)
products.to_csv(os.path.join(CLEAN, "products.csv"), index=False)
orders_out.to_csv(os.path.join(CLEAN, "orders.csv"), index=False)

# 3. Export clean tables to the SQLite relational database

with sqlite3.connect(DB_PATH) as con:
    customers_out.to_sql("customers", con, if_exists="replace", index=False)
    products.to_sql("products", con, if_exists="replace", index=False)
    orders_out.to_sql("orders", con, if_exists="replace", index=False)

# 4. Verify database schema integrity and export directories

with sqlite3.connect(DB_PATH) as con:
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", con)
print(tables)
print("Saved:", CLEAN, "\n    ", DB_PATH)

"""## 5. SQL analysis — seven required queries
Each query is saved in `sql/queries.sql` and executed against the generated SQLite database.
"""

# 1. Initialize SQL query export directory and configuration

sql_dir = os.path.join(ROOT, "sql")
os.makedirs(sql_dir, exist_ok=True)

# 2. Define the seven required analytical SQL queries

queries = {
"Q1_where_having": """SELECT customer_id, SUM(order_amount) AS high_value_spend
FROM orders WHERE status='completed' AND order_amount > 500
GROUP BY customer_id HAVING SUM(order_amount) > 8000
ORDER BY high_value_spend DESC""",

"Q2_city_category_revenue": """SELECT c.city, p.category, ROUND(SUM(o.order_amount),3) AS revenue
FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id)
WHERE o.status='completed' GROUP BY c.city, p.category ORDER BY c.city, p.category""",

"Q3_products_above_average": """SELECT product_id, product_name, price_inr FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products) ORDER BY price_inr DESC""",

"Q4_customers_no_completed_order": """SELECT c.customer_id FROM customers c
LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed'
WHERE o.order_id IS NULL ORDER BY c.customer_id""",

"Q5_spend_above_average": """WITH customer_spend AS (
 SELECT customer_id, SUM(order_amount) AS total_spend FROM orders
 WHERE status='completed' GROUP BY customer_id
), average_spend AS (SELECT AVG(total_spend) AS avg_spend FROM customer_spend)
SELECT customer_id, total_spend, avg_spend FROM customer_spend CROSS JOIN average_spend
WHERE total_spend > avg_spend ORDER BY total_spend DESC""",

"Q6_city_spend_ranks": """WITH customer_spend AS (
 SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
 FROM customers c JOIN orders o USING(customer_id)
 WHERE o.status='completed' GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, total_spend,
 RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
 DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,
 ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC, customer_id) AS row_num
FROM customer_spend ORDER BY city, spend_rank, customer_id""",

"Q7_daily_running_revenue": """WITH daily AS (
 SELECT substr(order_ts,1,10) AS day, SUM(order_amount) AS revenue
 FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)
)
SELECT day, ROUND(revenue,3) AS revenue,
 ROUND(SUM(revenue) OVER(ORDER BY day),3) AS cumulative_revenue
FROM daily ORDER BY day"""
}

# 3. Write the complete SQL script file to file system

with open(os.path.join(sql_dir, "queries.sql"), "w", encoding="utf-8") as f:
    for name, query in queries.items():
        f.write(f"-- {name}\n{query};\n\n")

# 4. Execute SQL queries and validate record dimensions

sql_results = {}
with sqlite3.connect(DB_PATH) as con:
    for name, query in queries.items():
        result = pd.read_sql_query(query, con)
        sql_results[name] = result
        print(name, result.shape)
        print(result.head(5),"\n")

"""## 6. pandas cross-check and amount distribution"""

# 1. Initialize merged analytical dataset and prevent column duplication

completed = orders.loc[orders.status.eq("completed")].copy()

# Drop city and category if they already exist in orders to prevent suffix duplication during merge

if "city" in completed.columns:
    completed = completed.drop(columns=["city"])
if "category" in completed.columns:
    completed = completed.drop(columns=["category"])

completed = completed.merge(customers[["customer_id","city"]], on="customer_id", how="left", validate="many_to_one")
completed = completed.merge(products[["product_id","category"]], on="product_id", how="left", validate="many_to_one")

# 2. Generate revenue pivot table by city and product category

pivot = pd.pivot_table(completed, index="city", columns="category", values="order_amount",
                       aggfunc="sum", fill_value=0, margins=True)
print(pivot)
print("\nPivot grand total INR:", float(pivot.loc["All","All"]))

# 3. Aggregate core revenue metrics by city

city_summary = completed.groupby("city").agg(
    total_revenue=("order_amount","sum"),
    mean_order_amount=("order_amount","mean"),
    completed_order_count=("order_amount","count")
).round(3)
print("\n",city_summary)

# 4. Conduct order amount IQR outlier detection analysis

amount = completed["order_amount"]
q1, q3 = amount.quantile([0.25,0.75])
iqr = q3-q1
summary = {
    "mean": amount.mean(), "median": amount.median(), "skewness": amount.skew(),
    "Q1": q1, "Q3": q3, "IQR": iqr,
    "lower_fence": q1-1.5*iqr, "upper_fence": q3+1.5*iqr,
    "outliers_below": int((amount < q1-1.5*iqr).sum()),
    "outliers_above": int((amount > q3+1.5*iqr).sum())
}
print("\n",pd.Series(summary).round(3))

# 5. Generate correlation matrix and locate highest associated pair

corr_cols = ["order_amount","discount_pct","quantity","delivery_minutes"]
corr = completed[corr_cols].corr(numeric_only=True)
print("\nCorrelation:",corr.round(3))
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
print("\nLargest absolute correlation pair:", pairs.abs().idxmax(), "correlation:", round(float(pairs.loc[pairs.abs().idxmax()]), 3))

"""## 7. Checkpoints and validation
These assertions are intentionally explicit so a clean-clone evaluator can detect pipeline regressions.
"""

import pandas as pd

# 1. Calculate actual metrics from pipeline outputs

actual_metrics = {
    # Verify dimensions of cleaned customer, product, and order datasets
    "customers_rows": len(customers_out),
    "products_rows": len(products),
    "orders_rows": len(orders_out),

    # Enforce database integrity and status domain constraints
    "orders_unique": orders_out["order_id"].is_unique,
    "no_pending": not orders_out["status"].eq("pending").any(),
    "no_missing_amount": orders_out["order_amount"].notna().all(),

    # Ensure delivery durations conform to status (e.g. no missing for completed)
    "completed_delivery_complete": orders_out.loc[orders_out.status.eq("completed"), "delivery_minutes"].notna().all(),
    "cancelled_delivery_missing": orders_out.loc[orders_out.status.eq("cancelled"), "delivery_minutes"].isna().all(),

    # Validate expected categories, cities, genders, and revenues
    "six_cities": customers_out["city"].nunique(),
    "gender_domain": set(customers_out["gender"].unique()),
    "completed_revenue": round(orders_out.loc[orders_out.status.eq("completed"), "order_amount"].sum(), 2),

    # Verify results of our analytical SQL queries against expected outputs
    "Q1_count": len(sql_results["Q1_where_having"]),
    "Q2_rows": len(sql_results["Q2_city_category_revenue"]),
    "Q3_count": len(sql_results["Q3_products_above_average"]),
    "Q4_count": len(sql_results["Q4_customers_no_completed_order"]),
    "Q5_count": len(sql_results["Q5_spend_above_average"]),
    "Q7_days": len(sql_results["Q7_daily_running_revenue"]),
}

# 2. Define target boolean logic checks

checks = {
    "customers_rows": actual_metrics["customers_rows"] == 1200,
    "products_rows": actual_metrics["products_rows"] == 40,
    "orders_rows": actual_metrics["orders_rows"] == 12309,
    "orders_unique": actual_metrics["orders_unique"],
    "no_pending": actual_metrics["no_pending"],
    "no_missing_amount": actual_metrics["no_missing_amount"],
    "completed_delivery_complete": actual_metrics["completed_delivery_complete"],
    "cancelled_delivery_missing": actual_metrics["cancelled_delivery_missing"],
    "six_cities": actual_metrics["six_cities"] == 6,
    "gender_domain": actual_metrics["gender_domain"] <= {"male","female","unknown"},
    "completed_revenue": actual_metrics["completed_revenue"] == 2998668.80,
    "Q1_count": actual_metrics["Q1_count"] == 30,
    "Q2_rows": actual_metrics["Q2_rows"] == 30,
    "Q3_count": actual_metrics["Q3_count"] == 11,
    "Q4_count": actual_metrics["Q4_count"] == 25,
    "Q5_count": actual_metrics["Q5_count"] == 550,
    "Q7_days": actual_metrics["Q7_days"] == 183,
}

# 3. Build and display summary comparison dataframe

df_checks = pd.DataFrame({
    "Passed": pd.Series(checks),
    "Actual Value": pd.Series(actual_metrics)
})
print(df_checks)

# 4. Assertion enforcement

failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise AssertionError(f"Part A checkpoint failures: {failed}")
print("\nALL CHECKPOINTS PASSED")