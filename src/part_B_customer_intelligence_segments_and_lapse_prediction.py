# Clone the repository and open the project
# !git clone https://github.com/poompavai-muni/Instamart_Customer_Intelligence_and_Support_Copilot.git

# import os
# List the contents of the cloned directory to verify its structure
# os.listdir('Instamart_Customer_Intelligence_and_Support_Copilot')

# Find the project folder.
possible_folders = [
    os.getcwd(),
    "/content/Instamart_Customer_Intelligence_and_Support_Copilot",
    "/content",
]

ROOT = None
for folder in possible_folders:
    if os.path.isfile(os.path.join(folder, "generate_data.py")):
        ROOT = folder
        break

# If the notebook was launched from a subfolder, check its parent folders.
folder = os.getcwd()
while ROOT is None:
    parent = os.path.dirname(folder)
    if parent == folder:
        break
    if os.path.isfile(os.path.join(parent, "generate_data.py")):
        ROOT = parent
        break
    folder = parent

if ROOT is None:
    raise FileNotFoundError(
        "Project folder not found. In Colab, clone the repository, then run this cell again. "
        "The folder must contain generate_data.py."
    )

os.chdir(ROOT)
print("Project folder:", ROOT)
print("Current working folder:", os.getcwd())

# Commented out IPython magic to ensure Python compatibility.
# %pip install -r requirements.txt

"""# Part B — Customer Intelligence: Segments and Lapse Prediction

**Goal:** build customer-level RFM features, segment customers with K-means, use PCA to inspect the segments, and train a leakage-aware lapse prediction model.

**Before running:** Part A must have completed successfully. Run cells in order and read each printed metric before continuing.
"""

# Part B Setup: Imports, Constants and Data Loading
import os
import pandas as pd, numpy as np, joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

CLEAN = os.path.join(ROOT, "data", "clean")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
customers = pd.read_csv(os.path.join(CLEAN,"customers.csv"), parse_dates=["signup_date"])
orders = pd.read_csv(os.path.join(CLEAN,"orders.csv"), parse_dates=["order_ts"])
orders["status"] = orders["status"].astype(str)
completed = orders[orders.status.eq("completed")].copy()

"""## 1. Build the RFM table as of 2025-10-01"""

# Build the RFM table as of 2025-10-01
snapshot = pd.Timestamp("2025-10-01 00:00:00")
rfm = completed.groupby("customer_id").agg(
    last_order_ts=("order_ts","max"),
    frequency=("order_id","nunique"),
    monetary=("order_amount","sum")
).reset_index()
rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
rfm = rfm[["customer_id","recency","frequency","monetary"]]
print("RFM customers:", len(rfm))
print(rfm[["recency","frequency","monetary"]].describe().round(2))

"""## 2. Compare K-means solutions for K=2…8"""

# Compare K-means solutions for K=2…8 and fit the final model
rfm_features = ["recency","frequency","monetary"]
scaler_rfm = StandardScaler()
X_rfm = scaler_rfm.fit_transform(rfm[rfm_features])
cluster_rows = []
for k in range(2,9):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X_rfm)
    cluster_rows.append({"K":k, "inertia":round((model.inertia_),1), "silhouette":round((silhouette_score(X_rfm,labels)),3)})
cluster_scores = pd.DataFrame(cluster_rows)
print(cluster_scores)
# Select K=4 as a clear, interpretable starting point consistent with the brief's reference profile.
FINAL_K = 4
kmeans = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42)
rfm["segment_id"] = kmeans.fit_predict(X_rfm)

import pandas as pd

cluster_table = pd.DataFrame(cluster_rows)
print(cluster_table.columns.tolist())   # check the column names

drop_table = cluster_scores[["K", "inertia"]].copy()

# How much the inertia fell compared with the previous K
drop_table["drop_from_previous"] = (-drop_table["inertia"].diff()).round(1)
drop_table["drop_%"] = (drop_table["drop_from_previous"] / drop_table["inertia"].shift(1) * 100).round(1)

# The first row (K=2) has no previous K, so show it as a dash
print(drop_table.fillna("-").to_string(index=False))

# Which K gives the biggest single percentage drop?
best = drop_table.loc[drop_table["drop_%"].idxmax()]
print()
print(f"Biggest % drop: K={int(best['K']) - 1} -> K={int(best['K'])}, "
      f"{best['drop_from_previous']} inertia units ({best['drop_%']}%)")

"""**Why the inertia of K=8 is always lower than that of K=2 yet that fact alone cannot pick K?**

The inertia of K=8 is lower than K=2 because increasing the number of clusters allows data points to be grouped into smaller, more compact clusters, reducing the within-cluster sum of squared distances. However, increasing K generally reduces inertia, even when the additional clusters do not provide meaningful improvement. Therefore, we cannot select K based on inertia alone; we use the Elbow Method to identify the point where the reduction in inertia starts slowing down significantly.

K=4 was chosen because, the elbow is sharpest at K=4, where inertia falls 41.4% from K=3. Its silhouette of 0.526 is the highest of all K from 2 to 8.

## 3. Profile and name segments; project with PCA
"""

# 3. Profile and name segments, apply PCA, and save the segment assignments

# Calculate segment summary statistics (mean recency, frequency, and monetary)
profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id","size"),
    mean_recency=("recency","mean"),
    mean_frequency=("frequency","mean"),
    mean_monetary=("monetary","mean")
).round(2)
print("Reference for k=4:\n",profile)

# Names are assigned by business behaviour, not by assuming cluster IDs have an order.
ordered = profile.sort_values(["mean_monetary","mean_frequency"], ascending=False)
names = {}
for rank, sid in enumerate(ordered.index):
    names[sid] = ["Premium High-Value","Frequent Essentials","Developing Customers","At-Risk / Occasional"][min(rank,3)]

# Mark the premium segment and assign labels back to the rfm table
premium_segment = profile["mean_monetary"].idxmax()
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"].eq(premium_segment)

# Perform 2D PCA projection of RFM features for visual diagnostic check
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(X_rfm)
print("\nPCA explained variance ratio:", pca.explained_variance_ratio_.round(3),
      "\ntotal:", round(float(pca.explained_variance_ratio_.sum()),3))

# Save the final segmented customer list to a CSV file
segments_path = os.path.join(OUT, "customer_segments.csv")
rfm[["customer_id","recency","frequency","monetary","segment_id","segment_name","is_premium"]].to_csv(segments_path,index=False)
print("\nSaved", segments_path)

"""## 4. Define a leakage-safe lapse cohort
The cutoff is 2025-09-01. Features use only orders strictly before the cutoff. The label checks whether the customer has a completed order during September. Segment labels are deliberately excluded from model features.
"""

# 4. Define a leakage-safe lapse cohort with feature engineering and baseline evaluation
cutoff = pd.Timestamp("2025-09-01")
window_end = pd.Timestamp("2025-10-01")

# Filter orders BEFORE the cutoff only (used for every feature to prevent target leakage)
before = orders[orders.order_ts < cutoff].copy()
before_completed = before[before.status.eq("completed")].copy()

# Define cohort = customers with at least one completed order before the cutoff
cohort_ids = before_completed["customer_id"].unique()

# Identify customers who completed an order in September 2025 (the label window)
completed_orders = orders[orders.status.eq("completed")]
september_ids = set(completed_orders.loc[
    (completed_orders.order_ts >= cutoff) & (completed_orders.order_ts < window_end), "customer_id"])

# Create cohort DataFrame and define the 'lapsed' target variable
cohort = pd.DataFrame({"customer_id": cohort_ids})
cohort["lapsed"] = (~cohort.customer_id.isin(september_ids)).astype(int)

# Aggregate order features: recency, frequency, monetary, average discount, and delivery time
feat = before_completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_disc_pct=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean"),
).reset_index()
feat["recency"] = (cutoff - feat["last_order_ts"]).dt.days

# Calculate customer-level cancellation rates
final_status = before[before.status.isin(["completed", "cancelled"])]
cancel = (final_status.assign(cancelled=final_status.status.eq("cancelled").astype(int))
          .groupby("customer_id")["cancelled"].mean().rename("cancellation_rate"))

# Map favourite product category based on historical purchases
fav = (before_completed.drop(columns=["category"], errors="ignore")
       .merge(pd.read_csv(os.path.join(CLEAN, "products.csv"))[["product_id", "category"]], on="product_id"))
fav = fav.groupby("customer_id")["category"].agg(lambda x: x.mode().iloc[0]).rename("favourite_category")

# Merge computed features and demographical information together
feat = feat.merge(cancel, on="customer_id", how="left").merge(fav, on="customer_id", how="left")
feat = feat.merge(customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id", how="left")
feat["tenure_days"] = (cutoff - feat["signup_date"]).dt.days

# Build the final model dataset by combining target labels and feature columns
model_df = cohort.merge(feat, on="customer_id", how="left")
model_df = model_df.drop(columns=["last_order_ts", "signup_date"])

# Display target distribution and dataset shape metrics
n = len(model_df)
n_lapsed = int(model_df.lapsed.sum())
n_active = n - n_lapsed
print("Cohort:", n)
print("Lapsed (1):", n_lapsed, "->", round(n_lapsed / n * 100, 2), "%")
print("Not lapsed (0):", n_active, "->", round(n_active / n * 100, 2), "%")
print(model_df.head())

# Perform stratified train-test split to ensure class distribution consistency
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

y = model_df["lapsed"]
X = model_df.drop(columns=["customer_id", "lapsed"])   # customer_id is a label, not a feature

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print("train rows:", len(X_train), "| test rows:", len(X_test))

# Evaluate baseline model predicting the majority "not lapsed" class for everyone
baseline = np.zeros(len(y_test), dtype=int)
print("baseline accuracy:", round(accuracy_score(y_test, baseline) * 100, 1), "%")
print("baseline lapsed-class F1:", f1_score(y_test, baseline, zero_division=0))

"""## 5. Split, preprocess without leakage"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

y = model_df["lapsed"].astype(int)
X = model_df.drop(columns=["customer_id", "lapsed"])
categorical = ["city", "gender", "favourite_category"]
numeric = [c for c in X.columns if c not in categorical]

# 1. confirm no feature column contains a missing value
print("missing values in each feature column:")
print(X.isna().sum())
assert X.isna().sum().sum() == 0, "a feature column has missing values"
print("No feature column contains a missing value")

# 2. split first, so everything below is learned from the training rows only
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 3. standardise numeric columns, one-hot encode the nominal ones with k-1 columns
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical),
])
X_train_prep = preprocessor.fit_transform(X_train)   # learns from TRAIN only
X_test_prep = preprocessor.transform(X_test)         # only applies what was learned

print("\ntrain matrix:", X_train_prep.shape, "| test matrix:", X_test_prep.shape)
encoder = preprocessor.named_transformers_["cat"]
for col, cats in zip(categorical, encoder.categories_):
    print(f"{col}: {len(cats)} categories -> {len(cats) - 1} columns")

# 4. learned mean of frequency vs the full-cohort mean
scaler = preprocessor.named_transformers_["num"]
learned = scaler.mean_[numeric.index("frequency")]
print("\nscaler mean of frequency (training rows only):", round(learned, 4))
print("full-cohort mean of frequency                 :", round(model_df["frequency"].mean(), 4))

"""The scaler’s mean (8.1147) differs from the full-cohort mean (8.0285) because the scaler is fitted only on the 924 training rows, a random 80% of the cohort, while the full-cohort mean also includes the 232 test rows that must stay unseen.

## 6. Train, Tune and Evaluate
"""

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# (a) baseline: always predict the majority class (not lapsed)
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
base_pred = baseline.predict(X_test)
print("(a) Baseline accuracy:", round(accuracy_score(y_test, base_pred), 3),
      "| lapsed-class F1:", round(f1_score(y_test, base_pred, zero_division=0), 3))

# (b) logistic regression: 5 values of C, 5-fold cross-validation, scoring F1
log_pipe = Pipeline([("prep", preprocessor), ("model", LogisticRegression(max_iter=1000))])
log_grid = GridSearchCV(log_pipe, {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
log_grid.fit(X_train, y_train)
print("(b) Logistic best params:", log_grid.best_params_, "| best CV F1:", round(log_grid.best_score_, 2))

# (c) random forest: 3 x 3 grid = 9 combinations x 5 folds = 45 fits
rf_pipe = Pipeline([("prep", preprocessor),
                    ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_grid = GridSearchCV(rf_pipe, {"model__max_depth": [3, 5, 8],
                                 "model__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1")
rf_grid.fit(X_train, y_train)
print("(c) Random forest best params:", rf_grid.best_params_, "| best CV F1:", round(rf_grid.best_score_, 2))
print("    random forest fits:", len(rf_grid.cv_results_["params"]) * 5)

# (d) tuning is finished -> only NOW is the test set used, once, for both models
def evaluate(name, estimator):
    pred = estimator.predict(X_test)
    print("\n", name)
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    print("Lapsed class -> Precision:", round(precision, 3), "| Recall:", round(recall, 3), "| F1:", round(f1, 3))
    return f1

log_f1 = evaluate("Tuned Logistic Regression", log_grid.best_estimator_)
rf_f1 = evaluate("Tuned Random Forest", rf_grid.best_estimator_)

if rf_f1 >= log_f1:
    selected_name, selected = "Random Forest", rf_grid.best_estimator_
else:
    selected_name, selected = "Logistic Regression", log_grid.best_estimator_
selected_f1 = max(rf_f1, log_f1)
print("\nSelected model:", selected_name, "| test F1:", round(selected_f1, 3))
assert selected_f1 >= 0.55, "selected model must reach test F1 >= 0.55"
print("Selected model meets the test F1 >= 0.55 requirement")

"""I selected the random forest. Missing a lapsing customer means losing that customer’s future revenue with no chance to win them back, while wrongly targeting a loyal customer only costs a small discount or message, so recall on the lapsed class matters more than precision. The forest found 60 of the 89 lapsing customers in the test set (recall 0.674) against 58 for logistic regression (0.652), with almost the same number of loyal customers wrongly flagged (33 vs 32). It also had the higher cross-validated F1 (0.69 vs 0.65) and the higher test F1 (0.659 vs 0.648), so the cross-validation and test results agree. The gap between the two models is small, only two customers on the test set, so the cost argument supports the choice more than the F1 difference does.

## 7. Class-balance experiment
"""

from sklearn.base import clone

# 1. put the TRAINING rows and their labels together
train_df = X_train.copy()
train_df["lapsed"] = y_train

lapsed_rows = train_df[train_df["lapsed"] == 1]
loyal_rows = train_df[train_df["lapsed"] == 0]
print("training counts before -> not lapsed:", len(loyal_rows), "| lapsed:", len(lapsed_rows))

# 2. draw random duplicates of lapsed rows (with replacement) until the classes are equal
n_extra = len(loyal_rows) - len(lapsed_rows)
extra_rows = lapsed_rows.sample(n=n_extra, replace=True, random_state=42)

# 3. keep EVERY original training row and add the duplicates
balanced = pd.concat([train_df, extra_rows])
X_train_bal = balanced.drop(columns="lapsed")
y_train_bal = balanced["lapsed"]
print("training counts after  -> not lapsed:", int((y_train_bal == 0).sum()),
      "| lapsed:", int((y_train_bal == 1).sum()), "| rows:", len(balanced))

# 4. refit the best logistic regression (same C) on the balanced training data
log_balanced = clone(log_grid.best_estimator_)
log_balanced.fit(X_train_bal, y_train_bal)

# 5. compare both models on the ORIGINAL, unchanged test set
def lapsed_metrics(estimator):
    pred = estimator.predict(X_test)
    return {"precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0)}

before = lapsed_metrics(log_grid.best_estimator_)   # trained on the original rows
after = lapsed_metrics(log_balanced)                # trained on the balanced rows

comparison = pd.DataFrame({"un-balanced": before, "over-sampled": after}).round(2)
comparison["change"] = (comparison["over-sampled"] - comparison["un-balanced"]).round(3)
print()
print(comparison)
print("test rows used:", len(X_test), "(unchanged)")

biggest = comparison["change"].abs().idxmax()
print(f"\nMetric that moved most: {biggest} ({comparison.loc[biggest, 'change']:+.3f})")

print("\nConfusion matrix, un-balanced  [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, log_grid.best_estimator_.predict(X_test), labels=[0, 1]))
print("Confusion matrix, over-sampled [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, log_balanced.predict(X_test), labels=[0, 1]))

"""* Recall moved most, rising from 0.65 to 0.85 (+0.20), while precision fell slightly from 0.64 to 0.59 and F1 improved from 0.65 to 0.70. For the retention team this means the over-sampled model catches 76 of the 89 customers who actually lapsed instead of 58, so 18 more at-risk customers get an offer. The price is 21 extra loyal customers contacted unnecessarily (53 false alarms instead of 32). That is a good trade when a retention offer costs much less than losing a customer.

* Random duplication adds no new
customers, only more weight on the existing lapsed ones. That is why precision doesn’t improve.

## 8. Interpret, Persist and Use
"""

import joblib

# (a) Most important features
def clean_names(names):
    # "num__frequency" -> "frequency", "cat__city_Delhi" -> "city_Delhi"
    return [n.split("__", 1)[1] for n in names]

# five most important features of the best random forest
rf_best = rf_grid.best_estimator_
rf_names = clean_names(rf_best.named_steps["prep"].get_feature_names_out())
rf_importance = pd.Series(rf_best.named_steps["model"].feature_importances_, index=rf_names)
print("Top 5 random forest features:")
print(rf_importance.sort_values(ascending=False).head(5).round(2))

# coefficient signs in the best logistic regression (features are standardised)
log_best = log_grid.best_estimator_
log_names = clean_names(log_best.named_steps["prep"].get_feature_names_out())
log_coef = pd.Series(log_best.named_steps["model"].coef_[0], index=log_names)
print("\nLogistic regression coefficients for three features:")
for feature in ["recency", "frequency", "cancellation_rate"]:
    value = log_coef[feature]
    print(f"{feature}: {value:+.2f} ({'raises' if value > 0 else 'lowers'} the chance of lapsing)")

# (b) Save, Reload, Compare
os.makedirs(OUT, exist_ok=True)
model_path = os.path.join(OUT, "lapse_model.joblib")
joblib.dump(selected, model_path)
reloaded = joblib.load(model_path)
same = np.array_equal(selected.predict(X_test), reloaded.predict(X_test))
print("\nSaved:", model_path)
print("Reloaded model reproduces the test predictions exactly:", same)
assert same

# (c) Retention list for Premium customers in the test set
test_ids = model_df.loc[X_test.index, "customer_id"].reset_index(drop=True)
retention = pd.DataFrame({
    "customer_id": test_ids,
    "lapse_probability": selected.predict_proba(X_test)[:, 1].round(2),
    "actually_lapsed": y_test.reset_index(drop=True).values,
})
retention = retention.merge(rfm[["customer_id", "is_premium"]], on="customer_id", how="left")
retention["is_premium"] = retention["is_premium"].fillna(False).astype(bool)

premium_test = retention[retention["is_premium"]]
print("\nPremium customers in the test set:", len(premium_test))
top = premium_test.sort_values("lapse_probability", ascending=False).head(10)
top = top[["customer_id", "lapse_probability", "actually_lapsed"]]
top.to_csv(os.path.join(OUT, "retention_list.csv"), index=False)
print(top.to_string(index=False))
print("Of these", len(top), "customers,", int(top["actually_lapsed"].sum()), "truly lapsed")
print("LIMITATION: the Premium flag comes from the whole-window RFM, which includes September, "
      "so this retention list is illustrative and not a leakage-free evaluation.")

# Acceptance check for Part B outputs
seg = pd.read_csv(os.path.join(OUT, "customer_segments.csv"))
print("\ncustomer_segments.csv columns:", seg.columns.tolist())
mean_monetary = seg.groupby("segment_id")["monetary"].mean()
premium_ids = seg.loc[seg["is_premium"], "segment_id"].unique()
print("Premium segment:", list(premium_ids), "| highest-mean-monetary segment:", mean_monetary.idxmax())
assert list(premium_ids) == [mean_monetary.idxmax()]
print("retention_list.csv columns:", pd.read_csv(os.path.join(OUT, "retention_list.csv")).columns.tolist())

"""## One-line business readings

**Random forest features:**

* **Frequency:** customers who order often have a habit, so frequency is the strongest sign they will stay.
* **Monetary:** large total spend goes with commitment (it overlaps with frequency).
* **Recency:** a longer gap since the last order signals a fading customer.
* **Cancellation_rate:** a higher share of cancelled orders points to service friction.
* **Avg_delivery:** slower average deliveries suggest dissatisfaction.

**Logistic coefficients (features are standardised):**

* **Frequency (-1.70):** each extra standard deviation of frequency strongly lowers the chance of lapsing, which agrees with the forest.
* **Recency (+0.04):** a longer gap raises the chance of lapsing, but the coefficient is almost zero, so I wouldn’t read much into it.
* **Cancellation_rate (-0.12):** the sign is counter-intuitive but small. It is probably an overlap with frequency, so avoid concluding that cancellations reduce lapsing.


**The retention list: what the result really shows**

The top 10 Premium customers have predicted lapse probabilities of only 0.45 to 0.50, and just 1 of the 10 truly lapsed. This is not a bug. The Premium flag uses whole-window RFM, which includes September spending, so customers who ordered in September look like high-value customers. Those are the same customers who did not lapse. This is why the list is illustrative and not a fair evaluation.
"""
