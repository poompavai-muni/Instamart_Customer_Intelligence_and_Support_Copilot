# Instamart Customer Intelligence & Support Copilot

A capstone project built as one repository in four connected Parts: a trusted analytical database (Part A), customer segments and a lapse model (Part B), a safe support-ticket triage pipeline (Part C), and a policy question-answering assistant with a refund agent (Part D).

All data is synthetic and fictional, produced by the supplied `generate_data.py`. All monetary values are in Indian Rupees (INR). All language-model calls go through the supplied offline `mock_llm.py`, so no network, API key or paid service is used anywhere.

## Overview

| Part | What it builds | Main outputs |
|---|---|---|
| A | Audits and cleans three raw exports, stores them in SQLite, validates them, answers seven SQL questions and cross-checks in pandas | `data/clean/*.csv`, `data/instamart.db`, `sql/queries.sql` |
| B | RFM table, K-means segments (K = 4), PCA, a lapse classifier with a class-balance experiment, and a Premium retention list | `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| C | Input guardrails, PII redaction, zero-shot and few-shot prompts, retries, output validation, a token budget and routing | `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv` |
| D | TF-IDF retrieval over five policies, a grounded answer function, five tools, a ReAct refund agent with safety limits and crash recovery | `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/` |

How the Parts connect: Part B reads Part A's clean orders and customers. Part C reads Part B's segment file to route tickets. Part D's agent reads Part A's database.

**Repository layout**

| Path | Purpose |
|---|---|
| `generate_data.py`, `mock_llm.py` | The two supplied files, unmodified |
| `requirements.txt`, `.gitignore` | Pinned libraries; ignores `.env`, `.venv/`, `venv/`, `__pycache__/` |
| `src/part_A_data_foundation_clean_store_query.py` | Part A scripts (audit, clean and store, validate, SQL runner, pandas analysis) |
| `sql/queries.sql` | The seven SQL queries |
| `src/part_B_customer_intelligence_segments_and_lapse_prediction.py` | Part B |
| `src/part_C_support_ticket_triage.py` | Part C |
| `src/part_D_policy_rag_assistant_and_refund_agent.py` | Part D |
| `data/raw/`, `data/clean/`, `data/policies/`, `data/instamart.db` | Raw, cleaned and policy data, and the SQLite database |
| `outputs/`, `state/` | Result files and agent checkpoints |
| `notebooks/` | Optional companion notebooks with explanations; the scripts above are the reproducible pipeline |

Rules followed throughout: only the Python standard library, pandas, numpy, scikit-learn and joblib are used; every random step uses seed 42; no chart is committed; no image is included.

## Reproduce

Run these commands from a clean clone, in order. Python 3.11 or newer is required (tested on Python 3.12 and 3.13).

1. `git clone https://github.com/poompavai-muni/Instamart_Customer_Intelligence_and_Support_Copilot.git`
2. `cd Instamart_Customer_Intelligence_and_Support_Copilot`
3. `python -m venv .venv`
4. `source .venv/bin/activate` (on Windows: `.venv\Scripts\activate`)
5. `pip install -r requirements.txt`
6. `python generate_data.py`
7. `python -m src.part_A_data_foundation_clean_store_query.py`
8. `python -m src.part_B_customer_intelligence_segments_and_lapse_prediction.py`
9. `python -m src.part_C_support_ticket_triage.py`
10. `python -m src.part_D_policy_rag_assistant_and_refund_agent.py`
11. `python mock_llm.py`

What to look for:

| Step | Expected evidence |
|---|---|
| 6 | Last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164` |
| 9 | Last line: `ALL CHECKPOINTS PASSED` |
| 12 to 14 | Every `CHECK PASSED` line prints; each script ends without an error |

After step 14 every required path exists: `data/clean/customers.csv`, `data/clean/products.csv`, `data/clean/orders.csv`, `data/instamart.db`, `sql/queries.sql`, `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv`, `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv`, `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl` and the folder `state/`.

Notes: run the commands from the repository root (the scripts use relative paths and `python -m` so that `mock_llm.py` can be imported). `requirements.txt` pins the library versions used; a saved model (`.joblib`) must be loaded with the same scikit-learn version that saved it.

## Part A - Data Foundation: Clean, Store, Query

**Setup.** The repository is on branch `main`. A virtual environment holds the libraries; `requirements.txt` pins them. `.gitignore` lists `.env`, `.venv/`, `venv/` and `__pycache__/`. The two supplied files are committed unmodified and `python generate_data.py` regenerates all raw data with a fixed seed.

### Audit of the raw data

Shapes: customers (1200, 5), products (40, 4), orders (13084, 9). Data types: identifiers, city, gender, status and timestamps are text; age and delivery_minutes are floats; quantity, discount_pct and price_inr are integers; order_amount is text because it holds non-numeric entries (exact dtype names depend on the pandas version).

Missing values (count and percentage), as pandas reads the files: customers `age` 61 (5.08%), `gender` 30 (2.50%); orders `order_amount` 53 (0.41%), `delivery_minutes` 2,541 (19.42%); every other column 0 (0.00%). Note that pandas silently reads the text `N/A` as missing, which is why `order_amount` shows 53 missing here although 120 entries cannot be used (see the table).

| Audit figure | Value |
|---|---|
| Raw orders rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status `pending` | 656 |
| Distinct `order_id` | 12,309 |
| `order_amount` entries that cannot be converted to a number | 120 (67 `error` + 53 `N/A`) |
| Negative `delivery_minutes` | 106 |
| Blank `delivery_minutes` | 2,541 |
| Customers: blank `age` | 61 |
| Customers: `age` outside 10-100 | 6 |
| Customers: distinct raw `city` spellings | 24 |
| Customers: distinct non-blank raw `gender` spellings | 10 |
| Customers: blank `gender` | 30 |

The difference between 13,084 rows and 12,309 order ids is 775 rows: 119 exact duplicates plus 656 older `pending` records.

### Cleaning rules and what each one changed

**Customers**

| Rule | Values changed |
|---|---|
| City: strip spaces, title case | 24 spellings become 6 cities |
| Gender: lower-case, map `m`/`male` to male and `f`/`female` to female, anything else to unknown | 10 spellings become 3 values; the 30 blanks become unknown |
| Age: values outside 10-100 set to missing, then every missing age filled with the median age of the customer's city (`groupby` and `transform`) | 6 invalid + 61 blank = 67 ages filled; filled values are exact medians (Hyderabad 37.5, Mumbai 38.5), not rounded |
| `signup_date` converted to a datetime | all 1,200 rows |

**Orders** (applied in this order)

| # | Rule | Rows or values changed |
|---|---|---|
| 1 | Convert `order_ts` to a datetime | all rows |
| 2 | Remove exact duplicate rows | 119 removed |
| 3 | For each `order_id` keep only the record with the latest `order_ts` | 656 older records removed |
| 4 | Convert `order_amount` to numeric (unparseable becomes missing), then fill missing amounts with `round(price_inr x quantity x (1 - discount_pct / 100), 2)` | 112 amounts filled |
| 5 | Set negative `delivery_minutes` to missing; fill missing times of completed orders with the median of that order's product category among completed orders; leave cancelled orders missing | 105 negatives set to missing; 774 completed times filled; 1,199 cancelled left missing |

The cleaning counts (112, 105) are smaller than the raw audit counts (120, 106) because some bad values sat on rows that rules 2 and 3 had already removed. The order of the rules matters for the same reason: removing duplicates first avoids repairing rows that are about to be deleted. Cancelled orders keep a missing delivery time because they were never delivered, and a filled value would be invented data.

Result after cleaning: 12,309 orders (one per `order_id`), 11,110 completed and 1,199 cancelled, no pending rows, 0 missing `order_amount`, 0 missing delivery times among completed orders, and completed revenue of INR 2,998,668.80. Cities: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Genders: female 610, male 560, unknown 30. No missing ages; every age is between 18 and 60.

### Store and validate

The clean tables are written to `data/clean/customers.csv`, `products.csv`, `orders.csv` and to `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `src/a3_validate.py` asserts 15 reference checkpoints (row counts, unique ids, status counts, no pending rows, missing-value counts, revenue, city and gender counts, age range) and prints exactly `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run from Python with `pd.read_sql_query`)

| Query | Result |
|---|---|
| Q1 `WHERE` then `HAVING`: completed orders above 500, customers with a total above 8000 | 30 customers |
| Q2 three-table join: revenue by city and category | 30 rows; they add up to INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery: products priced above the average | 11 products |
| Q4 `LEFT JOIN`: customers with no completed order | 25 customers |
| Q5 two CTEs: customers above the average total spend | 550 customers; average INR 2,552.06 |
| Q6 `RANK`, `DENSE_RANK`, `ROW_NUMBER` per city | 8 rows where `RANK()` differs from `ROW_NUMBER()` |
| Q7 daily revenue with a running total | 183 daily rows; last cumulative value INR 2,998,668.80 |

Top customer per city (Q6): Bengaluru C0082 (INR 12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10).

Why ties make `RANK()` differ from `ROW_NUMBER()`: customers with equal total spend share the same `RANK()`, while `ROW_NUMBER()` always gives every row a different number.

### pandas cross-check and exploratory analysis

The pivot table (`pd.pivot_table(..., margins=True)`) has a grand total of INR 2,998,668.80, equal to the SQL total from Q2 and Q7. Completed orders per city:

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Statistics of completed `order_amount`: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR 232.20. The 1.5 x IQR fences are -283.50 and 645.30: 0 outliers below the lower fence and 897 above the upper fence. The correlation matrix over completed orders (`order_amount`, `discount_pct`, `quantity`, `delivery_minutes`) has its largest absolute value for `order_amount` with `discount_pct` at -0.095; `order_amount` with `quantity` is +0.087.

**Analysis (under 200 words).** The data describes 11,110 completed quick-commerce orders in six Indian cities, in INR, with quantity, discount and delivery time. Order values are strongly right-skewed: the mean (269.91) is almost double the median (135.00) and skewness is 4.61, so a few large baskets pull the average up and the median is the better typical order. 897 orders lie above the upper fence of 645.30 and none below the lower one, which is expected for a positive, skewed quantity; they are most likely expensive products or large quantities, not errors. All correlations are very weak. The strongest, between order amount and discount (-0.095), is small, and a correlation of this size does not establish causation. Revenue is spread fairly evenly across the cities, with Hyderabad highest in total and Mumbai highest per order.

## Part B - Customer Segments and Lapse Prediction

Uses completed orders for customer behaviour. Reads `data/clean/orders.csv` and `data/clean/customers.csv`; writes the three files in `outputs/`.

### RFM table (snapshot 2025-10-01 00:00:00)

Recency is whole days from the snapshot to the last completed order, frequency is the number of distinct completed orders, and monetary is the sum of completed `order_amount`. Result: 1,175 customers (the other 25 never completed an order). Recency median 19, minimum 0, maximum 180; frequency mean 9.46, maximum 33; monetary mean 2,552.06, maximum 14,766.55.

### K-means (StandardScaler, `KMeans(n_clusters=K, n_init=10, random_state=42)`)

| K | Inertia | Silhouette | Inertia drop vs previous K |
|---|---|---|---|
| 2 | 1820.8 | 0.465 | - |
| 3 | 1268.2 | 0.445 | 30.3% |
| 4 | 743.1 | **0.526** | **41.4%** |
| 5 | 617.4 | 0.480 | 16.9% |
| 6 | 503.2 | 0.482 | 18.5% |
| 7 | 408.5 | 0.416 | 18.8% |
| 8 | 366.5 | 0.412 | 10.3% |

All values are inside the stated tolerances. **Why inertia alone cannot pick K:** inertia is the total squared distance from each point to its cluster centre, so adding clusters can only move centres closer to points; K=8 is therefore always lower than K=2, and it would reach zero when K equals the number of customers. This happens even for pure noise, so a lower inertia does not show that the extra clusters are real; the choice needs the elbow, the silhouette score and business interpretability.

**Final K = 4**, chosen from {3, 4, 5}:

| Criterion | K=3 | K=4 | K=5 |
|---|---|---|---|
| Elbow (inertia drop) | 30.3% | **41.4%, the sharpest bend** | 16.9% |
| Silhouette | 0.445 | **0.526, the highest of all K** | 0.480 |
| Smallest segment | 185 (15.7%) | 138 (11.7%) | 95 (8.1%) |
| Business reading | hides the high-basket customers inside the frequent buyers | four distinct groups, each with a clear action | splits the inactive group further without a new action |

### Segment profile, names and actions

Cluster numbers are only labels with no ordering; segments are named from their profile.

| Segment | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|
| **Premium** | 138 | 24.12 | 6.77 | 7,129.70 | VIP perks, early access and free-delivery thresholds; protect from churn |
| Frequent loyal | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty rewards or a subscription to lock in the habit |
| Occasional low-value | 436 | 28.86 | 3.77 | 1,057.62 | Bundles and nudges to raise order frequency |
| Lapsing | 182 | 109.04 | 2.20 | 685.99 | Time-limited win-back offer |

Premium is the segment with the highest mean monetary value, and `is_premium` is true for exactly that segment. PCA on the scaled RFM matrix (`PCA(n_components=2)`): explained variance ratios 0.634 and 0.218, total 0.852. `outputs/customer_segments.csv` has the columns `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium` (1,175 rows).

### The lapse problem (cutoff 2025-09-01)

The cohort is every customer with at least one completed order before the cutoff: **1,156 customers**. `lapsed = 1` if the customer has no completed order from 2025-09-01 through 2025-09-30: **443 lapsed (38.32%)** and 713 not lapsed (61.68%). Features (built only from orders before the cutoff plus the clean customer table): `recency` (days from the cutoff), `frequency`, `monetary`, `avg_disc_pct`, `cancellation_rate` (cancelled divided by completed plus cancelled orders), `avg_delivery`, `favourite_category` (ties go to the first alphabetically), `city`, `age`, `gender` and `tenure_days`. The split is `train_test_split(test_size=0.2, stratify=y, random_state=42)`: 924 training and 232 test rows. Baseline (always predict not lapsed): accuracy 61.6%, lapsed-class F1 0.0.

### Preprocessing without leakage

Every feature column has 0 missing values (checked by an assertion). Numeric columns are standardised and `city`, `gender`, `favourite_category` are one-hot encoded with k-1 columns each (5 + 2 + 4 = 11 dummy columns plus 8 numeric = 19), inside a scikit-learn `Pipeline` fitted on the training rows only. The scaler's learned mean of `frequency` is 8.1147, while the full-cohort mean is 8.0285; they differ because the scaler is fitted only on the 924 training rows, not on the 232 test rows.

### Training, tuning and evaluation

| Model | Search | Best setting | Best CV F1 |
|---|---|---|---|
| Logistic regression (`max_iter=1000`) | `C` in {0.01, 0.1, 1, 10, 100}, `cv=5`, `scoring="f1"` | `C = 100` (the edge of the grid) | 0.651 |
| Random forest (`n_estimators=100`, `random_state=42`) | `max_depth` in {3, 5, 8} x `min_samples_leaf` in {5, 10, 20}, `cv=5` (45 fits) | depth 5, leaf 10 | 0.687 |

Test-set results for the lapsed class (confusion matrix is [[TN, FP], [FN, TP]]):

| Model | Confusion matrix | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[110, 33], [29, 60]] | 0.645 | 0.674 | **0.659** |

**Selected model: random forest** (test F1 0.659, above the required 0.55). Missing a lapsing customer loses that customer's future revenue with no chance to win them back, while wrongly targeting a loyal customer costs only a small discount or message, so recall on the lapsed class matters more than precision. The forest found 60 of the 89 lapsing customers (recall 0.674 against 0.652) with almost the same number of false alarms (33 against 32), and it also had the higher cross-validated F1 (0.687 against 0.651). The two models are close (two customers on the test set), so the cost argument supports the choice more than the F1 difference does. The test set was used only after tuning was finished.

### Class-balance experiment

Only the training set was over-sampled: every original training row was kept and randomly chosen duplicates of lapsed rows (with replacement, `random_state=42`) were added until both classes had 570 rows (1,140 rows). The best logistic regression was refitted and evaluated on the original, unchanged 232-row test set.

| Lapsed-class metric | Un-balanced | Over-sampled | Change |
|---|---|---|---|
| Precision | 0.644 | 0.589 | -0.055 |
| Recall | 0.652 | 0.854 | **+0.202** |
| F1 | 0.648 | 0.697 | +0.049 |

Confusion matrices: un-balanced [[111, 32], [31, 58]]; over-sampled [[90, 53], [13, 76]]. Recall moved most. For the retention team this means the over-sampled model catches 76 of the 89 customers who really lapsed instead of 58 (18 more at-risk customers receive an offer), at the price of 21 extra loyal customers contacted (53 false alarms instead of 32); that is a good trade when an offer costs much less than losing a customer. Over-sampling inflates predicted probabilities, so they should be read as a ranking.

### Interpretation, saved model and retention list

| Rank | Random forest feature | Importance | Business reading |
|---|---|---|---|
| 1 | `frequency` | 0.366 | Ordering often builds a habit: the strongest sign that a customer stays |
| 2 | `monetary` | 0.229 | High total spend goes with commitment (it overlaps with frequency) |
| 3 | `recency` | 0.090 | A longer gap since the last order signals a fading customer |
| 4 | `cancellation_rate` | 0.084 | Many cancelled orders point to service friction |
| 5 | `avg_delivery` | 0.052 | Slower deliveries suggest dissatisfaction |

| Logistic regression feature | Coefficient | Reading |
|---|---|---|
| `frequency` | -1.698 | Strongly lowers the chance of lapsing |
| `recency` | +0.040 | Raises it, but the size is almost zero |
| `cancellation_rate` | -0.124 | Small and counter-intuitive; probably overlaps with frequency, so it should not be over-read |

The selected model is saved with `joblib.dump` to `outputs/lapse_model.joblib`; the reloaded model reproduces the original test predictions exactly. Among the 40 Premium customers in the test set, the top 10 by predicted lapse probability (0.4565 to 0.5048) are written to `outputs/retention_list.csv` (columns `customer_id, lapse_probability, actually_lapsed`); 1 of these 10 truly lapsed. The retention list is illustrative and not a leakage-free evaluation, because the Premium flag comes from the whole-window RFM, which includes the September window that defines the lapse label.

### Leakage statement

No encoder, scaler or resampler saw test rows: the split came first, the scaler and encoder were fitted on training rows only inside the pipeline, and over-sampling duplicated training rows only. No data dated 2025-09-01 or later entered the features (asserted in code); the September window was used only to build the label. The segment labels from the clustering were not used as features.

## Part C - Support Ticket Triage with Guardrails

`src/c_triage.py` processes `data/raw/tickets.csv` (164 tickets, read with `keep_default_na=False`) in file order.

**Input guardrails.** Before any model call, a ticket whose text is empty or whitespace gets `skipped_empty`, and a ticket with `est_tokens(text) > 400` gets `flagged_long`. Result: 4 empty tickets, 4 long tickets (each estimated at 734 tokens) and 156 eligible tickets. An assertion shows that none of them can reach the model.

**Redaction.** The three patterns are applied in this order to every ticket before it goes into a prompt; the user message is exactly the redacted text.

| Type | Pattern | Placeholder | Substitutions |
|---|---|---|---|
| Email | `[\w.+-]+@[\w-]+\.[\w.]+` | `[EMAIL]` | 55 |
| Phone | `\b[6-9]\d{9}\b` | `[PHONE]` | 91 |
| Order id | `ORD-\d{6}` | `[ORDER_ID]` | 132 |
| Total | | | **278** |

Re-applying the three patterns to the redacted texts finds 0 matches (asserted). An assertion inside the run loop also checks that no email, phone or order id appears in any message sent to the model.

**Prompts.** Both prompts contain `CLASSIFY`, the four categories (billing, delivery, technical, fraud) and the JSON-only reply form `{"category": "...", "reason": "..."}`. The zero-shot prompt (38 estimated tokens) has no `EXAMPLES:` line. The few-shot prompt (119 estimated tokens) is the same text plus an `EXAMPLES:` line with four labelled examples, one per category including fraud.

**Resilient calls.** Every model call goes through a retry function: at most 3 retries, with a wait of `BASE_DELAY * 2**n * random.uniform(0.9, 1.15)` before retry number n (`BASE_DELAY = 0.05` s), one log line per retry, and `mock_llm.reset()` before each full run. A full run over the 156 eligible tickets produces exactly 19 `TransientLLMError`s, all recovered by a retry, and no permanent error. With this mock every retry is retry number 0, so every wait lies between 0.045 and 0.0575 seconds (checked in code). Sample log lines:

```
[retry] ticket=T0002 retry_number=0 wait=0.053s
[retry] ticket=T0116 retry_number=0 wait=0.045s
[retry] ticket=T0153 retry_number=0 wait=0.048s
```

**Output validation.** A reply that is valid JSON with an allowed category is `classified_json`. Otherwise the lowercase words are taken with `re.findall(r"[a-z]+", text.lower())` and the reply is `classified_keyword` if exactly one allowed category word is present. Everything else, including valid JSON with a category outside the list, is `flagged_invalid_output` and goes to human review. A full run gives 135 `classified_json`, 12 `classified_keyword` (the mock's chatty replies such as "The customer appears to have a billing-related problem.") and 9 `flagged_invalid_output` (all `{"category": "shipping", ...}`). A plain whitespace split would give 135 / 0 / 21, because it would not separate `billing-related`.

### Zero-shot against few-shot (full run, no token budget)

| Run | classified_json | classified_keyword | Classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|---|---|---|
| Zero-shot | 135 | 12 | 147 | 126 | 85.7% | 10,336 | 70.3 |
| Few-shot | 135 | 12 | 147 | 147 | 100.0% | 22,972 | 156.3 |

Total tokens is the sum of `response.usage.total_tokens` over the 156 successful replies. Confusion table for the zero-shot run (rows are the true category, columns the predicted one):

| true \ predicted | billing | delivery | technical | fraud |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

Few-shot raised accuracy from 85.7% (126 of 147) to 100%, but it cost 22,972 tokens against 10,336, which is 156.3 tokens per classified ticket instead of 70.3. The 12,636 extra tokens bought 21 more correct tickets, about 602 tokens each. All 21 zero-shot errors were fraud or technical tickets sent to billing, and misrouting a fraud ticket is costly because it delays the security team, so the extra tokens are worth paying when errors are expensive. When wrong labels are cheap, or when the budget is tight and the labels are easy, zero-shot at roughly 45% of the cost is the better choice.

### Budget, routing and output files

- **Budget demo** (`outputs/triage_budget_demo.csv`): the zero-shot pipeline with `TOKEN_BUDGET = 6000` stopped after 6,036 tokens; the budget is checked before each call, so the last call can push the total slightly over. **65 tickets** ended as `budget_exhausted` (the first was T0093), and none of them, nor any empty or long ticket, caused a model call (the mock's own call counter equals the calls the code made).
- **Final results** (`outputs/triage_results.csv`): a few-shot run with `TOKEN_BUDGET = 100000`, 164 rows with the columns `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`. Routing reads `outputs/customer_segments.csv`: statuses `skipped_empty`, `flagged_long`, `flagged_invalid_output` and `budget_exhausted` go to `human_review` (NORMAL); otherwise `fraud` goes to `security_team` (HIGH); otherwise a Premium customer goes to `premium_desk` (HIGH); otherwise `support_queue` (NORMAL). A customer with no segment row counts as not premium (0 such tickets here).

| Route | Priority | Tickets |
|---|---|---|
| `security_team` | HIGH | 34 |
| `premium_desk` | HIGH | 11 |
| `support_queue` | NORMAL | 102 |
| `human_review` | NORMAL | 17 |

**Which outputs were rule-validated and which went to humans.** 147 of the 156 model replies passed automatic rules (135 strict JSON, 12 by the single-category-word rule). 17 tickets went to human review in the final run: 9 invalid model outputs, 4 empty tickets and 4 over-long tickets; in the budget demo a further 65 tickets were stopped by the budget. The few-shot prompt costs 156.3 tokens per classified ticket against 70.3 for zero-shot.

## Part D - Policy RAG Assistant and Refund Agent

`src/d_rag_agent.py` builds retrieval over the five policies in `data/policies/` and a refund agent that uses `data/instamart.db`.

**Chunking.** Each document is split into chunks of 40 words with an overlap of 10 (a new chunk starts 30 words later; the file name is stored with every chunk): 15 chunks in total.

| Document | Words | Chunks |
|---|---|---|
| refund_policy.txt | 119 | 4 |
| delivery_policy.txt | 100 | 3 |
| escalation_policy.txt | 78 | 3 |
| payment_policy.txt | 86 | 3 |
| privacy_policy.txt | 68 | 2 |

**Retrieval.** `TfidfVectorizer()` is fitted on the 15 chunk texts (vocabulary of 200 words); a query is transformed with the same fitted vectorizer and chunks are ranked by cosine similarity. `retrieve(query, k=3)` returns dictionaries with `source`, `score` and `text`. For `refund policy late delivery` the top three chunks all come from `refund_policy.txt` (scores 0.2719, 0.2514, 0.2454).

TF (term frequency) measures how often a word occurs inside one chunk, so a chunk that repeats a word is treated as more about it. IDF (inverse document frequency) measures how rare a word is across all chunks, so distinctive words such as "refund" count for more than words that appear everywhere. **Limitation:** TF-IDF matches exact words only, so paraphrases and different word forms are missed; the query "how do I escalate a complaint to a manager" ranked the refund policy first because "escalate" is not in the vocabulary (the policy says "escalation"), and "money back because my groceries arrived slowly" retrieved the escalation policy through the single shared word "back".

**Answer.** `answer(question)` retrieves three chunks and sends two messages to `mock_llm.chat`: a system message containing `POLICY ASSISTANT` (answer only from the context, otherwise say the information could not be found) and a user message of the form `CONTEXT:`, the three chunks each preceded by `[<source file name>]` on its own line, `QUESTION:` and the question. It returns the model's text and the sources used.

### Evaluation of retrieval and grounding (`data/raw/policy_eval.csv`, 14 questions)

| Overlap | Chunks | Answerable: expected document in top 3 | Answerable: expected fact in answer | Unanswerable: fallback "I could not find" |
|---|---|---|---|---|
| 10 | 15 | 12/12 | 12/12 | 2/2 |
| 0 | 13 | 12/12 | 10/12 | 2/2 |

With zero overlap two questions break. "What delivery time makes a completed order eligible for a full refund?" (expected fact `more than 45 minutes`) returns only "receives a full refund of the order amount." because the sentence "If a completed order is delivered in more than 45 minutes, the customer receives a full refund of the order amount." starts in chunk 0 and ends in chunk 1 of `refund_policy.txt`, so no chunk holds it whole. "How soon is money returned when a payment fails but is deducted?" (expected fact `48 hours`) fails for the same reason: the sentence is cut after "returned automatically" and "within 48 hours" falls into the next chunk. The 10-word overlap keeps such sentences whole.

### Tools

| Tool | Behaviour |
|---|---|
| `get_order(order_id)` | Reads `data/instamart.db`; returns `order_id`, `status`, `order_amount`, `delivery_minutes`, or `{"error": "order not found"}` (parameterised SQL) |
| `search_policy(query)` | Returns the `retrieve` output under the key `results` |
| `calculate_refund(order_amount, delivery_minutes)` | More than 45 minutes: 100% of `order_amount`; 45 minutes or less, or no delivery time: 0 |
| `issue_refund(order_id, refund_inr)` | Appends one JSON line (timestamp, tool name) to `outputs/refund_ledger.jsonl`; refuses amounts above INR 1000 as a safety guard |
| `escalate_to_human(order_id, refund_inr)` | Appends one JSON line to the ledger |

`TOOLS_SPEC` lists the five tools with a one-sentence description and a JSON-style parameter specification, and `TOOL_MAP` maps each tool name to its Python function; the agent looks the model's requested name up in `TOOL_MAP` and calls the function with the `arguments` dictionary.

### ReAct agent: the seven scenarios

`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` starts with a system message containing `AGENT`, calls the model, executes requested tools (assistant message with `tool_calls`, then a `tool` message with a JSON string) and loops until a final answer. Each run is appended as one JSON line to `outputs/agent_runs.jsonl` with every step's tool, arguments and observation.

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,147 |
| S2 | REFUND | 135.00 | 5 | 1,147 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,134 |
| S4 | DENIED | 0.00 | 4 | 805 |
| S5 | DENIED | 0.00 | 3 | 487 |
| S6 | NOT_FOUND | 0.00 | 2 | 172 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,300 |

Every refund amount equals `expected_refund_inr` within 0.01, and S1 takes 5 model calls. The ledger `outputs/refund_ledger.jsonl` holds exactly the three refund actions: `issue_refund` for ORD-100048 (128.25), `issue_refund` for ORD-100067 (135.00) and `escalate_to_human` for ORD-100145 (1,999.00).

### Stop-safely demonstrations

| Demonstration | Status | Model calls | Tokens |
|---|---|---|---|
| S7 under the default `token_limit=6000` | `FALLBACK_MAX_ITERATIONS` | 6 | 3,300 |
| S1 with `token_limit=200` | `FALLBACK_TOKEN_LIMIT` | 2 | 202 |
| S1 with `time_limit_seconds=0` | `FALLBACK_TIMEOUT` | 0 | 0 |

The token limit is checked after each model call and before the requested tool runs (the tool just requested is not executed); the time limit is checked at the top of each iteration with `time.monotonic()`. Every fallback returns the steps completed so far and a handoff summary in `final_answer`, for example: "FALLBACK_TOKEN_LIMIT: used 202 tokens, above the limit of 200; the tool just requested (search_policy) was not executed. Steps completed: get_order. A human should take over this request (order ORD-100048)."

### Crash and resume (scenario S1)

After every tool observation the run writes `state/<run_id>.json` (run id, scenario id, full message list, `tokens_used`, step number, tools executed, and the loop settings). The demo raises a simulated crash right after the second tool execution (after its checkpoint is saved) and then calls `resume_agent("demo_resume_S1")`.

| Part of the run | Tools executed, in order |
|---|---|
| Before the crash | `get_order`, `search_policy` |
| After the resume | `calculate_refund`, `issue_refund` |
| **Across both parts** | `get_order`, `search_policy`, `calculate_refund`, `issue_refund` (four executions, none repeated) |

The final status is `REFUND` with 128.25, using 5 model calls and 1,147 tokens, identical to the uninterrupted run, and the ledger holds a single line for the refund. A crash after tool 1, 2, 3 or 4 always resumes to the same result, and resuming a finished run returns the saved result without running anything again. `state/` holds `run_S1.json` to `run_S7.json` and `demo_resume_S1.json`.

### ReAct versus plan-and-execute

My ReAct agent asks the model before every action. It needed 5 model calls for S1 (1,147 tokens), 3 for S5 (487) and 2 for S6 (172). Cost grows faster than calls because every call resends the history: S1's calls cost 89, 113, 302, 317 and 326 tokens, as the policy passages are resent three times. A plan-and-execute design asks for the whole step list once and runs the tools in code, so S1 would need about 1 to 2 calls and roughly 130 to 450 tokens (estimated, not built).

The price is flexibility. ReAct sees every observation: for S6, "order not found" ended the run after 2 calls, and for S5 the cancelled status led straight to DENIED with no refund calculated. A planner replans only on an error, so NOT_FOUND would trigger a replan (about 2 calls, like ReAct), but a cancelled order is a valid result, not an error, so a fixed plan would go on to calculate and issue unless extra checks stopped it.

I would choose ReAct here. The next action depends on data (status, delivery minutes, amounts above INR 1,000 go to a human), and a wrong step can pay money. Checking each observation before acting is safer, and with at most five tools and hard limits (6 iterations, 6,000 tokens, 10 seconds) the extra calls cost little. At scale I would plan the fixed lookups once and keep ReAct for the decision.

No real network call and no API key is used anywhere in this Part.

## Limitations and Next Steps

**Limitations**

- **The retention list is illustrative and not a leakage-free evaluation.** The Premium flag is a post-hoc whole-window label: it comes from the whole-window RFM, which includes the September window that defines the lapse label, so Premium customers are more likely to be active in September (only 1 of the 10 listed customers truly lapsed). The model's own features still use only orders before the cutoff.
- **The data and the language model are simulated.** The data is synthetic with planted defects, and `mock_llm.py` is scripted: the Part C accuracies and the Part D agent path show that the pipeline, guardrails and loop work, not that a real model would reason correctly.
- **Small test set.** The lapse test set has 232 customers (89 lapsed), so the difference between the two models (two customers) is within noise; logistic regression's best `C` sits at the edge of its grid. Over-sampling raises recall but inflates predicted probabilities.
- **Median imputation** of ages and delivery times reduces variance and is simple to audit, but a model-based imputation might be more accurate.
- **Retrieval is word-based.** TF-IDF misses paraphrases and word forms, and word-count chunking can cut a sentence in two (visible in the zero-overlap run); answers can begin mid-sentence.
- **Regular-expression redaction** only finds the formats it was written for; names, addresses, and phone numbers with a prefix or spaces would pass through.
- **Resume guarantee.** A crash between executing a tool and saving its checkpoint would repeat that tool on resume (at-least-once). Tools are not idempotent apart from the guard in `issue_refund`.
- **Counts depend on prompt wording and library versions.** The budget-demo count (65) and the token totals depend on the exact prompts; estimator results carry small tolerances across scikit-learn versions.

**Next steps**

- Validate over several cutoffs (time-based cross-validation), calibrate predicted probabilities, and choose a contact threshold from a cost model.
- Define Premium from pre-cutoff data only, so the retention list becomes a fair test.
- Replace TF-IDF with stemming or embedding-based retrieval and chunk on sentence boundaries.
- Make every tool idempotent (for example, refuse a second refund for the same order) and add automated unit tests for the cleaning, routing and agent rules.
- Try a real hosted model behind the same interface, keeping any key in a `.env` file that is listed in `.gitignore`.
