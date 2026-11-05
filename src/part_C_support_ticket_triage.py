# !git clone https://github.com/poompavai-muni/Instamart_Customer_Intelligence_and_Support_Copilot.git

# import os
# List the contents of the cloned directory to verify its structure
# os.listdir('Instamart_Customer_Intelligence_and_Support_Copilot')

# Commented out IPython magic to ensure Python compatibility.
# %cd Instamart_Customer_Intelligence_and_Support_Copilot
# %pip install -r requirements.txt

import os

# Find the project folder without pathlib or shutil.
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

"""## Part C — Support Ticket Triage with Guardrails

**Goal:** redact personal information, classify support tickets using the supplied offline `mock_llm.py`, validate outputs, retry temporary errors safely, respect the token budget, and route tickets using Part B customer segments.

**Before running:** complete Part A and Part B. The supplied `mock_llm.py` must not be edited.

## 1. Input Guardrails
"""

import pandas as pd
from mock_llm import est_tokens          # supplied estimator: ceil(number of words x 1.3)

MAX_INPUT_TOKENS = 400

def input_guardrail(text):
    """Return a status if the ticket must NOT be sent to the model, otherwise None."""
    # (a) empty or whitespace-only text
    if text is None or str(text).strip() == "":
        return "skipped_empty"
    # (b) text that is too long
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return None

# keep_default_na=False: an empty text must stay "" and not become NaN
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
print("Tickets loaded:", len(tickets), "(kept in file order)")

# apply the guardrail to every ticket, in the order they appear
tickets["guard_status"] = [input_guardrail(text) for text in tickets["text"]]

empty = tickets[tickets["guard_status"] == "skipped_empty"]
long_ = tickets[tickets["guard_status"] == "flagged_long"]
eligible = tickets[tickets["guard_status"].isna()]

print("Skipped_empty:", len(empty), "| Flagged_long:", len(long_), "| Eligible:", len(eligible))
print("Estimated tokens of the long tickets:", sorted({est_tokens(t) for t in long_["text"]}))

# proof: nothing empty or too long is in the list allowed to reach the model
for text in eligible["text"]:
    assert text.strip() != "", "an empty ticket is eligible"
    assert est_tokens(text) <= MAX_INPUT_TOKENS, "a long ticket is eligible"
assert len(empty) + len(long_) + len(eligible) == len(tickets)
print("ASSERTIONS PASSED: no empty or long ticket can reach the model")

"""## 2. Redact Personal Information (PII Redaction)"""

import re

# The patterns and placeholders are fixed by the brief and applied IN THIS ORDER.
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_ID_PATTERN = r"ORD-\d{6}"

REDACTION_RULES = [                      # (name, pattern, placeholder)
    ("email", EMAIL_PATTERN, "[EMAIL]"),
    ("phone", PHONE_PATTERN, "[PHONE]"),
    ("order_id", ORDER_ID_PATTERN, "[ORDER_ID]"),
]

def redact(text):
    """Return (redacted_text, counts) where counts says how many of each type were replaced."""
    counts = {}
    for name, pattern, placeholder in REDACTION_RULES:
        text, n = re.subn(pattern, placeholder, text)    # subn = substitute AND count
        counts[name] = n
    return text, counts

def contains_pii(text):
    """True if any of the three patterns still matches the text."""
    return any(re.search(pattern, text) for _, pattern, _ in REDACTION_RULES)

# redact all 164 tickets and add up the substitutions
totals = {"email": 0, "phone": 0, "order_id": 0}
redacted_texts = []
for text in tickets["text"]:
    clean_text, counts = redact(text)
    redacted_texts.append(clean_text)
    for name in totals:
        totals[name] += counts[name]
tickets["redacted_text"] = redacted_texts

print("emails:", totals["email"], "| phones:", totals["phone"], "| order ids:", totals["order_id"])
print("Total substitutions:", sum(totals.values()))

# show by code that nothing is left: apply the three patterns again to the redacted texts
left = {name: sum(len(re.findall(pattern, t)) for t in tickets["redacted_text"])
        for name, pattern, _ in REDACTION_RULES}
print("left after redaction:", left, "\ntotal left:", sum(left.values()))
assert sum(left.values()) == 0
assert not any(contains_pii(t) for t in tickets["redacted_text"])
print("ASSERTION PASSED: no email, phone or order id remains in any redacted ticket")

print("\nExample (before -> after):")
print("  before:", tickets.iloc[0]["text"])
print("  after :", tickets.iloc[0]["redacted_text"])

"""## 3. Zero-shot and few-shot prompts"""

import mock_llm

CATEGORIES = ["billing", "delivery", "technical", "fraud"]

# zero-shot: instructions only, no examples
ZERO_SHOT_PROMPT = (
    "You CLASSIFY customer support tickets into exactly one category. "
    "Allowed categories: billing, delivery, technical, fraud. "
    'Reply with JSON only, no other text, in the form {"category": "...", "reason": "..."}.'
)

# few-shot: the SAME instructions plus labelled examples after a line that says EXAMPLES:
FEW_SHOT_PROMPT = ZERO_SHOT_PROMPT + """
EXAMPLES:
Ticket: I was charged twice for one order. -> {"category": "billing", "reason": "duplicate charge"}
Ticket: My rider has not arrived and the order is late. -> {"category": "delivery", "reason": "late delivery"}
Ticket: The app crashes every time I log in. -> {"category": "technical", "reason": "app crash"}
Ticket: Someone used my card without my permission. -> {"category": "fraud", "reason": "unauthorized card use"}"""

def build_messages(system_prompt, redacted_text):
    """The system message holds ALL the instructions; the user message is the redacted ticket text only."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": redacted_text},
    ]

def check_prompts():
    """Assert every prompt rule from the brief."""
    for name, prompt in [("zero-shot", ZERO_SHOT_PROMPT), ("few-shot", FEW_SHOT_PROMPT)]:
        assert "CLASSIFY" in prompt, name + ": missing the word CLASSIFY"
        for category in CATEGORIES:
            assert category in prompt, name + ": missing category " + category
        json_form = '{"category": "...", "reason": "..."}'
        assert "JSON" in prompt and json_form in prompt, name + ": JSON reply form missing"
        assert not contains_pii(prompt), name + ": the prompt itself contains personal data"
    assert "EXAMPLES:" not in ZERO_SHOT_PROMPT, "zero-shot prompt must not contain EXAMPLES:"
    assert "EXAMPLES:" in FEW_SHOT_PROMPT, "few-shot prompt must contain EXAMPLES:"
    example_lines = FEW_SHOT_PROMPT.split("EXAMPLES:")[1].strip().split("\n")
    assert len(example_lines) >= 2, "need at least two labelled examples"
    assert any('"category": "fraud"' in line for line in example_lines), "need a fraud example"
    return len(example_lines)

n_examples = check_prompts()
print("ALL PROMPT RULES PASSED \nLabelled examples in the few-shot prompt:", n_examples)
print("zero-shot prompt:", est_tokens(ZERO_SHOT_PROMPT), "estimated tokens")
print("few-shot prompt :", est_tokens(FEW_SHOT_PROMPT), "estimated tokens")

# tiny demo: one ticket that mentions both fraud and billing words
mock_llm.reset()
demo = "Unauthorized charge on my card, please refund it."
for name, prompt in [("zero-shot", ZERO_SHOT_PROMPT), ("few-shot", FEW_SHOT_PROMPT)]:
    messages = build_messages(prompt, demo)
    assert messages[1]["content"] == demo        # the user message is exactly the ticket text
    reply = mock_llm.chat(messages)
    print(name, "reply:", reply.choices[0].message.content)
mock_llm.reset()                                 # leave the mock's counters clean

"""## 4. Resilient calls"""

import random
import time

MAX_RETRIES = 3
BASE_DELAY = 0.05          # seconds

def call_with_retry(messages, ticket_id, stats, log, verbose=True):
    """Call the model. If it raises TransientLLMError, wait and try again (max 3 retries).
    Returns (response, retries_used). Raises the error if all retries fail."""
    for n in range(MAX_RETRIES + 1):                 # n = 0 is the first try
        stats["model_calls"] += 1                    # Count every call that reaches the model

        try:
            return mock_llm.chat(messages), n
        except mock_llm.TransientLLMError:
            stats["transient_errors"] += 1
            if n == MAX_RETRIES:                     # no retries left: give up
                raise
            # wait before retry number n (n = 0, 1, 2): BASE_DELAY * 2**n * jitter
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            line = f"[retry] ticket={ticket_id} retry={n + 1}/{MAX_RETRIES} wait={wait:.3f}s"
            log.append(line)                         # the log always keeps the line
            if verbose:
                print(line)                          # printing can be switched off
            time.sleep(wait)

def run_model_calls(tickets, system_prompt, token_budget=None, verbose=True):
    """Send every allowed ticket to the model (in file order) and keep the raw replies."""
    mock_llm.reset()              # restart the mock's counters for this run
    random.seed(42)               # same jitter every run -> same log lines
    stats = {"transient_errors": 0, "permanent_errors": 0, "model_calls": 0,
             "total_tokens": 0}   # Sum of response.usage.total_tokens
    log = []
    records = []
    running_total = 0              # usage.total_tokens summed so far
    for _, row in tickets.iterrows():
        record = {"ticket_id": row["ticket_id"], "customer_id": row["customer_id"],
                  "status": None, "raw_reply": None, "tokens_used": 0, "retries": 0,
                  "cumulative_tokens": 0}
        blocked = input_guardrail(row["text"])       # Task 1 Input guardrail
        if blocked:
            record["status"] = blocked               # skipped_empty / flagged_long: no model call
            record["cumulative_tokens"] = running_total
            records.append(record)
            continue
        if token_budget is not None and running_total >= token_budget:     # Task 7 Budget
            record["status"] = "budget_exhausted"
            record["cumulative_tokens"] = running_total
            records.append(record)
            continue
        redacted_text, _ = redact(row["text"])       # Task 2 PII redaction
        messages = build_messages(system_prompt, redacted_text)   # Task 3 Prompting
        assert messages[1]["content"] == redacted_text
        assert all(not contains_pii(m["content"]) for m in messages), "personal data in a prompt"
        try:
            response, retries = call_with_retry(messages, row["ticket_id"], stats, log, verbose)
        except mock_llm.TransientLLMError:
            stats["permanent_errors"] += 1
            record["status"] = "failed_llm_error"
            records.append(record)
            continue
        record["status"] = "called"                  # Task 5 turns this into a validated status
        record["raw_reply"] = response.choices[0].message.content
        record["tokens_used"] = response.usage.total_tokens
        stats["total_tokens"] += response.usage.total_tokens      # The run's token total
        running_total += record["tokens_used"]
        record["cumulative_tokens"] = running_total
        record["retries"] = retries
        records.append(record)
    return pd.DataFrame(records), stats, log

# one full run with the zero-shot prompt
results, stats, log = run_model_calls(tickets, ZERO_SHOT_PROMPT)
called = results[results["status"] == "called"]
print("\nTickets sent to the model:", len(called))
print("\nTransientLLMErrors:", stats["transient_errors"], "\nRetry log lines:", len(log),
      "\nPermanent errors:", stats["permanent_errors"])
assert len(called) == 156
assert stats["transient_errors"] == 19 and len(log) == 19
assert stats["permanent_errors"] == 0
print("\nCHECK PASSED: 19 transient errors, all recovered by a retry, no permanent error")

"""## 5. Validate every answer"""

import json
import re

def validate_reply(reply):
    """Return (status, category) for one model reply. A reply is never guessed at."""
    if reply is None:
        return "flagged_invalid_output", None

    # step 1: is the reply valid JSON?
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        is_json = False
    else:
        is_json = True

    if is_json:
        category = data.get("category") if isinstance(data, dict) else None
        if isinstance(category, str) and category in CATEGORIES:
            return "classified_json", category
        # valid JSON but the category is missing or outside the list -> NO keyword fallback
        return "flagged_invalid_output", None

    # step 2: not JSON -> accept only if EXACTLY ONE allowed category word is present
    words = re.findall(r"[a-z]+", reply.lower())       # "billing-related" -> billing, related
    found = [category for category in CATEGORIES if category in words]
    if len(found) == 1:
        return "classified_keyword", found[0]
    return "flagged_invalid_output", None

def apply_validation(results):
    """Turn every 'called' row into classified_json / classified_keyword / flagged_invalid_output."""
    results = results.copy()
    results["category"] = None
    for i, row in results.iterrows():
        if row["status"] == "called":
            status, category = validate_reply(row["raw_reply"])
            results.at[i, "status"] = status
            results.at[i, "category"] = category
    return results

# full run with the retry printing switched off, then validation
raw_results, stats, log = run_model_calls(tickets, ZERO_SHOT_PROMPT, verbose=False)
checked = apply_validation(raw_results)

counts = checked["status"].value_counts()
print("Classified_json:", int(counts.get("classified_json", 0)),
      "\nClassified_keyword:", int(counts.get("classified_keyword", 0)),
      "\nFlagged_invalid_output:", int(counts.get("flagged_invalid_output", 0)))
assert counts["classified_json"] == 135
assert counts["classified_keyword"] == 12
assert counts["flagged_invalid_output"] == 9
print("CHECK PASSED: 135 / 12 / 9")

keyword_rows = checked[checked["status"] == "classified_keyword"]
flagged_rows = checked[checked["status"] == "flagged_invalid_output"]
print("A chatty reply that was accepted:", keyword_rows.iloc[0]["raw_reply"], "->", keyword_rows.iloc[0]["category"])
print("All flagged replies are:", sorted(set(flagged_rows["raw_reply"])))

"""## 6. Compare zero-shot and few-shot prompting"""

CLASSIFIED = ["classified_json", "classified_keyword"]

def evaluate_run(checked, tickets, stats):
    """Summarise one validated run. Returns (summary dict, table of classified tickets)."""
    labels = tickets[["ticket_id", "true_category"]]
    merged = checked.merge(labels, on="ticket_id", how="left")
    classified = merged[merged["status"].isin(CLASSIFIED)]
    n_json = int((merged["status"] == "classified_json").sum())
    n_keyword = int((merged["status"] == "classified_keyword").sum())
    n_classified = n_json + n_keyword
    n_correct = int((classified["category"] == classified["true_category"]).sum())
    total_tokens = stats["total_tokens"]                     # sum of response.usage.total_tokens
    assert total_tokens == int(merged["tokens_used"].sum())  # same number, counted two ways
    summary = {
        "classified_json": n_json,
        "classified_keyword": n_keyword,
        "classified": n_classified,
        "correct": n_correct,
        "accuracy_%": round(n_correct / n_classified * 100, 1),
        "total_tokens": total_tokens,
        "tokens_per_classified": round(total_tokens / n_classified, 1),
    }
    return summary, classified

def confusion_table(classified):
    """Rows = true category, columns = predicted category."""
    table = pd.crosstab(classified["true_category"], classified["category"])
    return table.reindex(index=CATEGORIES, columns=CATEGORIES, fill_value=0)

summaries = {}
classified_by_run = {}
for name, prompt in [("zero-shot", ZERO_SHOT_PROMPT), ("few-shot", FEW_SHOT_PROMPT)]:
    raw_results, stats, log = run_model_calls(tickets, prompt, verbose=False)
    checked = apply_validation(raw_results)
    summaries[name], classified_by_run[name] = evaluate_run(checked, tickets, stats)

comparison = pd.DataFrame(summaries).T.astype(
    {"classified_json": int, "classified_keyword": int, "classified": int,
     "correct": int, "total_tokens": int})
print(comparison.to_string())                                # to_string() shows every column

print("\nConfusion table, zero-shot run (rows = true category, columns = predicted):")
print(confusion_table(classified_by_run["zero-shot"]))

errors = classified_by_run["zero-shot"]
errors = errors[errors["category"] != errors["true_category"]]
print("\nZero-shot errors by (true -> predicted):")
print(errors.groupby(["true_category", "category"]).size())

for name in summaries:
    assert summaries[name]["classified_json"] == 135 and summaries[name]["classified_keyword"] == 12
assert summaries["zero-shot"]["classified"] == 147 and summaries["few-shot"]["classified"] == 147
assert summaries["zero-shot"]["correct"] == 126 and summaries["zero-shot"]["accuracy_%"] == 85.7
assert summaries["few-shot"]["accuracy_%"] == 100.0
print("\nCHECK PASSED: 147 classified (135 json + 12 keyword), zero-shot 126/147 = 85.7%, few-shot 100%")

extra_tokens = summaries["few-shot"]["total_tokens"] - summaries["zero-shot"]["total_tokens"]
extra_correct = summaries["few-shot"]["correct"] - summaries["zero-shot"]["correct"]
print("\nExtra tokens paid for few-shot:", extra_tokens)
print("Extra correct tickets          :", extra_correct)
print("Cost per extra correct ticket  :", round(extra_tokens / extra_correct, 1), "tokens")

"""Few-shot raised accuracy from 85.7% (126 of 147) to 100% (147 of 147), but it cost 22,972 tokens against 10,336, which is 156.3 tokens per classified ticket instead of 70.3. The 12,636 extra tokens bought 21 more correct tickets, about 602 tokens each. All 21 zero-shot errors were fraud or technical tickets sent to billing, and misrouting a fraud ticket is costly because it delays the security team, so the extra tokens are worth paying when errors are expensive. When wrong labels are cheap, or when the budget is tight and the labels are easy, zero-shot at roughly 45% of the cost is the better choice.

## 7. Budget, Routing and Output
"""

def load_segments(path):
    """Read the Part B segment file; keep only what routing needs."""
    segments = pd.read_csv(path)
    segments["is_premium"] = segments["is_premium"].astype(str).str.lower().isin(["true", "1"])
    return segments[["customer_id", "segment_name", "is_premium"]]

def route_ticket(status, category, is_premium):
    """Return (route, priority). The order of the rules matters."""
    if status not in CLASSIFIED:                # skipped_empty, flagged_long, flagged_invalid_output,
        return "human_review", "NORMAL"         # budget_exhausted (and any failed call): a person decides
    if category == "fraud":
        return "security_team", "HIGH"
    if is_premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"

def build_results(checked, segments):
    """Attach segment info and the route to every ticket (164 rows in, 164 rows out)."""
    merged = checked.merge(segments, on="customer_id", how="left")      # no row in the file -> blank
    assert len(merged) == len(checked), "segment file has duplicate customer_id rows"
    premium_ids = set(segments.loc[segments["is_premium"], "customer_id"])
    merged["is_premium"] = merged["customer_id"].isin(premium_ids)      # no row = NOT premium
    routes = [route_ticket(st, cat, prem)
              for st, cat, prem in zip(merged["status"], merged["category"], merged["is_premium"])]
    merged["route"] = [r for r, _ in routes]
    merged["priority"] = [p for _, p in routes]
    return merged[["ticket_id", "customer_id", "status", "category", "segment_name",
                   "priority", "route", "tokens_used"]]

os.makedirs("outputs", exist_ok=True)

# (a) zero-shot run with a token budget of 6000
TOKEN_BUDGET_DEMO = 6000
raw, stats, log = run_model_calls(tickets, ZERO_SHOT_PROMPT, token_budget=TOKEN_BUDGET_DEMO, verbose=False)
demo = apply_validation(raw)
demo_out = demo[["ticket_id", "customer_id", "status", "category", "tokens_used", "cumulative_tokens"]]
demo_out.to_csv("outputs/triage_budget_demo.csv", index=False)

exhausted = demo[demo["status"] == "budget_exhausted"]
print("\n(a) TOKEN_BUDGET =", TOKEN_BUDGET_DEMO, "\nTickets budget_exhausted:", len(exhausted))
print("Tokens used when the budget stopped the run:", int(demo["tokens_used"].sum()))
print("First exhausted ticket:", exhausted.iloc[0]["ticket_id"])
assert len(exhausted) >= 1, "lower the demo budget: no ticket was exhausted"
assert (exhausted["tokens_used"] == 0).all() and exhausted["raw_reply"].isna().all()
# proof of "no call": the mock counted exactly the calls this code made (successes + errors)
assert mock_llm._state["calls"] == stats["model_calls"]
sent = demo[~demo["status"].isin(["skipped_empty", "flagged_long", "budget_exhausted"])]
assert stats["model_calls"] == len(sent) + stats["transient_errors"]
print("\nCHECK PASSED: exhausted, empty and long tickets caused no model call")

# (b) few-shot run with a budget of 100000, routed with the Part B segments
segments = load_segments("outputs/customer_segments.csv")
raw, stats, log = run_model_calls(tickets, FEW_SHOT_PROMPT, token_budget=100000, verbose=False)
final = build_results(apply_validation(raw), segments)
final.to_csv("outputs/triage_results.csv", index=False)
assert len(final) == 164
print("\n(b) triage_results.csv rows:", len(final))
print(final["status"].value_counts().to_string())
print(final.groupby(["route", "priority"]).size().to_string())
no_segment = int((~tickets["customer_id"].isin(segments["customer_id"])).sum())
print("\nTickets whose customer has no segment row (treated as not premium):", no_segment)

# re-check the routing rules from the saved file, written a different way
saved = pd.read_csv("outputs/triage_results.csv")
premium_ids = set(segments.loc[segments["is_premium"], "customer_id"])
for _, r in saved.iterrows():
    if r["status"] not in CLASSIFIED:
        expected = ("human_review", "NORMAL")
    elif r["category"] == "fraud":
        expected = ("security_team", "HIGH")
    elif r["customer_id"] in premium_ids:
        expected = ("premium_desk", "HIGH")
    else:
        expected = ("support_queue", "NORMAL")
    assert (r["route"], r["priority"]) == expected, r["ticket_id"]
print("\nCHECK PASSED: all 164 routes agree with the routing rules")

"""Cost. The few-shot prompt costs 156.3 tokens per classified ticket against 70.3 for zero-shot (22,972 vs 10,336 tokens in total for the 156 calls), about 2.2 times as much, because its 119-token prompt is sent with every ticket. Validated by rules: 147 of 156 model replies passed automatic checks, 135 as strict JSON with an allowed category and 12 by the single-category-word rule. Sent to humans: 17 tickets in the final run (9 invalid model outputs, 4 empty tickets, 4 over-long tickets) and, in the zero-shot budget demo at 6,000 tokens, 65 tickets stopped by the budget. Fraud tickets and tickets from Premium customers are routed by rule to the security team and the premium desk with HIGH priority, and the model never overrides these rules."""
