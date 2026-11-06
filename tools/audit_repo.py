"""audit_repo.py - scan this repository against the capstone brief.

Run from the repository ROOT (the folder that holds generate_data.py):

    python tools/audit_repo.py

It only READS the repository (it changes nothing). Every line is PASS, WARN or FAIL:
    FAIL = the brief requires it and it is missing or wrong
    WARN = probably a problem, or something to double-check
This tool is a helper for you; it is not part of the graded pipeline.
"""
import ast
import csv
import glob
import json
import os
import re
import sqlite3
import subprocess
import sys

RESULTS = []
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", "node_modules"}


def record(level, name, detail=""):
    RESULTS.append((level, name, detail))


def ok(name, detail=""):
    record("PASS", name, detail)


def warn(name, detail=""):
    record("WARN", name, detail)


def fail(name, detail=""):
    record("FAIL", name, detail)


def expect(condition, name, detail_ok="", detail_bad="", level_bad="FAIL"):
    record("PASS" if condition else level_bad, name, detail_ok if condition else detail_bad)


def git(*args):
    try:
        done = subprocess.run(["git"] + list(args), capture_output=True, text=True)
        return done.returncode, done.stdout.strip()
    except FileNotFoundError:
        return 127, ""


def walk_files(extensions):
    found = []
    for folder, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(extensions):
                found.append(os.path.join(folder, name)[2:])
    return sorted(found)


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


# ------------------------------------------------------------------------------
# 1. git: branch, clean tree, pushed, tracked files
# ------------------------------------------------------------------------------
TRACKED = None
code, inside = git("rev-parse", "--is-inside-work-tree")
if code != 0:
    warn("git repository", "not a git repository (or git is missing): git checks skipped")
else:
    _, branch = git("branch", "--show-current")
    expect(branch == "main", "branch is main", "main", "current branch is '" + branch + "' - the brief grades only main")
    _, porcelain = git("status", "--porcelain")
    lines = [x for x in porcelain.split("\n") if x]
    expect(not lines, "working tree is clean (everything committed)", "",
           str(len(lines)) + " uncommitted change(s), first ones: " + "; ".join(lines[:6]), "WARN")
    code, upstream = git("rev-parse", "--abbrev-ref", "@{u}")
    if code == 0:
        _, counts = git("rev-list", "--left-right", "--count", "HEAD...@{u}")
        ahead, behind = (int(x) for x in counts.split())
        expect(ahead == 0, "no commits waiting to be pushed", "in sync with " + upstream,
               str(ahead) + " local commit(s) not pushed to " + upstream)
        expect(behind == 0, "GitHub has no newer commits", "", str(behind) + " commit(s) on GitHub you do not have", "WARN")
    else:
        warn("upstream branch", "no upstream set: cannot tell whether everything is pushed (run: git fetch origin; git status -sb)")
    _, remote = git("remote", "get-url", "origin")
    if remote:
        has_login = re.search(r"//[^/@]+@", remote) is not None        # https://user:token@github.com/...
        expect(not has_login, "remote URL contains no login or token", remote,
               "the remote URL contains a login or token: " + remote)
    _, tracked_text = git("ls-files")
    TRACKED = set(tracked_text.split("\n")) if tracked_text else set()

if not os.path.exists("generate_data.py"):
    print("Run this from the repository root (generate_data.py not found here).")
    sys.exit(1)


def is_tracked(path):
    if TRACKED is None:
        return True
    if path.endswith("/"):
        return any(t.startswith(path) for t in TRACKED)
    return path in TRACKED


# ------------------------------------------------------------------------------
# 2. required paths (they must exist AND be committed, otherwise a clone will not have them)
# ------------------------------------------------------------------------------
REQUIRED = ["generate_data.py", "mock_llm.py", "requirements.txt", ".gitignore", "README.md",
            "data/clean/customers.csv", "data/clean/products.csv", "data/clean/orders.csv", "data/instamart.db",
            "sql/queries.sql",
            "outputs/customer_segments.csv", "outputs/lapse_model.joblib", "outputs/retention_list.csv",
            "outputs/triage_results.csv", "outputs/triage_budget_demo.csv",
            "outputs/agent_runs.jsonl", "outputs/refund_ledger.jsonl", "state/"]
for path in REQUIRED:
    exists = os.path.isdir(path) if path.endswith("/") else os.path.isfile(path)
    if not exists:
        fail("required path " + path, "MISSING")
    elif not is_tracked(path):
        fail("required path " + path, "exists but is NOT committed to git (a clean clone will not have it)")
    else:
        ok("required path " + path)

# ------------------------------------------------------------------------------
# 3. .gitignore, requirements, supplied files
# ------------------------------------------------------------------------------
if os.path.exists(".gitignore"):
    ignore = read(".gitignore")
    expect(re.search(r"^\.env\s*$", ignore, re.M) is not None, ".gitignore lists .env", "", "add a line:  .env")
    expect(re.search(r"^/?\.?venv/?\s*$|^/?\.venv/?\s*$", ignore, re.M) is not None, ".gitignore lists the virtual-environment folder",
           "", "add a line:  .venv/")
    expect("__pycache__" in ignore, ".gitignore lists __pycache__/", "", "add a line:  __pycache__/")
    for needed in ("data", "outputs", "state"):
        expect(re.search(r"^/?" + needed + r"/?\s*$", ignore, re.M) is None, ".gitignore does not hide " + needed + "/",
               "", needed + "/ is ignored - the brief needs it committed")

if os.path.exists("requirements.txt"):
    reqs = [x.strip().lower() for x in read("requirements.txt").split("\n") if x.strip() and not x.startswith("#")]
    for lib in ("pandas", "numpy", "scikit-learn", "joblib"):
        expect(any(r.startswith(lib) for r in reqs), "requirements.txt contains " + lib, "", "missing " + lib)
    expect(all("==" in r for r in reqs), "requirements.txt pins exact versions", "", "some lines have no == version", "WARN")
    if len(reqs) <= 5:
        warn("requirements.txt looks hand-written",
             "the brief says to create it with pip freeze inside your virtual environment (it then lists scipy, threadpoolctl, ... too)")

for supplied, marker in (("generate_data.py", "random.Random(42)"), ("mock_llm.py", "TransientLLMError")):
    if os.path.exists(supplied):
        expect(marker in read(supplied), supplied + " has the expected content", "", "marker '" + marker + "' not found - is this the supplied file?")
        if TRACKED is not None:
            _, history = git("log", "--oneline", "--", supplied)
            n = len([x for x in history.split("\n") if x])
            expect(n <= 1, supplied + " was committed once (unmodified)", "", str(n) + " commits touch this file - the brief says use it unmodified", "WARN")

# ------------------------------------------------------------------------------
# 4. README
# ------------------------------------------------------------------------------
readmes = [p for p in glob.glob("**/README*.md", recursive=True) if not any(d in p.split(os.sep) for d in SKIP_DIRS)]
expect(len(readmes) <= 1, "a single README.md", "", "several README files found: " + ", ".join(readmes) +
       " - the brief asks for ONE README.md holding every number (move the others to docs/ with other names)", "WARN")
if os.path.exists("README.md"):
    readme = read("README.md")
    headings = [l[3:].strip() for l in readme.split("\n") if l.startswith("## ")]
    wanted = ["Overview", "Reproduce", "Part A", "Part B", "Part C", "Part D", "Limitations and Next Steps"]
    in_order = len(headings) >= 7 and all(any(h.startswith(w) for h in headings) for w in wanted)
    positions = [next((i for i, h in enumerate(headings) if h.startswith(w)), -1) for w in wanted]
    expect(in_order and positions == sorted(positions), "README has the seven sections in order",
           ", ".join(headings), "top-level (##) headings are " + str(headings) + "; need " + str(wanted))
    expect("$" not in readme, "README has no dollar sign", "", "a dollar sign appears: use INR everywhere")
    expect("![" not in readme and "<img" not in readme.lower(), "README has no images", "", "remove images")
    reproduce = readme[readme.find("## Reproduce"):] if "## Reproduce" in readme else ""
    reproduce = reproduce.split("\n## ")[0] if reproduce else ""
    numbered = [l for l in reproduce.split("\n") if re.match(r"\s*\d+\.", l)]
    expect(len(numbered) >= 6, "Reproduce has numbered commands", str(len(numbered)) + " numbered lines",
           "Reproduce needs numbered shell commands from 'python -m venv' to the last script")
    expect("python -m venv" in reproduce, "Reproduce starts from python -m venv", "", "no 'python -m venv' line")
    # every script named in Reproduce must exist
    named = re.findall(r"python\s+-m\s+([\w.]+)", reproduce) + re.findall(r"python\s+([\w/.\-]+\.py)", reproduce)
    for item in named:
        if item == "venv":
            continue
        path = item if item.endswith(".py") else item.replace(".", "/") + ".py"
        expect(os.path.exists(path), "Reproduce command points to an existing script: " + item, "", path + " does not exist")
    needles = {"ALL CHECKPOINTS PASSED": "Part A validation line", "FALLBACK_TOKEN_LIMIT": "token-limit demo status",
               "FALLBACK_TIMEOUT": "timeout demo status", "FALLBACK_MAX_ITERATIONS": "S7 status",
               "post-hoc": "Premium post-hoc limitation sentence", "get_order": "tools executed in the crash-and-resume run",
               "2,998,668.80": "revenue figure", "278": "redaction total", "135": "validation counts"}
    for needle, what in needles.items():
        expect(needle in readme, "README mentions " + what, "", "'" + needle + "' not found")
    for phrase, what in (("encoder, scaler or resampler", "no-leakage statement"),
                         ("segment labels", "segment labels not used as features")):
        expect(phrase in readme, "README has the " + what, "", "phrase '" + phrase + "' not found", "WARN")
    reproduced_scripts = set(named)
    for path in walk_files((".py",)):
        if path.startswith("src/") and path[:-3].replace("/", ".") not in reproduced_scripts and path not in reproduced_scripts:
            warn("script not in Reproduce: " + path, "add it to the numbered commands, or remove it if it is not used")

# ------------------------------------------------------------------------------
# 5. code hygiene
# ------------------------------------------------------------------------------
ALLOWED_THIRD_PARTY = {"pandas", "numpy", "sklearn", "joblib"}
NETWORK = {"requests", "httpx", "aiohttp", "openai", "anthropic", "socket", "urllib", "http", "ftplib", "smtplib"}
local_modules = {os.path.basename(p)[:-3] for p in walk_files((".py",))} | {"src"}
repo_name = os.path.basename(os.path.abspath("."))
scripts = [p for p in walk_files((".py",)) if not p.startswith("tools/")]
for path in scripts:
    text = read(path)
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        fail("script " + path + " runs", "syntax error at line " + str(error.lineno) + ": " + str(error.text).strip())
        continue
    ok("script " + path + " has valid syntax")
    colab = [i + 1 for i, l in enumerate(text.split("\n")) if re.match(r"\s*[!%]", l)]
    expect(not colab, path + " has no notebook-only lines", "", "lines starting with ! or % at: " + str(colab[:6]))
    for needle in ("google.colab", "userdata", "/content/", "display("):
        expect(needle not in text, path + " has no '" + needle + "'", "", "'" + needle + "' found - a script cannot use it")
    expect(repo_name not in text, path + " does not mention the repo folder name", "", "remove lines that name the repo folder")
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    bad = sorted(m for m in modules if m not in sys.stdlib_module_names and m not in ALLOWED_THIRD_PARTY and m not in local_modules)
    expect(not bad, path + " uses only allowed libraries", "", "not allowed by the brief: " + ", ".join(bad))
    net = sorted(m for m in modules if m in NETWORK)
    expect(not net, path + " makes no network calls", "", "network-related imports: " + ", ".join(net), "WARN")

# ------------------------------------------------------------------------------
# 6. notebooks and secrets
# ------------------------------------------------------------------------------
SECRET = re.compile(r"ghp_[A-Za-z0-9]{10,}|github_pat_[A-Za-z0-9_]{10,}|x-oauth-basic|sk-[A-Za-z0-9]{20,}")
for path in walk_files((".ipynb",)):
    try:
        nb = json.loads(read(path))
    except ValueError:
        fail("notebook " + path, "not valid JSON")
        continue
    images = sum(1 for c in nb.get("cells", []) for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    expect(images == 0, "notebook " + path + " has no image outputs", "", str(images) + " image output(s): clear outputs (the brief accepts no images)")
leaks = []
for path in walk_files((".py", ".ipynb", ".md", ".txt", ".json", ".jsonl", ".csv", ".sql", ".cfg", ".env")):
    if path.endswith("audit_repo.py"):
        continue                                   # this tool names the secret patterns it searches for
    if os.path.getsize(path) < 5_000_000 and SECRET.search(read(path)):
        leaks.append(path)
expect(not leaks, "no access token or key in any file", "", "possible secret in: " + ", ".join(leaks) + " - remove it and revoke the token")
expect(not os.path.exists(".env") or not is_tracked(".env"), ".env is not committed", "", ".env is tracked by git")

# ------------------------------------------------------------------------------
# 7. content of the output files
# ------------------------------------------------------------------------------
try:
    import pandas as pd
except ImportError:
    pd = None
    warn("pandas not installed", "output-content checks skipped")

if pd is not None:
    def load(path, **kw):
        return pd.read_csv(path, **kw) if os.path.exists(path) else None

    orders = load("data/clean/orders.csv")
    if orders is not None:
        expect(len(orders) == 12309 and orders["order_id"].is_unique, "clean orders: 12,309 unique rows", "", "got " + str(len(orders)) + " rows")
        expect(list(orders.columns) == ["order_id", "customer_id", "product_id", "quantity", "order_ts", "status",
                                        "discount_pct", "order_amount", "delivery_minutes"],
               "clean orders has the nine original columns", "", "columns are " + str(list(orders.columns)) + " (drop helper columns such as category before saving)", "WARN")
        done = orders[orders["status"] == "completed"]
        expect(len(done) == 11110 and (orders["status"] != "pending").all(), "clean orders: 11,110 completed, no pending", "", "status counts are wrong")
        expect(abs(done["order_amount"].sum() - 2998668.80) < 0.01, "completed revenue is 2,998,668.80", "", "revenue is " + str(round(done["order_amount"].sum(), 2)))
    customers = load("data/clean/customers.csv")
    if customers is not None:
        expect(customers["city"].nunique() == 6 and customers["age"].isna().sum() == 0 and customers["age"].between(18, 60).all(),
               "clean customers: 6 cities, no missing age, ages 18-60", "", "city/age cleaning is not as in the brief")
    if os.path.exists("data/instamart.db"):
        try:
            conn = sqlite3.connect("data/instamart.db")
            tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
            expect({"customers", "products", "orders"} <= tables, "database has customers, products, orders", "", "tables found: " + str(tables))
            if "orders" in tables:
                expect(conn.execute("select count(*) from orders").fetchone()[0] == 12309, "database orders has 12,309 rows", "", "row count differs")
            conn.close()
        except sqlite3.Error as error:
            fail("database opens", str(error))
    if os.path.exists("sql/queries.sql"):
        n = len([q for q in read("sql/queries.sql").split(";") if q.strip() and "select" in q.lower()])
        expect(n == 7, "sql/queries.sql holds 7 queries", "", str(n) + " queries found")

    seg = load("outputs/customer_segments.csv")
    if seg is not None:
        expect(list(seg.columns) == ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"],
               "customer_segments.csv columns", "", "columns are " + str(list(seg.columns)))
        prem = seg[seg["is_premium"].astype(str).str.lower() == "true"]
        top = seg.groupby("segment_id")["monetary"].mean().idxmax()
        expect(len(seg) == 1175 and prem["segment_id"].nunique() == 1 and prem["segment_id"].iloc[0] == top,
               "Premium flag marks exactly the highest-mean-monetary segment", "", "check is_premium")
    ret = load("outputs/retention_list.csv")
    if ret is not None:
        expect(list(ret.columns) == ["customer_id", "lapse_probability", "actually_lapsed"] and len(ret) <= 10,
               "retention_list.csv columns and at most 10 rows", "", "columns " + str(list(ret.columns)) + ", rows " + str(len(ret)))
    if os.path.exists("outputs/lapse_model.joblib"):
        try:
            import joblib
            joblib.load("outputs/lapse_model.joblib")
            ok("lapse_model.joblib loads")
        except Exception as error:      # a different scikit-learn version is the usual cause
            warn("lapse_model.joblib loads", str(error)[:150])

    tri = load("outputs/triage_results.csv", keep_default_na=False)
    if tri is not None and seg is not None:
        expect(len(tri) == 164 and list(tri.columns) == ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"],
               "triage_results.csv: 164 rows and the required columns", "", "rows " + str(len(tri)) + ", columns " + str(list(tri.columns)))
        premium_ids = set(prem["customer_id"]) if len(seg) else set()
        classified = {"classified_json", "classified_keyword"}
        wrong = 0
        for _, r in tri.iterrows():
            if r["status"] not in classified:
                want = ("human_review", "NORMAL")
            elif r["category"] == "fraud":
                want = ("security_team", "HIGH")
            elif r["customer_id"] in premium_ids:
                want = ("premium_desk", "HIGH")
            else:
                want = ("support_queue", "NORMAL")
            wrong += (r["route"], r["priority"]) != want
        expect(wrong == 0, "triage routing follows the stated rules", "", str(wrong) + " rows are routed differently from the rules")
        counts = tri["status"].value_counts().to_dict()
        expect(counts.get("classified_json") == 135 and counts.get("classified_keyword") == 12 and counts.get("flagged_invalid_output") == 9,
               "triage status counts 135 / 12 / 9", "", "counts are " + str(counts))
    demo = load("outputs/triage_budget_demo.csv", keep_default_na=False)
    if demo is not None:
        n = int((demo["status"] == "budget_exhausted").sum())
        expect(n >= 1, "budget demo has budget_exhausted tickets", str(n) + " exhausted", "no ticket is budget_exhausted: lower the demo budget")

if os.path.exists("outputs/agent_runs.jsonl"):
    runs = {}
    try:
        for line in read("outputs/agent_runs.jsonl").split("\n"):
            if line.strip():
                row = json.loads(line)
                runs[row["scenario_id"]] = row
        expect(len(runs) == 7, "agent_runs.jsonl has 7 runs", "", str(len(runs)) + " distinct scenarios")
        for sid, row in runs.items():
            expect({"scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps"} <= set(row), "agent run " + sid + " has the required keys", "", "keys: " + str(sorted(row)))
        expected = {"S1": "REFUND", "S2": "REFUND", "S3": "ESCALATE", "S4": "DENIED", "S5": "DENIED", "S6": "NOT_FOUND", "S7": "FALLBACK_MAX_ITERATIONS"}
        for sid, status in expected.items():
            expect(sid in runs and runs[sid]["status"] == status, "scenario " + sid + " ends " + status, "", "got " + str(runs.get(sid, {}).get("status")))
        expect("S1" in runs and runs["S1"]["llm_calls"] == 5, "S1 takes 5 model calls", "", "got " + str(runs.get("S1", {}).get("llm_calls")))
        expect("S7" in runs and runs["S7"]["llm_calls"] == 6 and runs["S7"]["tokens_used"] <= 6000, "S7 uses all 6 calls within 6000 tokens", "", "check S7")
    except (ValueError, KeyError) as error:
        fail("agent_runs.jsonl is valid", str(error))
if os.path.exists("outputs/refund_ledger.jsonl"):
    ledger = [json.loads(l) for l in read("outputs/refund_ledger.jsonl").split("\n") if l.strip()]
    tools = {x.get("tool") for x in ledger}
    expect({"issue_refund", "escalate_to_human"} <= tools and all("timestamp" in x for x in ledger),
           "ledger holds issued and escalated actions with timestamps", "", "tools found: " + str(tools))
    seen = {}
    for x in ledger:
        seen[(x["order_id"], x["tool"])] = seen.get((x["order_id"], x["tool"]), 0) + 1
    dup = [k for k, v in seen.items() if v > 1]
    expect(not dup, "ledger has no duplicate refund for the same order", "", "repeated: " + str(dup) + " (a demo may be writing to the real ledger)", "WARN")
if os.path.isdir("state"):
    checkpoints = glob.glob("state/*.json")
    good = 0
    for path in checkpoints:
        try:
            data = json.loads(read(path))
            good += {"run_id", "scenario_id", "messages", "tokens_used", "step_number", "tools_executed"} <= set(data)
        except ValueError:
            pass
    expect(good >= 1, "state/ holds valid checkpoints", str(good) + " valid of " + str(len(checkpoints)), "no valid checkpoint found")

# ------------------------------------------------------------------------------
# 8. which script produces what (are Parts A-D all scripts, not only notebooks?)
# ------------------------------------------------------------------------------
parts = {"Part A (clean + validate)": ("ALL CHECKPOINTS PASSED",), "Part A (SQL runner)": ("queries.sql",),
         "Part B": ("customer_segments.csv", "to_csv"), "Part C": ("triage_results",), "Part D": ("agent_runs",)}
for label, needles in parts.items():
    owners = [p for p in scripts if all(n in read(p) for n in needles)]
    expect(bool(owners), label + " has a runnable script", ", ".join(owners), "no .py file produces this Part's output: it exists only as a notebook, so the Reproduce commands cannot regenerate it")

# ------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------
order = {"FAIL": 0, "WARN": 1, "PASS": 2}
counts = {k: sum(1 for r in RESULTS if r[0] == k) for k in order}
print("=" * 78)
print("REPOSITORY AUDIT:", counts["PASS"], "passed |", counts["WARN"], "warnings |", counts["FAIL"], "failures")
print("=" * 78)
for level in ("FAIL", "WARN"):
    items = [r for r in RESULTS if r[0] == level]
    if items:
        print()
        print(level + "  (" + str(len(items)) + ")")
        for _, name, detail in items:
            print("  -", name + (": " + detail if detail else ""))
print()
print("passed checks:", counts["PASS"], "(run with  python tools/audit_repo.py --all  to list them)")
if "--all" in sys.argv:
    for level, name, detail in RESULTS:
        if level == "PASS":
            print("  PASS", name + (" - " + detail if detail else ""))
print()
print("Submit this link form (it must resolve to the main branch):")
print("  https://github.com/<user>/<repo>/tree/main")
