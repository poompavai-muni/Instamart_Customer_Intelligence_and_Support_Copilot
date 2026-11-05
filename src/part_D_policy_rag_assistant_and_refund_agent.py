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

"""## Part D - Policy RAG assistant and refund agent.

## 1: Chunk the five policy documents
**Fixed-size with overlap:** 40 words per chunk, 10 words of overlap.

"""

import os

POLICY_DIR = "data/policies"
CHUNK_SIZE = 40                     # words per chunk
OVERLAP = 10                        # words shared by two neighbouring chunks
STEP = CHUNK_SIZE - OVERLAP         # each chunk starts 30 words after the previous one


# Chunk the documents
def chunk_document(text, source, size=CHUNK_SIZE, overlap=OVERLAP):
    """Split one document into overlapping chunks of `size` words.

    Chunk 0 covers words 0-39, chunk 1 starts 30 words later, and so on. We stop after the
    first chunk whose end reaches or passes the last word of the document.
    """
    words = text.split()                         # a "word" is anything separated by whitespace
    if not words:
        return []
    step = size - overlap
    chunks = []
    start = 0
    while True:
        end = start + size
        chunks.append({
            "source": source,                    # the file name travels with every chunk
            "chunk_id": len(chunks),             # position inside its own document
            "start_word": start,
            "end_word": min(end, len(words)),    # last word position (exclusive) actually used
            "text": " ".join(words[start:end]),
        })
        if end >= len(words):                    # this chunk reaches or passes the last word: stop
            break
        start += step
    return chunks


def load_chunks(policy_dir=POLICY_DIR, overlap=OVERLAP):
    """Read every .txt file in the policy folder (alphabetical order) and chunk it."""
    all_chunks = []
    for file_name in sorted(os.listdir(policy_dir)):
        if not file_name.endswith(".txt"):
            continue
        with open(os.path.join(policy_dir, file_name), encoding="utf-8") as f:
            text = f.read()
        all_chunks.extend(chunk_document(text, file_name, overlap=overlap))
    return all_chunks

# Chunk the five policy documents and check the result
print("===== TASK 1: CHUNKING =====")
chunks = load_chunks()

counts = {}
for chunk in chunks:
    counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1

print("Total chunks:", len(chunks))
for source, n in counts.items():
    words_in_file = len(open(os.path.join(POLICY_DIR, source), encoding="utf-8").read().split())
    print(f"  {source}: {n} chunks  ({words_in_file} words)")

assert len(chunks) == 15
assert counts == {"delivery_policy.txt": 3, "escalation_policy.txt": 3,
                  "payment_policy.txt": 3, "privacy_policy.txt": 2, "refund_policy.txt": 4}
print("CHECK PASSED: 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2)")
# structural checks: size, overlap, coverage and source name
for source in counts:
    doc = [c for c in chunks if c["source"] == source]
    original = open(os.path.join(POLICY_DIR, source), encoding="utf-8").read().split()
    for c in doc:
        assert len(c["text"].split()) <= CHUNK_SIZE, "chunk longer than 40 words"
        assert c["source"] == source
    for first, second in zip(doc, doc[1:]):
        assert second["start_word"] - first["start_word"] == STEP, "chunks must start 30 words apart"
        assert first["text"].split()[-OVERLAP:] == second["text"].split()[:OVERLAP], "10-word overlap broken"
    # coverage: the first chunk plus the NEW words of every later chunk rebuild the document
    rebuilt = doc[0]["text"].split()
    for c in doc[1:]:
        rebuilt += c["text"].split()[OVERLAP:]
    assert rebuilt == original, "some words are missing from the chunks"
print("\nCHECK PASSED: every chunk <= 40 words, neighbours share 10 words, no word is lost, source recorded")

refund = [c for c in chunks if c["source"] == "refund_policy.txt"]
print("\nRefund_policy.txt chunk 0 (words", refund[0]["start_word"], "to", refund[0]["end_word"] - 1, "):")
print("  ", refund[0]["text"])
print("\nRefund_policy.txt chunk 1 (words", refund[1]["start_word"], "to", refund[1]["end_word"] - 1, "):")
print("  ", refund[1]["text"])
print("\nShared words:", " ".join(refund[0]["text"].split()[-OVERLAP:]))

"""## 2. Vectorize and Retrieve"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_INDEX = {}          # holds the chunks, the fitted vectorizer and the chunk vectors once built

def build_index(overlap=OVERLAP):
    """Chunk the documents and fit TF-IDF on the chunk texts. Calling it again replaces the index."""
    chunks = load_chunks(overlap=overlap)
    vectorizer = TfidfVectorizer()
    chunk_matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    _INDEX["chunks"] = chunks
    _INDEX["vectorizer"] = vectorizer
    _INDEX["matrix"] = chunk_matrix
    return chunks, vectorizer, chunk_matrix

def retrieve(query, k=3):
    """Return the k chunks most similar to the query as [{"source", "score", "text"}, ...]."""
    if not _INDEX:
        build_index()
    query_vector = _INDEX["vectorizer"].transform([query])           # SAME fitted vectorizer, no refitting
    scores = cosine_similarity(query_vector, _INDEX["matrix"])[0]    # one score per chunk
    order = np.argsort(-scores, kind="stable")                       # best first; ties keep chunk order
    results = []
    for i in order[:k]:
        results.append({
            "source": _INDEX["chunks"][i]["source"],
            "score": float(scores[i]),
            "text": _INDEX["chunks"][i]["text"],
        })
    return results

chunks, vectorizer, chunk_matrix = build_index()
print("Chunk matrix shape (chunks x vocabulary words):", chunk_matrix.shape)

query = "refund policy late delivery"
results = retrieve(query, k=3)
print("Query:", query)
for rank, r in enumerate(results, start=1):
    print(f"  {rank}. {r['source']}  score={r['score']:.4f}")
assert len(results) == 3
assert all(set(r) == {"source", "score", "text"} for r in results)
assert all(r["source"] == "refund_policy.txt" for r in results), "top 3 must all be refund_policy.txt"
assert results[0]["score"] >= results[1]["score"] >= results[2]["score"]
print("CHECK PASSED: the top three chunks all come from refund_policy.txt, best first")

"""**TF (term frequency)** measures how often a word occurs inside one chunk, so a chunk that repeats a word is treated as more about it.

 **IDF (inverse document frequency)** measures how rare a word is across all the chunks, so distinctive words such as “refund” count for more than words that appear everywhere, such as “the”.

**Limitation:** TF-IDF only matches exact words, so it misses paraphrases and different word forms. In my tests the query “how do I escalate a complaint to a manager” ranked the refund policy first, because the word “escalate” is not in the vocabulary (the policy only says “escalation”). Likewise “money back because my groceries arrived slowly” retrieved the escalation policy, because the single shared word was “back”, from “call the customer back”, while the refund policy never says “money back”.

## 3. Augment and answer
"""

import mock_llm

POLICY_SYSTEM_PROMPT = (
    "You are a POLICY ASSISTANT for a quick-commerce company. "
    "Answer the question using only the information in the CONTEXT. "
    "If the CONTEXT does not contain the answer, say that you could not find the information."
)

def build_policy_messages(question, results):
    """Two messages: the instructions (system) and CONTEXT + QUESTION (user)."""
    context = "\n".join("[" + r["source"] + "]\n" + r["text"] for r in results)   # [file] on its own line
    user_text = "CONTEXT:\n" + context + "\nQUESTION:\n" + question
    return [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

def answer(question):
    """Retrieve three chunks, ask the model, and return its text and the sources used."""
    results = retrieve(question, k=3)
    messages = build_policy_messages(question, results)
    response = mock_llm.chat(messages)
    sources = []
    for r in results:                            # keep the best-ranked order, no repeats
        if r["source"] not in sources:
            sources.append(r["source"])
    return {"answer": response.choices[0].message.content, "sources": sources}

# check the message format
question = "What delivery time makes a completed order eligible for a full refund?"
results = retrieve(question, k=3)
messages = build_policy_messages(question, results)

assert len(messages) == 2 and [m["role"] for m in messages] == ["system", "user"]
assert "POLICY ASSISTANT" in messages[0]["content"]
assert "only" in messages[0]["content"] and "could not find" in messages[0]["content"]
user_text = messages[1]["content"]
assert user_text.startswith("CONTEXT:\n")
assert user_text.endswith("\nQUESTION:\n" + question)
tag_lines = [line for line in user_text.split("\n") if line.startswith("[") and line.endswith(".txt]")]
assert tag_lines == ["[" + r["source"] + "]" for r in results], "each chunk needs its [file] line"
assert len(tag_lines) == 3
print("CHECK PASSED: 2 messages, POLICY ASSISTANT system prompt, CONTEXT/QUESTION user message")
print("\nUser message sent to the model:")
print(user_text)

reply = answer(question)
print("\nAnswer :", reply["answer"])
print("Sources:", reply["sources"])
assert "more than 45 minutes" in reply["answer"] and "refund_policy.txt" in reply["sources"]

outside = answer("What is the capital of France?")
print("\nQuestion outside the policies ->", outside["answer"])
assert outside["answer"].startswith("I could not find")
print("\nCHECK PASSED: grounded answer for a covered question, fallback for an uncovered one")

"""## 4. Evaluate retrieval and grounding"""

import csv
import re

EVAL_PATH = "data/raw/policy_eval.csv"
FALLBACK_START = "I could not find"

def load_eval_questions(path=EVAL_PATH):
    with open(path, encoding="utf-8", newline="") as f:      # newline="" lets csv handle the file's line endings
        return list(csv.DictReader(f))

def evaluate(rows):
    """Run every question through answer() and count the three kinds of success."""
    answerable = [r for r in rows if r["expected_doc"] != "NONE"]
    unanswerable = [r for r in rows if r["expected_doc"] == "NONE"]
    details = []
    for r in answerable:
        out = answer(r["question"])
        details.append({
            "question": r["question"], "expected_doc": r["expected_doc"], "expected_fact": r["expected_fact"],
            "doc_ok": r["expected_doc"] in out["sources"],              # expected document among the top 3
            "fact_ok": r["expected_fact"] in out["answer"],             # case-sensitive substring
            "answer": out["answer"], "sources": out["sources"],
        })
    fallback_ok = 0
    for r in unanswerable:
        out = answer(r["question"])
        if out["answer"].startswith(FALLBACK_START):
            fallback_ok += 1
    summary = {
        "answerable": len(answerable),
        "doc_in_top3": sum(d["doc_ok"] for d in details),
        "fact_in_answer": sum(d["fact_ok"] for d in details),
        "unanswerable": len(unanswerable),
        "fallback": fallback_ok,
    }
    return summary, details

def run_evaluation(overlap):
    """Rebuild the index with the given overlap, evaluate, then restore the normal index."""
    build_index(overlap=overlap)
    n_chunks = len(_INDEX["chunks"])
    summary, details = evaluate(load_eval_questions())
    summary["chunks"] = n_chunks
    used_chunks = list(_INDEX["chunks"])                  # keep the chunks used, to explain failures
    build_index()                                         # back to overlap 10 for everything else
    return summary, details, used_chunks

def explain_failure(detail, chunks, policy_dir=POLICY_DIR):
    """Say WHY a question failed: is the sentence that holds the fact cut by a chunk boundary?"""
    fact = detail["expected_fact"]
    with open(os.path.join(policy_dir, detail["expected_doc"]), encoding="utf-8") as f:
        doc_words = f.read().split()
    sentences = [x.strip() for x in re.split(r"(?<=[.])\s+", " ".join(doc_words)) if x.strip()]
    holders = [x for x in sentences if fact in x]
    if not holders:
        return "the expected fact is not in the document"
    sentence_words = holders[0].split()
    # where does this sentence sit in the document? (word positions)
    first = next(i for i in range(len(doc_words)) if doc_words[i:i + len(sentence_words)] == sentence_words)
    last = first + len(sentence_words) - 1
    doc_chunks = [c for c in chunks if c["source"] == detail["expected_doc"]]
    if any(c["start_word"] <= first and last < c["end_word"] for c in doc_chunks):
        return "the whole sentence is inside one chunk, so this is not a boundary problem"
    start_chunk = next(c["chunk_id"] for c in doc_chunks if c["start_word"] <= first < c["end_word"])
    end_chunk = next(c["chunk_id"] for c in doc_chunks if c["start_word"] <= last < c["end_word"])
    return ("the sentence holding the fact (words " + str(first) + " to " + str(last) + " of "
            + detail["expected_doc"] + ") is cut by a chunk boundary: it starts in chunk " + str(start_chunk)
            + " and ends in chunk " + str(end_chunk) + ", so no single chunk contains it whole. Sentence: \""
            + " ".join(sentence_words) + "\"")

summary10, details10, chunks10 = run_evaluation(overlap=10)
summary0, details0, chunks0 = run_evaluation(overlap=0)

print("| overlap | chunks | answerable: expected doc in top 3 | answerable: fact in answer | unanswerable: fallback |")
for name, sm in [(10, summary10), (0, summary0)]:
    print(f"| {name:>7} | {sm['chunks']:>6} | {sm['doc_in_top3']}/{sm['answerable']:<31} | "
          f"{sm['fact_in_answer']}/{sm['answerable']:<24} | {sm['fallback']}/{sm['unanswerable']:<21} |")

assert (summary10["doc_in_top3"], summary10["fact_in_answer"], summary10["fallback"]) == (12, 12, 2)
print("CHECK PASSED (overlap 10): 12/12, 12/12 and 2/2")

broken = [d for d in details0 if not (d["doc_ok"] and d["fact_ok"])]
assert len(broken) >= 1, "with overlap 0 at least one answerable question should fail"
print("CHECK PASSED (overlap 0): at least one answerable question fails")
for d in broken:
    print("\nBROKEN with overlap 0:", d["question"])
    print("  expected fact:", repr(d["expected_fact"]), "| fact in answer:", d["fact_ok"])
    print("  answer       :", d["answer"])
    print("  why          :", explain_failure(d, chunks0))

print("\nIndex restored to overlap", OVERLAP, "->", len(_INDEX["chunks"]), "chunks")
assert len(_INDEX["chunks"]) == 15

"""## 5. Define tools and the tool map"""

import json
import os
import sqlite3
import time

DB_PATH = "data/instamart.db"
LEDGER_PATH = "outputs/refund_ledger.jsonl"
LATE_MINUTES = 45                  # delivered in MORE than 45 minutes -> full refund
ESCALATION_LIMIT_INR = 1000        # a refund ABOVE this goes to a human

def get_order(order_id):
    """Read one order from the Part A SQLite database."""
    connection = sqlite3.connect(DB_PATH)
    try:
        row = connection.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
            (order_id,),                                  # ? placeholder: the id is never pasted into the SQL text
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}

def search_policy(query):
    """Search the policy documents with the retrieval built in tasks 1-2."""
    return {"results": retrieve(query)}

def calculate_refund(order_amount, delivery_minutes):
    """Refund rule: more than 45 minutes -> 100% of the order amount, otherwise 0."""
    if delivery_minutes is None or delivery_minutes <= LATE_MINUTES:    # no delivery time = never delivered
        return {"refund_inr": 0.0}
    return {"refund_inr": round(float(order_amount), 2)}

def _write_ledger(tool_name, order_id, refund_inr):
    """Append ONE JSON line (with a timestamp and the tool name) to the ledger file."""
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    line = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tool": tool_name, "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")

def issue_refund(order_id, refund_inr):
    """Record an automatic refund. Safety guard: refunds above INR 1000 are never issued automatically."""
    if refund_inr > ESCALATION_LIMIT_INR:
        return {"error": "refund above INR 1000 must be escalated to a human"}
    _write_ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}

def escalate_to_human(order_id, refund_inr):
    """Record that a human must decide this refund."""
    _write_ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}

def _params(properties, required):
    return {"type": "object", "properties": properties, "required": required}

TOOLS_SPEC = [
    {"name": "get_order",
     "description": "Look up one order in the database by its order id.",
     "parameters": _params({"order_id": {"type": "string", "description": "Order id such as ORD-100048"}},
                           ["order_id"])},
    {"name": "search_policy",
     "description": "Search the company policy documents and return the most relevant passages.",
     "parameters": _params({"query": {"type": "string", "description": "What to look up in the policies"}},
                           ["query"])},
    {"name": "calculate_refund",
     "description": "Compute the refund in INR for an order from its amount and delivery time.",
     "parameters": _params({"order_amount": {"type": "number", "description": "Order amount in INR"},
                            "delivery_minutes": {"type": "number", "description": "Minutes the delivery took"}},
                           ["order_amount", "delivery_minutes"])},
    {"name": "issue_refund",
     "description": "Issue an automatic refund and record it in the refund ledger.",
     "parameters": _params({"order_id": {"type": "string", "description": "Order id to refund"},
                            "refund_inr": {"type": "number", "description": "Refund amount in INR"}},
                           ["order_id", "refund_inr"])},
    {"name": "escalate_to_human",
     "description": "Send a refund above the automatic limit to a human lead and record it in the ledger.",
     "parameters": _params({"order_id": {"type": "string", "description": "Order id to escalate"},
                            "refund_inr": {"type": "number", "description": "Refund amount in INR"}},
                           ["order_id", "refund_inr"])},
]

TOOL_MAP = {                      # tool name -> Python function
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}

def run_tool(name, arguments):
    """Look the name up in TOOL_MAP and call the function with the arguments dictionary."""
    if name not in TOOL_MAP:
        return {"error": "unknown tool: " + name}
    return TOOL_MAP[name](**arguments)

# Check everything
names = [t["name"] for t in TOOLS_SPEC]
print("\nTools in TOOLS_SPEC:", names)
assert len(TOOLS_SPEC) == 5 and set(names) == set(TOOL_MAP), "TOOLS_SPEC and TOOL_MAP must list the same 5 tools"
for tool in TOOLS_SPEC:
    assert set(tool) == {"name", "description", "parameters"}
    assert tool["description"].count(".") == 1 and tool["description"].endswith("."), "one sentence"
    assert tool["parameters"]["type"] == "object"
    fn = TOOL_MAP[tool["name"]]
    argument_names = list(fn.__code__.co_varnames[:fn.__code__.co_argcount])
    assert argument_names == list(tool["parameters"]["properties"]) == tool["parameters"]["required"], tool["name"]
print("\nCHECK PASSED: 5 tools, each with name, one-sentence description and a spec that matches the function's arguments")

for order_id in ["ORD-100048", "ORD-100067", "ORD-100145", "ORD-100001", "ORD-100025", "ORD-999999"]:
    print("  get_order", order_id, "->", run_tool("get_order", {"order_id": order_id}))
assert run_tool("get_order", {"order_id": "ORD-999999"}) == {"error": "order not found"}
assert run_tool("get_order", {"order_id": "ORD-100048"}) == {
    "order_id": "ORD-100048", "status": "completed", "order_amount": 128.25, "delivery_minutes": 65.0}

found = run_tool("search_policy", {"query": "refund policy late delivery"})
assert list(found) == ["results"] and len(found["results"]) == 3
assert all(r["source"] == "refund_policy.txt" for r in found["results"])

assert run_tool("calculate_refund", {"order_amount": 128.25, "delivery_minutes": 65}) == {"refund_inr": 128.25}
assert run_tool("calculate_refund", {"order_amount": 1999.0, "delivery_minutes": 54}) == {"refund_inr": 1999.0}
assert run_tool("calculate_refund", {"order_amount": 109.65, "delivery_minutes": 20}) == {"refund_inr": 0.0}
assert run_tool("calculate_refund", {"order_amount": 80.0, "delivery_minutes": 45}) == {"refund_inr": 0.0}
assert run_tool("calculate_refund", {"order_amount": 80.0, "delivery_minutes": 45.5}) == {"refund_inr": 80.0}
assert run_tool("calculate_refund", {"order_amount": 45.0, "delivery_minutes": None}) == {"refund_inr": 0.0}
print("\nCHECK PASSED: get_order, search_policy and calculate_refund behave as specified")

# the ledger tools: test with a temporary ledger so the real one stays clean
real_ledger = LEDGER_PATH
LEDGER_PATH = "outputs/test_ledger.jsonl"
if os.path.exists(LEDGER_PATH):
    os.remove(LEDGER_PATH)
assert run_tool("issue_refund", {"order_id": "ORD-100048", "refund_inr": 128.25}) == {"refund_inr": 128.25}
assert run_tool("escalate_to_human", {"order_id": "ORD-100145", "refund_inr": 1999.0}) == {"refund_inr": 1999.0}
assert "error" in run_tool("issue_refund", {"order_id": "ORD-100145", "refund_inr": 1999.0})   # guard
with open(LEDGER_PATH, encoding="utf-8") as f:
    ledger_lines = [json.loads(line) for line in f]
print("  ledger lines written:", len(ledger_lines))
for line in ledger_lines:
    print("   ", line)
assert len(ledger_lines) == 2, "one line per executed tool, none for the refused call"
assert [l["tool"] for l in ledger_lines] == ["issue_refund", "escalate_to_human"]
assert all("timestamp" in l for l in ledger_lines)
os.remove(LEDGER_PATH)
LEDGER_PATH = real_ledger
assert run_tool("no_such_tool", {}) == {"error": "unknown tool: no_such_tool"}
print("\nCHECK PASSED: ledger tools append one JSON line each (timestamp + tool name); a refund above 1000 is refused")

"""## 6. The ReAct loop"""

import csv
import json
import os
import re
import sqlite3
import time
import mock_llm

SCENARIOS_PATH = "data/raw/agent_scenarios.csv"
RUNS_PATH = "outputs/agent_runs.jsonl"

AGENT_SYSTEM_PROMPT = (
    "You are a refund AGENT for a quick-commerce company. "
    "Use the tools to look up the order, check the policy and calculate the refund, "
    "then issue the refund, escalate it to a human, or deny it. "
    "Finish with one line that starts with REFUND:, ESCALATE:, DENIED: or NOT_FOUND:."
)

STATE_DIR = "state"
CHECKPOINT_KEYS = ["run_id", "scenario_id", "user_message", "messages", "tokens_used", "llm_calls",
                   "step_number", "tools_executed", "steps", "elapsed_seconds",
                   "max_iterations", "token_limit", "time_limit_seconds", "finished", "status", "final_answer"]


class SimulatedCrash(Exception):
    """Raised on purpose, to test that a run can be resumed."""


def handoff_summary(status, reason, steps, user_message):
    """A short note for the human who takes over: why we stopped and what was already done."""
    counts = {}
    for step in steps:
        counts[step["tool"]] = counts.get(step["tool"], 0) + 1
    done = ", ".join(name + (" x" + str(n) if n > 1 else "") for name, n in counts.items()) or "none"
    order = re.search(r"ORD-\d{6}", user_message)
    order_note = " (order " + order.group(0) + ")" if order else ""
    return (status + ": " + reason + ". Steps completed: " + done +
            ". A human should take over this request" + order_note + ".")


def checkpoint_path(run_id, state_dir=STATE_DIR):
    return os.path.join(state_dir, run_id + ".json")


def save_checkpoint(state, state_dir=STATE_DIR):
    """Write the run's state to state/<run_id>.json (write a temp file first, then swap it in)."""
    os.makedirs(state_dir, exist_ok=True)
    state["elapsed_seconds"] = time.monotonic() - state["clock_start"]
    data = {key: state[key] for key in CHECKPOINT_KEYS}
    path = checkpoint_path(state["run_id"], state_dir)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(path + ".tmp", path)             # a crash while writing can never leave a half-written checkpoint


def _execute_call(state, call, save, state_dir, crash_after_step):
    """Run ONE requested tool, record it, and save a checkpoint straight after the observation."""
    try:
        result = run_tool(call["name"], call["arguments"])
    except (TypeError, KeyError, ValueError, sqlite3.Error) as error:       # the model's arguments can be wrong
        result = {"error": "tool failed: " + str(error)}
    state["steps"].append({"tool": call["name"], "arguments": call["arguments"], "observation": result})
    state["messages"].append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                              "content": json.dumps(result)})
    state["step_number"] += 1
    state["tools_executed"].append(call["name"])
    if save:
        save_checkpoint(state, state_dir)                                    # saved BEFORE any crash can happen
    if crash_after_step is not None and state["step_number"] == crash_after_step:
        raise SimulatedCrash("simulated crash right after tool execution number " + str(crash_after_step))


def _pending_tool_calls(messages):
    """Tool calls the model requested whose result is not in the messages yet (only after a crash)."""
    for message in reversed(messages):
        if message["role"] == "assistant" and message.get("tool_calls"):
            answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
            return [c for c in message["tool_calls"] if c["id"] not in answered]
    return []


def _result(state):
    return {"status": state["status"], "final_answer": state["final_answer"], "steps": state["steps"],
            "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"]}


def _run_loop(state, save, state_dir, crash_after_step):
    """The think -> act -> observe loop, working on a state dictionary that can be saved and restored."""
    state["clock_start"] = time.monotonic() - state["elapsed_seconds"]     # time spent before a crash still counts
    final_answer = None
    stop_status = None              # set when a safety limit ends the run
    stop_reason = None

    for call in _pending_tool_calls(state["messages"]):                    # finish what a crashed run had started
        _execute_call(state, call, save, state_dir, crash_after_step)

    while state["llm_calls"] < state["max_iterations"]:
        # (c) time limit, checked at the TOP of every iteration
        elapsed = time.monotonic() - state["clock_start"]
        if elapsed >= state["time_limit_seconds"]:
            stop_status = "FALLBACK_TIMEOUT"
            stop_reason = ("ran for " + format(elapsed, ".2f") + " seconds, at or above the limit of "
                           + str(state["time_limit_seconds"]) + " seconds")
            break

        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)       # THINK: ask the model
        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens
        reply = response.choices[0].message

        if not reply.tool_calls:                                            # no tool asked for: final answer
            final_answer = reply.content
            break

        # (b) token limit, checked after every model call and BEFORE the requested tool is executed
        if state["tokens_used"] > state["token_limit"]:
            stop_status = "FALLBACK_TOKEN_LIMIT"
            stop_reason = ("used " + str(state["tokens_used"]) + " tokens, above the limit of "
                           + str(state["token_limit"]) + "; the tool just requested ("
                           + reply.tool_calls[0].name + ") was not executed")
            break

        # ACT: remember what the model asked for, then run every requested tool
        calls = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in reply.tool_calls]
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})
        for call in calls:
            _execute_call(state, call, save, state_dir, crash_after_step)   # OBSERVE happens inside

    if final_answer is None:        # no final answer: one of the three limits ended the run
        if stop_status is None:     # (a) the loop ran out of iterations
            stop_status = "FALLBACK_MAX_ITERATIONS"
            stop_reason = "stopped after " + str(state["llm_calls"]) + " model calls without a final answer"
        final_answer = handoff_summary(stop_status, stop_reason, state["steps"], state["user_message"])
        status = stop_status
    else:
        status = final_answer.split(":")[0].strip() if ":" in final_answer else "UNKNOWN"

    state["finished"] = True
    state["status"] = status
    state["final_answer"] = final_answer
    if save:
        save_checkpoint(state, state_dir)                                   # final checkpoint: marked finished
    return _result(state)


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_step=None, save_checkpoints=True, state_dir=STATE_DIR):
    """Start a new run. Returns status, final_answer, steps, llm_calls and tokens_used.

    run_id: name of the checkpoint file (a unique one is made if you give none).
    crash_after_step: test hook - raise SimulatedCrash after that many tool executions.
    """
    if run_id is None:
        run_id = scenario_id + "_" + str(int(time.time() * 1000))
    state = {
        "run_id": run_id, "scenario_id": scenario_id, "user_message": user_message,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},            # must contain the word AGENT
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0, "llm_calls": 0, "step_number": 0, "tools_executed": [], "steps": [],
        "elapsed_seconds": 0.0, "max_iterations": max_iterations, "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds, "finished": False, "status": None, "final_answer": None,
    }
    return _run_loop(state, save_checkpoints, state_dir, crash_after_step)


def resume_agent(run_id, state_dir=STATE_DIR, save_checkpoints=True):
    """Load state/<run_id>.json and continue the run from where it stopped."""
    with open(checkpoint_path(run_id, state_dir), encoding="utf-8") as f:
        state = json.load(f)
    if state["finished"]:                       # already complete: return the saved result, run nothing again
        return _result(state)
    return _run_loop(state, save_checkpoints, state_dir, None)


def refund_from_steps(steps):
    """The refund amount that was issued or escalated (0.0 when nothing was)."""
    amount = 0.0
    for step in steps:
        if step["tool"] in ("issue_refund", "escalate_to_human") and "refund_inr" in step["observation"]:
            amount = step["observation"]["refund_inr"]
    return amount


def run_all_scenarios(scenarios_path=SCENARIOS_PATH, runs_path=RUNS_PATH):
    """Run every scenario and append one JSON line per run to the runs file (starts a fresh file)."""
    os.makedirs(os.path.dirname(runs_path), exist_ok=True)
    for path in (runs_path, LEDGER_PATH):          # a fresh batch: no lines left over from an earlier run
        if os.path.exists(path):
            os.remove(path)
    with open(scenarios_path, encoding="utf-8", newline="") as f:
        scenarios = list(csv.DictReader(f))
    runs = []
    for row in scenarios:
        result = run_agent(row["message"], row["scenario_id"], run_id="run_" + row["scenario_id"])
        record = {"scenario_id": row["scenario_id"], "status": result["status"],
                  "final_answer": result["final_answer"], "llm_calls": result["llm_calls"],
                  "tokens_used": result["tokens_used"], "refund_inr": refund_from_steps(result["steps"]),
                  "steps": result["steps"]}
        with open(runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        runs.append((row, record))
    return runs

# Run the seven scenarios and check
runs = run_all_scenarios()

print("| scenario | status                   | refund INR | model calls | tokens used |")
for row, record in runs:
    print(f"| {record['scenario_id']:>8} | {record['status']:<24} | {record['refund_inr']:>10.2f} | "
          f"{record['llm_calls']:>11} | {record['tokens_used']:>11} |")

for row, record in runs:
    expected = row["expected_outcome"]
    if expected == "FALLBACK":
        assert record["status"].startswith("FALLBACK"), row["scenario_id"]
    else:
        assert record["status"] == expected, row["scenario_id"]
    assert abs(record["refund_inr"] - float(row["expected_refund_inr"])) <= 0.01, row["scenario_id"]
by_id = {record["scenario_id"]: record for _, record in runs}
assert by_id["S1"]["llm_calls"] == 5
assert by_id["S7"]["llm_calls"] == 6 and by_id["S7"]["tokens_used"] < 6000
print("\nCHECK PASSED: Every status and refund amount matches agent_scenarios.csv;\n    S1 took 5 model calls; ")
print(f"    S7 used all 6 calls with {by_id["S7"]["tokens_used"]}-token")

with open(RUNS_PATH, encoding="utf-8") as f:
    lines = [json.loads(x) for x in f]
assert len(lines) == 7
for line in lines:
    assert {"scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps"} <= set(line)
    for step in line["steps"]:
        assert set(step) == {"tool", "arguments", "observation"}
with open(LEDGER_PATH, encoding="utf-8") as f:
    ledger = [json.loads(x) for x in f]
print("\nagent_runs.jsonl lines:", len(lines), "| ledger lines:", len(ledger))
assert [l["tool"] for l in ledger] == ["issue_refund", "issue_refund", "escalate_to_human"]
print("\nCHECK PASSED: 7 run records with every step, and the ledger holds exactly the 3 refund actions")

"""## 7. Stop safely"""

s1_message = "My order ORD-100048 reached me very late. I would like a refund."
real_ledger = LEDGER_PATH
LEDGER_PATH = "outputs/test_ledger.jsonl"           # the demos must not touch the real ledger
if os.path.exists(LEDGER_PATH):
    os.remove(LEDGER_PATH)

# (a) iteration limit: S7
s7 = run_agent("[LOOP] Refund order ORD-100048.", "S7")
print("(a) S7                       ->", s7["status"], "| model calls:", s7["llm_calls"], "| tokens:", s7["tokens_used"])
assert s7["status"] == "FALLBACK_MAX_ITERATIONS" and s7["llm_calls"] == 6 and s7["tokens_used"] <= 6000

# (b) token limit: S1 with token_limit=200
low = run_agent(s1_message, "S1", token_limit=200)
print("(b) S1, token_limit=200      ->", low["status"], "| model calls:", low["llm_calls"],
      "| tokens:", low["tokens_used"], "| steps:", [x["tool" ] for x in low["steps"]])
assert low["status"] == "FALLBACK_TOKEN_LIMIT" and low["tokens_used"] > 200
assert "search_policy" not in [x["tool"] for x in low["steps"]], "the requested tool must NOT run"
assert not os.path.exists(LEDGER_PATH), "no refund may be written"

# (c) time limit: S1 with time_limit_seconds=0 stops before the first model call
timeout = run_agent(s1_message, "S1", time_limit_seconds=0)
print("(c) S1, time_limit_seconds=0 ->", timeout["status"], "| model calls:", timeout["llm_calls"],
      "| tokens:", timeout["tokens_used"], "| steps:", timeout["steps"])
assert timeout["status"] == "FALLBACK_TIMEOUT" and timeout["llm_calls"] == 0 and timeout["steps"] == []

# (c) again, with a fake clock, to test a timeout in the MIDDLE of a run
ticks = iter([0.0, 0.2, 0.9, 1.4, 2.0, 3.0, 4.0])      # start, then one reading per iteration
real_monotonic = time.monotonic
time.monotonic = lambda: next(ticks)
middle = run_agent(s1_message, "S1", time_limit_seconds=1.0)
time.monotonic = real_monotonic
print("(c) S1, fake clock, limit 1s ->", middle["status"], "| model calls:", middle["llm_calls"],
      "| steps:", [x["tool"] for x in middle["steps"]])
assert middle["status"] == "FALLBACK_TIMEOUT" and middle["llm_calls"] in (1,2)

for result in [s7, low, timeout, middle]:
    assert result["final_answer"].startswith(result["status"] + ":")
    assert "Steps completed:" in result["final_answer"] and "A human should take over" in result["final_answer"]
print()
for name, result in [("S7", s7), ("token limit", low), ("timeout", timeout)]:
    print(name, "->", result["final_answer"])
for scenario_id, message, expected in [("S1", s1_message, "REFUND"),
                                       ("S6", "Refund for order ORD-999999 please.", "NOT_FOUND")]:
    assert run_agent(message, scenario_id)["status"] == expected      # the defaults do not stop normal runs
os.remove(LEDGER_PATH)
LEDGER_PATH = real_ledger
print("CHECK PASSED: Iteration, token and time limits each end a run with their own status, a handoff summary,")
print("              No tool executed after the token limit, and normal runs are not stopped by the defaults")

"""## 8. Make the run resumable"""

# The crash and resume demo
s1_message = "My order ORD-100048 reached me very late. I would like a refund."
expected_tools = ["get_order", "search_policy", "calculate_refund", "issue_refund"]
real_ledger = LEDGER_PATH

def read_ledger(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f]

# an uninterrupted S1 for comparison (its ledger is temporary)
LEDGER_PATH = "outputs/test_ledger_clean.jsonl"
if os.path.exists(LEDGER_PATH):
    os.remove(LEDGER_PATH)
clean = run_agent(s1_message, "S1", save_checkpoints=False)
os.remove(LEDGER_PATH)

# the demo: crash after the SECOND tool execution (after its checkpoint is saved)
RUN_ID = "demo_resume_S1"
LEDGER_PATH = "outputs/resume_demo_ledger.jsonl"
for path in (LEDGER_PATH, checkpoint_path(RUN_ID)):
    if os.path.exists(path):
        os.remove(path)
try:
    run_agent(s1_message, "S1", run_id=RUN_ID, crash_after_step=2)
    raise AssertionError("the simulated crash did not happen")
except SimulatedCrash as crash:
    print("1) crash:", crash)

with open(checkpoint_path(RUN_ID), encoding="utf-8") as f:
    checkpoint = json.load(f)
for key in ["run_id", "scenario_id", "messages", "tokens_used", "step_number", "tools_executed"]:
    assert key in checkpoint, key
print("\n2) checkpoint", checkpoint_path(RUN_ID), "| step_number:", checkpoint["step_number"],
      "| tokens_used:", checkpoint["tokens_used"], "| messages:", len(checkpoint["messages"]))
print("   tools executed before the crash:", checkpoint["tools_executed"])
assert checkpoint["step_number"] == 2 and checkpoint["tools_executed"] == ["get_order", "search_policy"]
assert read_ledger(LEDGER_PATH) == [], "no refund may exist before the crash"

result = resume_agent(RUN_ID)
after_resume = [step["tool"] for step in result["steps"]][checkpoint["step_number"]:]
print("\n3) resumed ->", result["status"], "| refund:", refund_from_steps(result["steps"]))
print("   tools executed after the resume :", after_resume)
executed = [step["tool"] for step in result["steps"]]
print("\nTools executed across both parts:", executed)
assert executed == expected_tools, "four executions, none repeated"
assert result["status"] == "REFUND" and refund_from_steps(result["steps"]) == 128.25
assert result["llm_calls"] == clean["llm_calls"] == 5 and result["tokens_used"] == clean["tokens_used"]
ledger = read_ledger(LEDGER_PATH)
assert len(ledger) == 1 and ledger[0]["tool"] == "issue_refund" and ledger[0]["refund_inr"] == 128.25
print("   ledger lines:", len(ledger), "| model calls:", result["llm_calls"], "| tokens:", result["tokens_used"])

again = resume_agent(RUN_ID)                         # resuming a FINISHED run must run nothing again
assert again == result and len(read_ledger(LEDGER_PATH)) == 1
print("\n4) resuming the finished run again returns the saved result and repeats nothing")
print("\nCHECK PASSED: tools across crash and resume = get_order, search_policy, calculate_refund, issue_refund; REFUND 128.25")

# stronger test: crash after EVERY possible step (1, 2, 3 and 4), resume each one
for n in [1, 2, 3, 4]:
    run_id = "demo_crash_after_" + str(n)
    LEDGER_PATH = "outputs/test_ledger_n.jsonl"
    for path in (LEDGER_PATH, checkpoint_path(run_id)):
        if os.path.exists(path):
            os.remove(path)
    try:
        run_agent(s1_message, "S1", run_id=run_id, crash_after_step=n)
    except SimulatedCrash:
        pass
    resumed = resume_agent(run_id)
    assert [x["tool"] for x in resumed["steps"]] == expected_tools, n
    assert resumed["status"] == "REFUND" and refund_from_steps(resumed["steps"]) == 128.25
    assert len(read_ledger(LEDGER_PATH)) == 1, "the refund must be recorded exactly once"
    os.remove(LEDGER_PATH)
    os.remove(checkpoint_path(run_id))
print("\nCHECK PASSED: a crash after tool 1, 2, 3 or 4 always resumes to REFUND 128.25 with a single ledger line")

os.remove("outputs/resume_demo_ledger.jsonl")        # the demo ledger was only evidence for the checks
LEDGER_PATH = real_ledger

"""## 9. ReAct versus Plan-and-execute

## ReAct vs plan-and-execute
My **ReAct** agent asks the model before every action. It needed 5 model calls for S1 (1,147 tokens), 3 for S5 (487) and 2 for S6 (172). Cost grows faster than calls because every call resends the history: S1’s calls cost 89, 113, 302, 317 and 326 tokens, as the policy passages are resent three times. A **Plan-and-execute** design asks for the whole step list once and runs the tools in code, so S1 would need about 1 to 2 calls and roughly 130 to 450 tokens (estimated, not built).

The price is flexibility.

**ReAct** sees every observation: for **S6, “order not found”** ended the run after 2 calls, and for **S5** the cancelled status led straight to **DENIED** with no refund calculated.

A **planner** replans only on an error, so **NOT_FOUND would trigger a replan (about 2 calls, like ReAct)**, but a cancelled order is a valid result, not an error, so a fixed plan would go on to calculate and issue unless extra checks stopped it.

I would choose **ReAct** here. The next action depends on data (status, delivery minutes, amounts above INR 1,000 go to a human), and a wrong step can pay money. Checking each observation before acting is safer, and with at most five tools and hard limits (6 iterations, 6,000 tokens, 10 seconds) the extra calls cost little. **At scale I would plan the fixed lookups once and keep ReAct for the decision.**
"""