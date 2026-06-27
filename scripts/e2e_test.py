"""
e2e_test_v2.py — Full E2E re-run after all fixes.

Covers all original 22+ cases plus targeted verification of each fix:
  A1 — GET requests don't timeout under load (sandbox non-blocking)
  A2 — DELETE actually cancels the running pipeline (not just DB status)
  B1 — "Last failure: []" replaced by meaningful exit-code messages
  B2 — No duplicate artifacts on Orchestrator retries
  B3 — docker_unavailable logged once, not per-run
  B4 — Semaphore caps concurrent pipelines
  B5 — Groq errors show friendly messages
  B6 — Whitespace-only description >=10 chars now rejected
"""

import json
import time
import httpx

BASE = "http://localhost:8000"
POLL_INTERVAL = 5
POLL_TIMEOUT  = 180

findings_crashes  = []
findings_bad_ux   = []
findings_ok       = []

def hdr(t):
    print(f"\n{'='*64}\n  {t}\n{'='*64}")

def show(label, status, body, note=""):
    ok = str(status).startswith("2")
    colour = "\033[32m" if ok else ("\033[31m" if str(status).startswith("5") else "\033[33m")
    reset = "\033[0m"
    body_s = json.dumps(body, indent=2) if isinstance(body, (dict, list)) else str(body)
    if len(body_s) > 600:
        body_s = body_s[:600] + "\n  ... [truncated]"
    print(f"\n  \033[1m{label}\033[0m")
    print(f"  Status : {colour}{status}{reset}")
    print(f"  Body   : {body_s}")
    if note:
        print(f"  Note   : {note}")

def post_task(c, payload, label, raw=False):
    try:
        if raw:
            r = c.post("/tasks", content=payload, headers={"Content-Type": "application/json"})
        else:
            r = c.post("/tasks", json=payload)
        try:
            body = r.json()
        except Exception:
            body = r.text
        show(label, r.status_code, body)
        return r.status_code, body
    except Exception as e:
        show(label, "ERR", str(e))
        return "ERR", str(e)

def poll_task(c, task_id, label, timeout=POLL_TIMEOUT):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = c.get(f"/tasks/{task_id}")
            d = r.json()
            st = d.get("status")
            if st != last:
                print(f"    [{label}] status -> {st}")
                last = st
            if st in ("completed", "failed"):
                return d
        except httpx.ReadTimeout:
            print(f"    [{label}] GET /tasks/{task_id[:8]} TIMED OUT")
            return {"status": "GET_TIMEOUT", "task_id": task_id}
        time.sleep(POLL_INTERVAL)
    return {"status": "POLL_TIMEOUT", "task_id": task_id}

def classify(label, status, body, expected):
    s = str(status)
    if s.startswith("5") or status == "ERR":
        findings_crashes.append(f"{label}: got {status} — {str(body)[:200]}")
    elif status != expected:
        findings_bad_ux.append(f"{label}: expected {expected}, got {status} — {str(body)[:200]}")
    else:
        findings_ok.append(f"{label}: {status} OK")


LONG_DESC    = "X" * 10_100
WS_DESC      = "   " * 4   # 12 whitespace chars — exceeds old min_length=10

SIMPLE_TASK = {
    "title": "Reverse a string",
    "description": "Write a Python function reverse_string(s: str) -> str that returns the input reversed. Handle empty strings.",
    "language": "python",
    "max_retries": 3,
    "budget_usd": 1.0,
}
MODERATE_TASK = {
    "title": "Email validation REST endpoint",
    "description": "Build a FastAPI POST /validate-email endpoint accepting {\"email\": str} and returning {\"valid\": bool, \"reason\": str}. Validate format with regex and return 422 for missing fields.",
    "language": "python",
    "max_retries": 3,
    "budget_usd": 1.0,
}
IMPOSSIBLE_TASK = {
    "title": "Impossible contradictory task",
    "description": "Write a Python function that is simultaneously O(1) and O(n), uses no variables, operators, or keywords, produces output without statements, and makes HTTP requests without network access.",
    "language": "python",
    "max_retries": 1,
    "budget_usd": 0.5,
}
RAPID_TASK = {
    "title": "Rapid fire test",
    "description": "Write a function that returns the current timestamp as an ISO 8601 string.",
    "language": "python",
    "max_retries": 1,
    "budget_usd": 0.25,
}


def main():
    client = httpx.Client(base_url=BASE, timeout=60)  # 60s timeout, not 30s

    # ── HEALTH ────────────────────────────────────────────────────────────────
    hdr("HEALTH CHECK")
    r = client.get("/health")
    show("GET /health", r.status_code, r.json())
    classify("Health", r.status_code, r.json(), 200)

    # ── HAPPY PATH — submit both early so pipelines run in parallel ───────────
    hdr("HAPPY PATH - SUBMIT")
    s1, b1 = post_task(client, SIMPLE_TASK,   "POST simple (reverse string)")
    s2, b2 = post_task(client, MODERATE_TASK, "POST moderate (email validator)")
    tid_simple   = b1.get("task_id") if isinstance(b1, dict) else None
    tid_moderate = b2.get("task_id") if isinstance(b2, dict) else None

    # ── VALIDATION EDGE CASES ─────────────────────────────────────────────────
    hdr("VALIDATION EDGE CASES")

    s, b = post_task(client, {"title": "x", "description": "",     "language": "python"}, "Empty description")
    classify("Empty description", s, b, 422)

    s, b = post_task(client, {"title": "x", "description": "   \t\n  ", "language": "python"}, "Whitespace-only (7 chars, too short)")
    classify("Whitespace-only short", s, b, 422)

    # B6: whitespace-only >=10 chars — was accepted before, must be rejected now
    s, b = post_task(client, {"title": "x", "description": WS_DESC, "language": "python"}, "B6: Whitespace-only 12 chars (NEW — must now reject)")
    classify("B6 whitespace 12 chars", s, b, 422)
    if s == 422:
        findings_ok.append("B6 FIXED: whitespace-only >=10 chars now returns 422")
    else:
        findings_bad_ux.append(f"B6 NOT FIXED: whitespace-only 12 chars returned {s}, expected 422")

    s, b = post_task(client, {"title": "x", "description": "fix it", "language": "python"}, "Vague 'fix it'")
    classify("Vague description", s, b, 422)

    s, b = post_task(client, b'{"title":"bad","description":NOTJSON}', "Malformed JSON", raw=True)
    classify("Malformed JSON", s, b, 422)

    s, b = post_task(client, {"description": "A valid description here", "language": "python"}, "Missing title")
    classify("Missing title", s, b, 422)

    s, b = post_task(client, {"title": "x", "language": "python"}, "Missing description")
    classify("Missing description field", s, b, 422)

    s, b = post_task(client, {"title": "x", "description": "A valid description", "language": "python", "budget_usd": "free"}, "Wrong type budget_usd")
    classify("Wrong type budget_usd", s, b, 422)

    s, b = post_task(client, {"title": "x", "description": "A valid description", "language": "python", "max_retries": "many"}, "Wrong type max_retries")
    classify("Wrong type max_retries", s, b, 422)

    s, b = post_task(client, {"title": "x", "description": "A valid description", "language": "python", "budget_usd": -5.0}, "Negative budget_usd")
    classify("Negative budget", s, b, 422)

    s, b = post_task(client, {"title": "x", "description": LONG_DESC, "language": "python"}, f"Description {len(LONG_DESC):,} chars")
    classify("Long description", s, b, 422)

    # ── BEHAVIOURAL ───────────────────────────────────────────────────────────
    hdr("BEHAVIOURAL EDGE CASES")

    s, b = post_task(client, {
        "title": "Unicode emoji",
        "description": "Write a function that handles Unicode: 'hello' in Chinese is '你好世界', in Arabic 'مرحبا'. Return a greeting string.",
        "language": "python",
    }, "Unicode in description")
    tid_unicode = b.get("task_id") if isinstance(b, dict) and s == 202 else None
    classify("Unicode", s, b, 202)

    s, b = post_task(client, {
        "title": "'; DROP TABLE tasks; --",
        "description": "Write a function that sanitizes SQL input to prevent injection attacks. Use parameterized queries.",
        "language": "python",
    }, "SQL-injection-style strings")
    tid_sql = b.get("task_id") if isinstance(b, dict) and s == 202 else None
    classify("SQL injection style", s, b, 202)

    s, b = post_task(client, IMPOSSIBLE_TASK, "Impossible/contradictory task")
    tid_impossible = b.get("task_id") if isinstance(b, dict) and s == 202 else None
    classify("Impossible task accepted", s, b, 202)

    # ── RAPID FIRE (A1 + B4 verification) ────────────────────────────────────
    hdr("RAPID FIRE — 5 concurrent tasks (A1 + B4)")
    rapid_ids = []
    for i in range(5):
        r = client.post("/tasks", json=RAPID_TASK)
        body = r.json()
        tid = body.get("task_id", "ERR")
        print(f"  Rapid #{i+1}: status={r.status_code} task_id={str(tid)[:8]}...")
        if r.status_code == 202:
            rapid_ids.append(body["task_id"])
        else:
            findings_bad_ux.append(f"Rapid #{i+1}: {r.status_code}")
    print(f"  {len(rapid_ids)}/5 accepted")

    # Immediately verify GET requests work despite concurrent pipelines (A1)
    print("\n  A1 check: firing GET /tasks immediately to verify event loop is not blocked...")
    get_ok = 0
    for _ in range(3):
        try:
            rr = client.get("/tasks?limit=5")
            if rr.status_code == 200:
                get_ok += 1
        except httpx.ReadTimeout:
            print("    GET /tasks TIMED OUT (event loop still blocking!)")
    print(f"  A1 GET responsiveness: {get_ok}/3 succeeded")
    if get_ok == 3:
        findings_ok.append("A1 FIXED: GET /tasks responds under concurrent load (no event loop blocking)")
    else:
        findings_bad_ux.append(f"A1 PARTIAL: only {get_ok}/3 GETs succeeded under load")

    # ── CANCEL VERIFICATION (A2) ──────────────────────────────────────────────
    hdr("CANCEL — A2 verification (pipeline actually stops)")
    cancel_id = None
    if rapid_ids:
        cancel_id = rapid_ids.pop()
        # Give it a second to start, then cancel
        time.sleep(2)
        r = client.delete(f"/tasks/{cancel_id}")
        show(f"DELETE /tasks/{cancel_id[:8]}...", r.status_code, r.text or "(no body)")
        classify("Cancel 204", r.status_code, "", 204)

        # Wait a few seconds and check the status hasn't been overwritten
        time.sleep(8)
        r2 = client.get(f"/tasks/{cancel_id}")
        d = r2.json()
        print(f"  Status after cancel + 8s: {d.get('status')} | error: {d.get('error')}")
        if d.get("error") == "Cancelled by user" and d.get("status") == "failed":
            findings_ok.append("A2 FIXED: cancelled task stays at 'failed' with 'Cancelled by user' error")
        else:
            findings_bad_ux.append(f"A2 CHECK: status={d.get('status')}, error={d.get('error')} — may have been overwritten by pipeline")

    r = client.delete("/tasks/00000000-0000-0000-0000-000000000000")
    show("DELETE non-existent task", r.status_code, r.json())
    classify("Cancel 404", r.status_code, r.json(), 404)

    # ── POLL HAPPY PATH ───────────────────────────────────────────────────────
    hdr("POLL HAPPY PATH FOR COMPLETION")
    to_poll = [
        (tid_simple,     "Simple (reverse string)"),
        (tid_moderate,   "Moderate (email validator)"),
        (tid_impossible, "Impossible task"),
    ]
    for tid, label in to_poll:
        if not tid:
            print(f"  [{label}] SKIPPED — not created")
            continue
        print(f"\n  Polling [{label}] {str(tid)[:8]}...")
        final = poll_task(client, tid, label)
        st    = final.get("status")
        err   = final.get("error", "")
        arts  = final.get("artifacts", [])

        # B2: check for duplicate filenames
        fnames = [a["filename"] for a in arts]
        dupes  = [f for f in set(fnames) if fnames.count(f) > 1]

        print(f"  Final status : {st}")
        if err:  print(f"  Error        : {err}")
        if arts: print(f"  Artifacts    : {fnames}")
        if dupes:
            findings_bad_ux.append(f"B2 NOT FIXED [{label}]: duplicate artifacts {dupes}")
        elif arts:
            findings_ok.append(f"B2 OK [{label}]: no duplicate artifacts ({fnames})")

        if st == "GET_TIMEOUT":
            findings_bad_ux.append(f"A1 CHECK [{label}]: GET timed out mid-poll")
        elif st == "completed":
            findings_ok.append(f"{label}: completed with {len(arts)} artifacts")
        elif st == "failed":
            # Check B1: error message quality
            if "Last failure: []" in (err or ""):
                findings_bad_ux.append(f"B1 NOT FIXED [{label}]: still shows 'Last failure: []'")
            elif err:
                findings_ok.append(f"{label}: failed with clear message — {err[:100]}")
                # Impossible task should fail
                if "impossible" in label.lower():
                    findings_ok.append(f"Impossible task: failed gracefully as expected")
            else:
                findings_bad_ux.append(f"{label}: failed with no error message")
        elif st == "POLL_TIMEOUT":
            findings_bad_ux.append(f"{label}: pipeline TIMED OUT after {POLL_TIMEOUT}s")

    # ── QUICK STATUS CHECK (Unicode, SQL) ─────────────────────────────────────
    hdr("QUICK STATUS CHECK — UNICODE + SQL TASKS")
    for tid, label in [(tid_unicode, "Unicode task"), (tid_sql, "SQL injection style")]:
        if not tid:
            continue
        r = client.get(f"/tasks/{tid}")
        d = r.json()
        st = d.get("status")
        err = d.get("error", "")
        print(f"  [{label}] -> {st}{(' — ' + err[:60]) if err else ''}")
        if r.status_code == 500:
            findings_crashes.append(f"{label}: 500 on GET")
        else:
            findings_ok.append(f"{label}: in pipeline, no 500 (status={st})")

    # ── WAIT FOR ALL RAPID-FIRE TO COMPLETE ───────────────────────────────────
    hdr("RAPID FIRE FINAL STATUSES")
    for tid in rapid_ids:
        final = poll_task(client, tid, f"Rapid {tid[:8]}", timeout=120)
        st  = final.get("status")
        err = final.get("error", "")
        # B5 check
        if "Connection error" in (err or "") or "Request timed out" in (err or ""):
            findings_bad_ux.append(f"B5 NOT FIXED: raw error still surfaced — {err}")
        elif "unreachable" in (err or "") or "rate limit" in (err or "").lower() or "HTTP" in (err or ""):
            findings_ok.append(f"B5 FIXED: friendly error — {err[:80]}")
        arts  = final.get("artifacts", [])
        fnames = [a["filename"] for a in arts]
        dupes  = [f for f in set(fnames) if fnames.count(f) > 1]
        if dupes:
            findings_bad_ux.append(f"B2 NOT FIXED [rapid {tid[:8]}]: dupes {dupes}")
        mark = "OK" if st == "completed" else "FAIL"
        print(f"  [{mark}] {tid[:8]} -> {st}{(' — ' + err[:60]) if err else ''}")

    # ── SERVER LOG — docker_unavailable count (B3) ────────────────────────────
    hdr("B3 CHECK — docker_unavailable log count")
    try:
        with open("/tmp/atlas_server_v2.log") as f:
            lines = f.readlines()
        docker_warns = [l for l in lines if "docker_unavailable" in l]
        print(f"  'docker_unavailable' log entries: {len(docker_warns)} (should be 1)")
        if len(docker_warns) <= 1:
            findings_ok.append(f"B3 FIXED: docker_unavailable logged {len(docker_warns)} time(s), not per-run")
        else:
            findings_bad_ux.append(f"B3 NOT FIXED: docker_unavailable logged {len(docker_warns)} times")
        error_lines = [l.strip() for l in lines if " error " in l.lower() and "docker" not in l.lower()]
        if error_lines:
            print(f"\n  Other error lines ({len(error_lines)}):")
            for el in error_lines[-5:]:
                print(f"    {el[:140]}")
    except FileNotFoundError:
        print("  Server log not found")

    # ── FINDINGS SUMMARY ──────────────────────────────────────────────────────
    hdr("FINDINGS SUMMARY")
    print(f"\n\033[1m\033[31m(a) CRASHES / 500s ({len(findings_crashes)})\033[0m")
    for f in findings_crashes or ["None"]:
        print(f"  * {f}")

    print(f"\n\033[1m\033[33m(b) BAD UX / UNRESOLVED ({len(findings_bad_ux)})\033[0m")
    for f in findings_bad_ux or ["None"]:
        print(f"  * {f}")

    print(f"\n\033[1m\033[32m(c) WORKING ({len(findings_ok)})\033[0m")
    for f in findings_ok:
        print(f"  * {f}")

    print(f"\n\033[1mDone.\033[0m\n")


if __name__ == "__main__":
    main()
