"""Poll all tasks until terminal, then print a summary."""
import time
import httpx

BASE = "http://localhost:8000"
POLL_INTERVAL = 8
TIMEOUT = 300  # 5 min max

client = httpx.Client(base_url=BASE, timeout=90)

def wait_for_all_terminal(timeout=TIMEOUT):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            tasks = client.get("/tasks?limit=100").json()
        except Exception as e:
            print(f"  GET /tasks failed: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        statuses = {t["task_id"]: t["status"] for t in tasks}
        in_flight = [tid for tid, st in statuses.items() if st not in ("completed", "failed")]
        print(f"  {time.strftime('%H:%M:%S')} — {len(tasks)} tasks total, {len(in_flight)} still running...")

        if not in_flight:
            return tasks
        time.sleep(POLL_INTERVAL)
    # Return whatever state we have on timeout
    return client.get("/tasks?limit=100").json()

print("Waiting for all tasks to reach terminal state...")
tasks = wait_for_all_terminal()

print("\n" + "="*80)
print(f"{'ID':10} {'STATUS':12} {'TITLE':40} {'ERROR'}")
print("="*80)
for t in sorted(tasks, key=lambda x: x["created_at"]):
    tid = t["task_id"][:8]
    st  = t["status"]
    title = (t.get("title") or "")[:38]
    err   = (t.get("error") or "")[:60]
    arts  = len(t.get("artifacts") or [])
    mark  = "OK" if st == "completed" else ("FAIL" if st == "failed" else "????")
    print(f"[{mark}] {tid}  {st:12}  {title:40}  {err}")
    if arts:
        print(f"         artifacts: {[a['filename'] for a in t.get('artifacts',[])[:4]]}")

completed = [t for t in tasks if t["status"] == "completed"]
failed    = [t for t in tasks if t["status"] == "failed"]
other     = [t for t in tasks if t["status"] not in ("completed","failed")]
print(f"\nSummary: {len(completed)} completed, {len(failed)} failed, {len(other)} other")
