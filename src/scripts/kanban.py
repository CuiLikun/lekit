"""Lightweight, zero-dependency Web Dashboard for task execution telemetry."""

import glob
import json
import mimetypes
import os
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR = os.environ.get("DASHBOARD_OUTPUTS", DEFAULT_OUTPUTS_DIR)
DEFAULT_PORT = int(os.environ.get("DASHBOARD_PORT", "8000"))
TEMPLATES_DIR = PROJECT_ROOT / "src" / "templates"
STATIC_DIR = TEMPLATES_DIR / "static"


def aggregate_records(outputs_dir=None, min_runs=5):
    """Scan all run_task_*.json in outputs_dir and aggregate data for the dashboard.

    Args:
        outputs_dir: directory containing run_task_*.json files.
        min_runs: drop sessions whose summary.total_runs is strictly less than
            this value. Default 5 matches the dashboard's default filter and
            excludes short debug runs from the KPI / trend / TOP5 stats.
            Set to 0 to disable the filter.
    """
    if outputs_dir is None:
        outputs_dir = OUTPUTS_DIR
    files = sorted(glob.glob(os.path.join(outputs_dir, "run_task_*.json")))
    runs = []
    total_runs_count = 0
    total_aborts_count = 0
    seen_policies = set()

    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        filename = os.path.basename(filepath)
        timestamp = data.get("timestamp", "")
        tasks = data.get("tasks", [])
        summary = data.get("summary", {})

        # Skip empty sessions: no runs means nothing to report and would only
        # pollute the trend chart and KPI counts.
        if summary.get("total_runs", 0) <= 0:
            continue

        # Apply the min-runs filter: drop sessions whose total run count is
        # below the threshold. This is the SESSION-level summary.total_runs
        # (not per-task runs), so a single short debug session is removed in
        # its entirety from the KPI / trend / TOP5 stats.
        if min_runs > 0 and summary.get("total_runs", 0) < min_runs:
            continue

        # Track policies
        for t in tasks:
            policy = t.get("policy")
            if policy:
                seen_policies.add(policy)

        runs.append(
            {
                "filename": filename,
                "timestamp": timestamp,
                "summary": summary,
                "tasks": tasks,
            }
        )

        total_runs_count += summary.get("total_runs", 0)
        total_aborts_count += summary.get("total_aborts", 0)

    # Compute global KPI
    average_success_rate = None
    if total_runs_count > 0:
        completed_count = max(0, total_runs_count - total_aborts_count)
        average_success_rate = round(completed_count / total_runs_count * 100.0, 2)

    # Sort runs: newest first by filename (which has timestamp embedded)
    runs.sort(key=lambda r: r["filename"], reverse=True)

    # Compute trend chart data, grouped by (task_id, prompt)
    # Labels must be in chronological (ascending) order
    trend_labels = []
    prompt_to_data = defaultdict(dict)  # task_key -> { label_index: rate }
    prompt_to_latest_policy = {}

    # Use reversed sorted runs (oldest -> newest) for trend chart
    for idx, run in enumerate(reversed(runs)):
        ts_short = run["timestamp"][:10] if run["timestamp"] else "Unknown"
        trend_labels.append(f"Session {ts_short}")
        for task in run["tasks"]:
            tid = task.get("task_id", "?")
            prompt = task.get("prompt", "Unknown")
            rate = task.get("success_rate", None)
            policy = task.get("policy", "")
            key = f"[#{tid}] {prompt}"
            prompt_to_data[key][idx] = rate
            if policy:
                prompt_to_latest_policy[key] = policy

    # Generate datasets
    trend_datasets = []
    # Sort task labels for consistent ordering
    for label in sorted(prompt_to_data.keys()):
        values_dict = prompt_to_data[label]
        # Build data array aligned with trend_labels
        data_points = []
        for i in range(len(trend_labels)):
            data_points.append(values_dict.get(i))
        latest_policy = prompt_to_latest_policy.get(label, "")
        full_label = f"{label} ({latest_policy})" if latest_policy else label
        trend_datasets.append(
            {
                "label": full_label,
                "data": data_points,
            }
        )

    # Per-task TOP 5 policies: for every task that has run at least once, build
    # a chart of its top 5 policies ranked by aggregated success rate on that
    # task. Within each task, policies are sorted by rate desc, then runs desc,
    # then policy name asc. Tasks with zero runs in every session are excluded.
    # Tasks themselves are ordered by their best (top) policy rate desc, then
    # total_runs desc, then task_id asc.
    PER_TASK_POLICY_TOP_N = 5
    # task_key -> { task_id, prompt, "policies": { policy: {runs, aborts} } }
    per_task = {}

    for run in runs:
        for t in run["tasks"]:
            runs_n = int(t.get("runs", 0) or 0)
            if runs_n <= 0:
                continue
            tid = t.get("task_id", "?")
            prompt = t.get("prompt", "Unknown")
            policy = t.get("policy") or ""
            key = f"[#{tid}] {prompt}"
            entry = per_task.setdefault(
                key,
                {
                    "task_id": tid,
                    "prompt": prompt,
                    "policies": defaultdict(lambda: {"runs": 0, "aborts": 0}),
                },
            )
            if policy:
                p_agg = entry["policies"][policy]
                p_agg["runs"] += runs_n
                p_agg["aborts"] += int(t.get("aborts", 0) or 0)

    task_top5_policies = []
    for key, entry in per_task.items():
        policy_list = []
        for policy, agg in entry["policies"].items():
            completed = max(0, agg["runs"] - agg["aborts"])
            rate = round(completed / agg["runs"] * 100.0, 2) if agg["runs"] > 0 else None
            policy_list.append(
                {
                    "policy": policy,
                    "runs": agg["runs"],
                    "aborts": agg["aborts"],
                    "success_rate": rate,
                }
            )
        # Sort policies within the task: rate desc, runs desc, name asc.
        policy_list.sort(
            key=lambda p: (
                -(p["success_rate"] if p["success_rate"] is not None else -1),
                -p["runs"],
                p["policy"],
            )
        )
        top_policies = policy_list[:PER_TASK_POLICY_TOP_N]
        best_rate = top_policies[0]["success_rate"] if top_policies else None
        total_runs = sum(p["runs"] for p in top_policies)
        task_top5_policies.append(
            {
                "title": key,
                "task_id": entry["task_id"],
                "prompt": entry["prompt"],
                "best_rate": best_rate,
                "total_runs": total_runs,
                "policies": top_policies,
            }
        )

    # Order tasks: best_rate desc, total_runs desc, task_id asc
    task_top5_policies.sort(
        key=lambda c: (
            -(c["best_rate"] if c["best_rate"] is not None else -1),
            -c["total_runs"],
            c["task_id"] if c["task_id"] is not None else 0,
        )
    )

    return {
        "kpi": {
            "total_runs": total_runs_count,
            "total_aborts": total_aborts_count,
            "average_success_rate_percent": average_success_rate,
            "total_policies": len(seen_policies),
        },
        "runs": runs,
        "trend_chart": {
            "labels": trend_labels,
            "datasets": trend_datasets,
        },
        "task_top5_policies": task_top5_policies,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    """Handle dashboard routes: GET /, GET /api/records."""

    def log_message(self, format, *args):
        # Suppress default stderr access logs for cleaner console
        return

    def do_GET(self):
        if self.path.startswith("/api/records"):
            # Parse optional `?min_runs=N` query param. Missing/empty/non-int
            # values fall back to the aggregate_records default (5).
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            try:
                min_runs = int(params.get("min_runs", [""])[0])
            except ValueError:
                min_runs = 5
            self._send_json(aggregate_records(min_runs=min_runs))
        elif self.path == "/" or self.path == "/index.html":
            self._serve_index()
        elif self.path.startswith("/static/"):
            self._serve_static(self.path[len("/static/") :])
        else:
            self.send_error(404)

    def _send_json(self, data):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _serve_index(self):
        index_path = TEMPLATES_DIR / "kanban.html"
        if not os.path.exists(index_path):
            self.send_error(500, "kanban.html missing")
            return
        with open(index_path, "rb") as f:
            payload = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_static(self, rel_path: str):
        """Serve a file from templates/static/, blocking path traversal."""
        # Strip query string if present
        rel_path = rel_path.split("?", 1)[0].split("#", 1)[0]
        # Reject path traversal attempts
        safe_path = os.path.normpath(rel_path).lstrip(os.sep)
        if safe_path.startswith("..") or os.path.isabs(safe_path):
            self.send_error(403)
            return
        file_path = os.path.join(STATIC_DIR, safe_path)
        if not os.path.isfile(file_path):
            self.send_error(404)
            return
        ctype, _ = mimetypes.guess_type(file_path)
        ctype = ctype or "application/octet-stream"
        try:
            with open(file_path, "rb") as f:
                payload = f.read()
        except OSError:
            self.send_error(500, "Failed to read static file")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        # webfonts need CORS in some contexts but same-origin is fine
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(payload)


def main():
    port = DEFAULT_PORT
    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"[dashboard] serving outputs dir: {OUTPUTS_DIR}")
    print(f"[dashboard] listening on http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[dashboard] shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
