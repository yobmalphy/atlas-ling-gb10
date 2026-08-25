#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run, grade, and report the published second GB10 coding benchmark only."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pickle
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
DATASET_SHA256 = "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"
DATA_URL = (
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/"
    f"resolve/{DATASET_REVISION}/test6.jsonl"
)
CACHE = ROOT / ".cache" / "test6.jsonl"
SELECTED = [
    "abc387_c",
    "abc389_d",
    "abc390_d",
    "abc390_c",
    "abc394_d",
    "abc396_d",
    "abc397_c",
    "abc398_c",
]
SYSTEM_MESSAGE = (
    "You are an expert Python programmer. Produce a correct and efficient "
    "solution that follows the requested interface and passes hidden tests."
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_suite() -> dict[str, dict[str, Any]]:
    if CACHE.exists():
        payload = CACHE.read_bytes()
    else:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=120) as response:
            payload = response.read()
        CACHE.write_bytes(payload)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != DATASET_SHA256:
        raise RuntimeError(
            f"test6.jsonl SHA-256 mismatch: expected {DATASET_SHA256}, got {actual}"
        )
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    by_id = {row["question_id"]: row for row in rows}
    missing = set(SELECTED) - by_id.keys()
    if missing:
        raise RuntimeError(f"missing official LiveCodeBench tasks: {sorted(missing)}")
    for task_id in SELECTED:
        difficulty = str(by_id[task_id].get("difficulty", "")).lower()
        if difficulty != "medium":
            raise RuntimeError(f"{task_id} is not medium difficulty: {difficulty!r}")
    return by_id


def lcb_prompt(task: dict[str, Any]) -> str:
    prompt = f"### Question:\n{task['question_content']}\n\n"
    if task.get("starter_code"):
        prompt += (
            "### Format: You will use the following starter code to write the "
            "solution to the problem and enclose your code within delimiters.\n"
            f"```python\n{task['starter_code']}\n```\n\n"
        )
    else:
        prompt += (
            "### Format: Read the inputs from stdin solve the problem and write "
            "the answer to stdout (do not directly test on the sample inputs). "
            "Enclose your code within delimiters as follows. Ensure that when "
            "the python program runs, it reads the inputs, runs the algorithm "
            "and writes output to STDOUT.\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return prompt + "### Answer: (use the provided format with backticks)\n\n"


def request_chat(endpoint: str, served_model: str, prompt: str) -> dict[str, Any]:
    body = {
        "model": served_model,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 8192,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 512},
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    payload["_client_wall_seconds"] = time.monotonic() - started
    return payload


def extract_code(text: str) -> str:
    python_blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", text, flags=re.I | re.S)
    if python_blocks:
        return python_blocks[0].strip() + "\n"
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S)
    if blocks:
        return blocks[0].strip() + "\n"
    opening = re.search(r"```(?:python|py)?\s*\n", text, flags=re.I)
    if opening:
        return text[opening.end() :].strip() + "\n"
    cleaned = re.sub(r"^```(?:python|py)?\s*\n", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip() + "\n"


def decoded_tests(task: dict[str, Any]) -> dict[str, Any]:
    public = json.loads(task["public_test_cases"])
    raw_private = task["private_test_cases"]
    try:
        private = json.loads(raw_private)
    except json.JSONDecodeError:
        private = json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw_private))))
    all_tests = public + private
    metadata = json.loads(task["metadata"])
    return {
        "input_output": json.dumps(
            {
                "inputs": [case["input"] for case in all_tests],
                "outputs": [case["output"] for case in all_tests],
                "fn_name": metadata.get("func_name"),
            }
        )
    }


def grade_solution(record: dict[str, Any], task: dict[str, Any], lcb_repo: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        sample = temp / "sample.json"
        solution = temp / "solution.py"
        grader = temp / "grader.py"
        atomic_json(sample, decoded_tests(task))
        solution.write_text(record["solution"])
        grader.write_text(
            "import json,multiprocessing,sys\n"
            "multiprocessing.set_start_method('fork', force=True)\n"
            f"sys.path.insert(0,{str(lcb_repo)!r})\n"
            "from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness\n"
            "def main():\n"
            f"    sample=json.load(open({str(sample)!r}))\n"
            f"    solution=open({str(solution)!r}).read()\n"
            "    result,meta=check_correctness(sample,solution,timeout=10,debug=False)\n"
            "    passed=bool(result) and all(x is True for x in result)\n"
            "    print(json.dumps({'passed':passed,'results':result,'metadata':meta}))\n"
            "if __name__ == '__main__': main()\n"
        )
        process = subprocess.run(
            [sys.executable, str(grader)],
            text=True,
            capture_output=True,
            timeout=900,
            cwd=temp,
        )
        if process.returncode != 0:
            return {
                "passed": False,
                "grader_error": process.stderr[-4000:],
                "exit_code": process.returncode,
            }
        try:
            return json.loads(process.stdout.splitlines()[-1])
        except Exception:
            return {
                "passed": False,
                "grader_error": process.stdout[-4000:] + process.stderr[-4000:],
            }


def suite_manifest(by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": task_id,
            "title": by_id[task_id]["question_title"],
            "platform": by_id[task_id]["platform"],
            "contest_date": by_id[task_id]["contest_date"],
            "difficulty": by_id[task_id]["difficulty"],
        }
        for task_id in SELECTED
    ]


def run(args: argparse.Namespace) -> None:
    by_id = load_suite()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark": "livecodebench-test6-medium-second-test",
        "model_label": args.model_label,
        "served_model": args.served_model,
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.checkpoint_revision,
        "atlas_commit": args.atlas_commit,
        "atlas_image": args.atlas_image,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "livecodebench/code_generation_lite",
            "revision": DATASET_REVISION,
            "file": "test6.jsonl",
            "sha256": DATASET_SHA256,
            "difficulty": "medium",
        },
        "contract": {
            "max_seq_len": 262144,
            "tp_size": 1,
            "max_num_seqs": 1,
            "max_batch_size": 1,
            "kv_cache_dtype": "bf16",
            "mtp_drafts": args.mtp_drafts,
            "temperature": 0.0,
            "top_p": 1.0,
            "thinking_budget": 512,
            "max_tokens": 8192,
            "prefix_caching": False,
        },
        "tasks": suite_manifest(by_id),
    }
    atomic_json(output / "manifest.json", manifest)
    for index, task_id in enumerate(SELECTED, 1):
        path = output / f"{index:02d}-{task_id}.json"
        if path.exists() and not args.force:
            print(f"SKIP {index:02d}/08 {task_id}", flush=True)
            continue
        print(f"RUN  {index:02d}/08 {task_id}", flush=True)
        started = time.monotonic()
        try:
            raw = request_chat(args.endpoint, args.served_model, lcb_prompt(by_id[task_id]))
            message = raw["choices"][0]["message"]
            content = message.get("content") or ""
            record = {
                "task_id": task_id,
                "suite": "livecodebench-test6-medium",
                "model_label": args.model_label,
                "status": "captured",
                "content": content,
                "reasoning_content": message.get("reasoning_content") or "",
                "solution": extract_code(content),
                "finish_reason": raw["choices"][0].get("finish_reason"),
                "usage": raw.get("usage", {}),
                "client_wall_seconds": raw.get(
                    "_client_wall_seconds", time.monotonic() - started
                ),
                "captured_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            record = {
                "task_id": task_id,
                "suite": "livecodebench-test6-medium",
                "model_label": args.model_label,
                "status": "request_error",
                "error": repr(exc),
                "captured_at": datetime.now(timezone.utc).isoformat(),
            }
        atomic_json(path, record)
        usage = record.get("usage", {})
        print(
            f"     {record['status']} tokens={usage.get('completion_tokens')} "
            f"tps={usage.get('response_token/s')} "
            f"ttft_ms={usage.get('time_to_first_token_ms')}",
            flush=True,
        )
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output / "manifest.json", manifest)


def grade(args: argparse.Namespace) -> None:
    by_id = load_suite()
    result_dir = Path(args.result_dir).resolve()
    for path in sorted(result_dir.glob("[0-9][0-9]-*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "captured" and not args.force:
            continue
        print(f"GRADE {record['task_id']}", flush=True)
        record["grade"] = grade_solution(
            record, by_id[record["task_id"]], Path(args.lcb_repo).resolve()
        )
        record["status"] = "graded"
        atomic_json(path, record)
        print(f"      passed={record['grade'].get('passed')}", flush=True)


def report(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    root = Path(args.results_root).resolve()
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(model_dir.glob("[0-9][0-9]-*.json")):
            record = json.loads(path.read_text())
            usage = record.get("usage", {})
            detail = usage.get("completion_tokens_details", {})
            rows.append(
                {
                    "model": record["model_label"],
                    "task": record["task_id"],
                    "passed": bool(record.get("grade", {}).get("passed")),
                    "tokens": usage.get("completion_tokens"),
                    "tps": usage.get("response_token/s"),
                    "ttft": usage.get("time_to_first_token_ms"),
                    "drafts": detail.get("accepted_prediction_tokens", 0),
                    "finish": record.get("finish_reason"),
                }
            )
    if not rows:
        raise RuntimeError(f"no result records found under {root}")
    lines = [
        "# Reproduced LiveCodeBench test6 medium comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Model | Passed | Median TPS | Median TTFT ms | Accepted drafts | Length cutoffs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        tps = [float(row["tps"]) for row in subset if row["tps"] is not None]
        ttft = [float(row["ttft"]) for row in subset if row["ttft"] is not None]
        lines.append(
            f"| {model} | {sum(row['passed'] for row in subset)}/{len(subset)} | "
            f"{statistics.median(tps):.2f} | {statistics.median(ttft):.1f} | "
            f"{sum(int(row['drafts'] or 0) for row in subset)} | "
            f"{sum(row['finish'] == 'length' for row in subset)} |"
        )
    lines += [
        "",
        "| Model | Task | Pass | Tokens | TPS | TTFT ms | Accepted drafts | Finish |",
        "|---|---|:---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['task']} | {'yes' if row['passed'] else 'no'} | "
            f"{row['tokens']} | {float(row['tps']):.2f} | {float(row['ttft']):.1f} | "
            f"{row['drafts']} | {row['finish']} |"
        )
    destination = Path(args.output).resolve()
    destination.write_text("\n".join(lines) + "\n")
    print(destination)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--endpoint", required=True)
    run_parser.add_argument("--model-label", required=True)
    run_parser.add_argument("--served-model", required=True)
    run_parser.add_argument("--checkpoint", required=True)
    run_parser.add_argument("--checkpoint-revision", required=True)
    run_parser.add_argument("--atlas-commit", required=True)
    run_parser.add_argument("--atlas-image", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--mtp-drafts", type=int, required=True)
    run_parser.add_argument("--force", action="store_true")
    run_parser.set_defaults(func=run)
    grade_parser = commands.add_parser("grade")
    grade_parser.add_argument("--result-dir", required=True)
    grade_parser.add_argument("--lcb-repo", required=True)
    grade_parser.add_argument("--force", action="store_true")
    grade_parser.set_defaults(func=grade)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--results-root", required=True)
    report_parser.add_argument("--output", required=True)
    report_parser.set_defaults(func=report)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
