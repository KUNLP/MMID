#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import subprocess
import sys
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple, TextIO


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENT_ROOT = PROJECT_ROOT.parent 
BENCHMARK_DIR = PROJECT_ROOT/ "benchmark"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

class TeeWriter:
    def __init__(self, *writers: TextIO):
        self.writers = writers

    def write(self, text: str):
        for w in self.writers:
            w.write(text)
            w.flush()

    def flush(self):
        for w in self.writers:
            w.flush()


KST = timezone(timedelta(hours=9))

MODEL_CONFIGS = {
    "qwen3-2b": {"model_id": "Qwen/Qwen3-VL-2B-Instruct", "cache_dir": "../models/Qwen3-VL-2B-Instruct"},
    "qwen3-4b": {"model_id": "Qwen/Qwen3-VL-4B-Instruct", "cache_dir": "../models/Qwen3-VL-4B-Instruct"},
    "qwen3-8b": {"model_id": "Qwen/Qwen3-VL-8B-Instruct", "cache_dir": "../models/Qwen3-VL-8B-Instruct"},
    "qwen2-2b": {"model_id": "Qwen/Qwen2-VL-2B-Instruct", "cache_dir": "../models/Qwen2-VL-2B-Instruct"},
    "qwen2-7b": {"model_id": "Qwen/Qwen2-VL-7B-Instruct", "cache_dir": "../models/Qwen2-VL-7B-Instruct"},
}

TASK_RE = re.compile(r"task(\d+)", re.IGNORECASE)
TYPE_RE = re.compile(r"type(\d+)", re.IGNORECASE)


def kst_timestamp() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")


def list_benchmark_files(language_arg: str) -> List[str]:
    lang_benchmark_dir = BENCHMARK_DIR / language_arg
    if not lang_benchmark_dir.exists():
        return []
    files = sorted([p.name for p in lang_benchmark_dir.glob("*.jsonl")])
    return files


def build_groups_from_files(files: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}

    for tid in range(1, 9):
        key = f"task{tid}"
        groups[key] = [f for f in files if re.search(rf"task{tid}\b|task{tid}_|task{tid}-", f, re.IGNORECASE) or
                       (TASK_RE.search(f) and int(TASK_RE.search(f).group(1)) == tid)]

    # task1 subtype
    groups["task3_type1"] = [f for f in groups.get("task3", []) if TYPE_RE.search(f) and TYPE_RE.search(f).group(1) == "1"]
    groups["task3_type2"] = [f for f in groups.get("task3", []) if TYPE_RE.search(f) and TYPE_RE.search(f).group(1) == "2"]

    # task8 subtype
    groups["task8_type1"] = [f for f in groups.get("task8", []) if TYPE_RE.search(f) and TYPE_RE.search(f).group(1) == "1"]
    groups["task8_type2"] = [f for f in groups.get("task8", []) if TYPE_RE.search(f) and TYPE_RE.search(f).group(1) == "2"]

    groups["all"] = files[:]
    return groups


def get_benchmarks_to_run(benchmark_args: list, language_arg: str) -> list:
    files = list_benchmark_files(language_arg)
    groups = build_groups_from_files(files)
    
    if not benchmark_args:
        return groups.get("all", [])

    benchmarks: List[str] = []
    for arg in benchmark_args:
        if arg in groups:
            benchmarks.extend(groups[arg])
        elif arg in files:
            benchmarks.append(arg)
        elif arg.endswith(".jsonl"):
            benchmarks.append(arg)
        else:
            print(f"[WARNING] Unknown benchmark/group: {arg}")

    seen = set()
    uniq = []
    for x in benchmarks:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def infer_meta_from_filename(fname: str) -> Tuple[int, str, Optional[str], str]:
    """
    Return: (task_id, task_name, task_subtype_value, question_type)
    - task_id: extracted from `task(\d+)` in the filename
    - subtype: extracted from `type(\d+)` in the filename, if present
    - question_type: inferred as `multiple_choice`, `yes_or_no`, or `open`
    """
    m = TASK_RE.search(fname)
    if not m:
        raise ValueError(f"Cannot infer task_id from filename: {fname}")
    task_id = int(m.group(1))
    task_name = f"task{task_id}"

    subtype = None
    mt = TYPE_RE.search(fname)
    if mt:
        subtype = mt.group(1)

    low = fname.lower()
    if "multiple_choice" in low:
        qtype = "multiple_choice"
    elif "yes_or_no" in low or "yes-no" in low or "yn" in low:
        qtype = "yes_or_no"
    else:
        qtype = "open"

    if task_id == 1:
        qtype = "open"

    return task_id, task_name, subtype, qtype


def run_one(
    model_id: str,
    benchmark_file: str,
    *,
    device_map: str,
    seed: int,
    shuffle_seed: int,
    max_items: int,
    save_prompts: bool,
    mode: str,
    template_key: str,
    lang: str = "ko",
    question_ids: Optional[List[str]] = None,
    question_ids_file: Optional[str] = None,
    question_id_regex: Optional[str] = None,
    cache_dir: Optional[str] = None,
    count: int = -1,
):
    task_id, task_name, subtype, qtype = infer_meta_from_filename(benchmark_file)

    lang_benchmark_dir = BENCHMARK_DIR / lang
    bench_path = lang_benchmark_dir / benchmark_file
    data_path_arg = benchmark_file if bench_path.exists() else str(benchmark_file)

    cmd = [
        sys.executable,
        "-m",
        "experiments.Qwen3_VL_Instruct.evaluator",
        "--data_root_path", str(lang_benchmark_dir),
        "--data_path", data_path_arg,
        "--output_root", str(RESULTS_DIR),
        "--model_id", model_id,
        "--device_map", device_map,
        "--seed", str(seed),
        "--shuffle_seed", str(shuffle_seed),
        "--task_id", str(task_id),
        "--task_name", task_name,
        "--question_type", qtype,
        "--mode", mode,
        "--template_key", template_key,
        "--lang", lang,
    ]

    if max_items > 0:
        cmd.extend(["--max_items", str(max_items)])
    if save_prompts:
        cmd.append("--save_prompts")
    
    cmd += ["--count", str(count)]

    if question_ids:
        cmd.append("--question_ids")
        cmd.extend([str(x) for x in question_ids])
    if question_ids_file:
        cmd.extend(["--question_ids_file", str(question_ids_file)])
    if question_id_regex:
        cmd.extend(["--question_id_regex", str(question_id_regex)])

    print(f"\n{'='*60}\n[RUN] {benchmark_file}\n{'='*60}")
    print(f"[META] task_id={task_id} task_name={task_name} subtype={subtype} qtype={qtype} mode={mode} template_key={template_key}")
    if question_ids or question_ids_file or question_id_regex:
        print(f"[FILTER] question_ids={question_ids} file={question_ids_file} regex={question_id_regex}")
    print("Command:", " ".join(cmd))
    sys.stdout.flush()

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        bufsize=1
    )
    assert p.stdout is not None
    out_lines = []
    for line in p.stdout:
        print(line, end="", flush=True)
        out_lines.append(line)
    p.wait()
    return p.returncode, "".join(out_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=["all"],
                        help="groups: all, task1..task8, task1_type1, task1_type2, task8_type1, task8_type2 OR explicit *.jsonl")
    parser.add_argument("--language", type=str, default="ko", choices=["ko", "en"])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--device_map", type=str, default="cuda", choices=["cuda", "auto"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--max_items", type=int, default=-1)
    parser.add_argument("--save_prompts", action="store_true")

    parser.add_argument("--mode", type=str, default="image", choices=["image", "tags"],
                        help="evaluation mode: 'image' uses images in prompts, 'tags' uses tags in prompts")
    parser.add_argument("--template_key", type=str, default=None, choices=[None, "Image", "Tags"],
                        help="prompt template key to use; if None, use 'Image' for mode='image' and 'Tags' for mode='tags'")
    parser.add_argument("--no_log", action="store_true", help="disable logging to file")

    parser.add_argument("--question_ids", nargs="*", default=None,
                        help="Run only these Question_ID(s). Example: --question_ids QID1 QID2")
    parser.add_argument("--question_ids_file", type=str, default=None,
                        help="Text file containing Question_IDs (newline or comma separated).")
    parser.add_argument("--question_id_regex", type=str, default=None,
                        help="Regex to match Question_ID. Example: --question_id_regex '^65d6'")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--count", type=int, default=-1)


    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    ts = kst_timestamp()
    log_file = None
    original_stdout = sys.stdout


    print("-"*40, "\n[Benchmarks] ", args.benchmarks, args.language, "\n", "-"*40)

    if not args.no_log:
        log_path = LOGS_DIR / f"run_qwen3_vl_{ts}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        sys.stdout = TeeWriter(original_stdout, log_file)
        print(f"[LOG] Logging to: {log_path}")

    try:
        benchmarks = get_benchmarks_to_run(args.benchmarks, args.language)

        if not benchmarks:
            print(f"[ERROR] No benchmarks found to run. benchmark_dir={BENCHMARK_DIR}")
            print("Tip: check benchmark folder exists and contains *.jsonl")
            return

        if args.template_key is None:
            template_key = "Image" if args.mode == "image" else "Tags"
        else:
            template_key = args.template_key

        model_ids: List[str] = []
        cache_dir: List[str] = []

        if args.models:
            for a in args.models:
                if a in MODEL_CONFIGS:
                    model_ids.append(MODEL_CONFIGS[a]["model_id"])
                    cache_dir.append(MODEL_CONFIGS[a]["cache_dir"])

                else:
                    print(f"[WARNING] Unknown model alias: {a}")
        elif args.model_id:
            model_ids = [args.model_id]
            cache_dir = [args.cache_dir]
        else:
            model_ids = [MODEL_CONFIGS["qwen3-4b"]["model_id"]]
            cache_dir = [MODEL_CONFIGS["qwen3-4b"]["cache_dir"]]



        print(f"\n{'#'*60}\n# Runner @ {ts}\n# Models: {model_ids}\n# Benchmarks: {len(benchmarks)}\n# mode={args.mode} template_key={template_key} language={args.language}\n{'#'*60}")
        if args.question_ids or args.question_ids_file or args.question_id_regex:
            print(f"# Question_ID filter: ids={args.question_ids} file={args.question_ids_file} regex={args.question_id_regex}")
            print(f"{'#'*60}")

        lang_benchmark_dir = BENCHMARK_DIR / args.language
        for mid, c_dir in zip(model_ids, cache_dir):
            print(f"model_save_dir: {c_dir}")
            for bf in benchmarks:
                if (lang_benchmark_dir / bf).exists():
                    pass
                else:
                    print(f"[WARN] benchmark file not found under benchmark/{args.language}/: {lang_benchmark_dir / bf} (will try as-is)")

                code, _ = run_one(
                    model_id=mid,
                    benchmark_file=bf,
                    device_map=args.device_map,
                    seed=args.seed,
                    shuffle_seed=args.shuffle_seed,
                    max_items=args.max_items,
                    save_prompts=args.save_prompts,
                    mode=args.mode,
                    template_key=template_key,
                    lang=args.language,
                    # pass through filters
                    question_ids=args.question_ids,
                    question_ids_file=args.question_ids_file,
                    question_id_regex=args.question_id_regex,
                    cache_dir=c_dir,
                    count=args.count,
                )
                if code != 0:
                    print(f"[FAILED] {mid} / {bf} (returncode={code})")

        print(f"\n{'#'*60}\n# Finished @ {kst_timestamp()}\n{'#'*60}")

    finally:
        if log_file:
            sys.stdout = original_stdout
            log_file.close()
            print(f"[LOG] Log saved to: {log_path}")


if __name__ == "__main__":
    main()
