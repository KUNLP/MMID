#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import re
import time
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Set

from tqdm.auto import tqdm

from ..Common_Utils.common import ensure_dir, kst_timestamp, seed_everything, resolve_device_map
from ..Common_Utils.io_utils import load_dataset, load_processed_indices
from ..Common_Utils.shuffle import build_shuffle_mapping
from ..Common_Utils.image_cache import url_to_cached_path  
from .modeling import load_qwen3vl, load_generation_config, run_generate_batch
from .task_logic import prepare_options_lists, build_messages_dispatch, score_prediction


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # data & output
    p.add_argument("--data_root_path", type=str, default="benchmark")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_root", type=str, default="results")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--max_items", type=int, default=-1)
    p.add_argument("--save_prompts", action="store_true")

    p.add_argument(
        "--question_ids",
        nargs="*",
        default=None,
        help="Run only these Question_ID(s). Example: --question_ids QID1 QID2",
    )
    p.add_argument(
        "--question_ids_file",
        type=str,
        default=None,
        help="Text file containing Question_IDs (newline or comma separated).",
    )
    p.add_argument(
        "--question_id_regex",
        type=str,
        default=None,
        help="Regex to match Question_ID. Example: --question_id_regex '^65d6.*'",
    )

    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--device_map", type=str, default="auto", choices=["auto", "cuda"])

    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--shuffle_seed", type=int, default=42)

    p.add_argument("--generation_config_path", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--repetition_penalty", type=float, default=None)

    p.add_argument("--batch_size", type=int, default=1)

    p.add_argument("--task_id", type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--task_name", type=str, default=None,
                choices=["task1", "task2", "task3", "task4", "task5", "task6", "task7", "task8"])

    # Task1 subtype filter (Type1/Type2)
    p.add_argument("--task_subtype_key", type=str, default="Task_Type")
    p.add_argument("--task_subtype_value", type=str, default=None, choices=["1", "2"])

    # mode / question type
    p.add_argument("--mode", type=str, default="image", choices=["image", "tags"])
    p.add_argument("--question_type", type=str, required=True, choices=["multiple_choice", "yes_or_no", "open"])

    # template key selector
    p.add_argument("--template_key", type=str, default="Image", choices=["Image", "Tags"])

    # benchmark language
    p.add_argument("--lang", type=str, default="ko", choices=["ko", "en"],
                   help="Benchmark language: 'ko' for Korean benchmark, 'en' for English benchmark")

    p.add_argument("--speed_debug", action="store_true",
                  help="print timing breakdown (build vs generate vs write)")
    p.add_argument("--log_every", type=int, default=10,
                  help="print debug every N processed samples (not skipped)")
    p.add_argument("--show_n_images", action="store_true",
                  help="print number of images per sample in messages")
    p.add_argument("--print_step_timing", action="store_true",
                  help="print step timing inside run_generate(_batch)")
    p.add_argument("--localize_images", action="store_true",
                  help="replace http image refs in messages with local cached file path")

    p.add_argument(
        "--count",
        type=int,
        default=1,
        help="number of count to do experminents",
    )

    return p


def _item_matches_subtype(item: Dict[str, Any], key: Optional[str], value: Optional[str]) -> bool:
    if not key or value is None:
        return True
    v = item.get(key, None)
    if v is None:
        return False
    return str(v) == str(value)


def _infer_task_name(task_id: int) -> str:
    return {1:"task1",2:"task2",3:"task3",4:"task4",5:"task5",6:"task6",7:"task7",8:"task8"}[int(task_id)]


def _short(s: Any, n: int = 40) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _count_images_in_messages(messages: Any) -> int:
    n_img = 0
    try:
        if isinstance(messages, dict) and "images" in messages:
            n_img = len(messages.get("images") or [])
        elif isinstance(messages, list):
            for m in messages:
                c = m.get("content", None)
                if isinstance(c, list):
                    for x in c:
                        if isinstance(x, dict) and x.get("type") in ("image", "image_url"):
                            n_img += 1
    except Exception:
        pass
    return n_img


def _localize_images_in_messages(messages: Any) -> Any:
    """
    Replace image references in messages that use HTTP URLs with local file paths using url_to_cached_path.
    """
    def _rewrite(obj: Any) -> Any:
        if isinstance(obj, dict):
            newd = {}
            for k, v in obj.items():
                if isinstance(v, str) and v.startswith("http") and k in ("image", "url", "image_url"):
                    newd[k] = url_to_cached_path(v)
                else:
                    newd[k] = _rewrite(v)
            return newd
        if isinstance(obj, list):
            return [_rewrite(x) for x in obj]
        return obj

    return _rewrite(messages)


def _load_question_id_filters(args: argparse.Namespace) -> Tuple[Optional[Set[str]], Optional[re.Pattern]]:
    """
    Return (qid_set, qid_regex).
    Both can be active; filtering is AND.
    """
    qid_set: Optional[Set[str]] = None
    qid_regex: Optional[re.Pattern] = None

    if args.question_ids:
        qid_set = set(str(x).strip() for x in args.question_ids if str(x).strip())

    if args.question_ids_file:
        p = Path(args.question_ids_file)
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="ignore")
            raw = []
            for line in text.splitlines():
                raw.extend([x.strip() for x in line.split(",")])
            file_ids = [x for x in raw if x]
            if file_ids:
                if qid_set is None:
                    qid_set = set(file_ids)
                else:
                    qid_set |= set(file_ids)

    if args.question_id_regex:
        qid_regex = re.compile(args.question_id_regex)

    return qid_set, qid_regex


def _qid_pass(qid: Any, qid_set: Optional[Set[str]], qid_regex: Optional[re.Pattern]) -> bool:
    if qid is None:
        return False if (qid_set or qid_regex) else True
    qid_str = str(qid)
    if qid_set is not None and qid_str not in qid_set:
        return False
    if qid_regex is not None and not qid_regex.search(qid_str):
        return False
    return True


def run_eval(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    qid_set, qid_regex = _load_question_id_filters(args)

    # timing accumulators
    t_build = 0.0
    t_gen = 0.0
    t_write = 0.0
    n_build = 0
    n_proc = 0  
    min_pixels = 448
    max_pixels = 640
    
    if args.task_name is None:
        args.task_name = _infer_task_name(args.task_id)

    # load model
    device_map = resolve_device_map(args.device_map)
    model, processor = load_qwen3vl(args.model_id, device_map=device_map, min_pixels=min_pixels, max_pixels=max_pixels)

    overrides: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(args.do_sample),
    }
    if args.do_sample:
        overrides["temperature"] = float(args.temperature)
    if args.top_p is not None:
        overrides["top_p"] = float(args.top_p)
    if args.top_k is not None:
        overrides["top_k"] = int(args.top_k)
    if args.repetition_penalty is not None:
        overrides["repetition_penalty"] = float(args.repetition_penalty)

    gen_cfg = load_generation_config(
        model_id=args.model_id,
        generation_config_path=args.generation_config_path,
        overrides=overrides,
    )

    data_path = Path(args.data_root_path) / Path(args.data_path)
    dataset = load_dataset(data_path)
    if args.max_items > 0:
        dataset = dataset[: args.max_items]

    processed_indices = set()
    file_mode = "w"

    ts = kst_timestamp()
    out_dir = Path(args.output_root) / Path(args.model_id) / Path(str(args.count)) / Path(args.data_path) / f"{args.task_name}_{args.mode}_{ts}"
    out_dir = Path(str(out_dir).replace("../models", ""))
    ensure_dir(out_dir)

    if args.resume:
        rp = Path(args.resume)
        if rp.exists():
            processed_indices = load_processed_indices(rp)
            pred_path = rp
            summary_path = rp.parent / f"summary_{args.task_name}_{args.mode}_{ts}.json"
            file_mode = "a"
        else:
            pred_path = out_dir / f"predictions_{args.task_name}_{args.mode}_{ts}.jsonl"
            summary_path = out_dir / f"summary_{args.task_name}_{args.mode}_{ts}.json"
    else:
        pred_path = out_dir / f"predictions_{args.task_name}_{args.mode}_{ts}.jsonl"
        summary_path = out_dir / f"summary_{args.task_name}_{args.mode}_{ts}.json"

    total = 0
    correct = 0
    skipped = 0
    last_info: Dict[str, Any] = {}

    # Buffer Batch
    batch_recs: List[Dict[str, Any]] = []
    batch_msgs: List[Any] = []
    batch_metas: List[Dict[str, Any]] = []

    def flush_batch(wf):
        nonlocal total, correct, batch_recs, batch_msgs, batch_metas, last_info
        nonlocal t_gen, t_write, n_proc

        if not batch_msgs:
            return

        tg0 = time.perf_counter()
        responses = run_generate_batch(
            model=model,
            processor=processor,
            batch_messages=batch_msgs,
            generation_config=gen_cfg,
            max_retries=1,
            print_step_timing=bool(args.print_step_timing),
        )
        tg1 = time.perf_counter()
        t_gen += (tg1 - tg0)

        tw0 = time.perf_counter()
        for rec_i, meta_i, resp_i in zip(batch_recs, batch_metas, responses):
            scored = score_prediction(
                task_id=meta_i["task_id"],
                question_type=meta_i["question_type"],
                response_text=resp_i,
                gt_output=meta_i["gt_output"],
                correct_letter=meta_i["correct_letter"],
            )
            rec_i.update(scored)

            total += 1
            n_proc += 1
            if rec_i.get("is_correct") is True:
                correct += 1

            last_info = {
                "idx": rec_i.get("idx"),
                "qid": rec_i.get("Question_ID"),
                "gt": rec_i.get("gt_answer", ""),
                "pred": rec_i.get("pred_text", resp_i),
                "ok": rec_i.get("is_correct"),
            }

            wf.write(json.dumps(rec_i, ensure_ascii=False) + "\n")

        wf.flush()
        tw1 = time.perf_counter()
        t_write += (tw1 - tw0)

        # Debug after batch processing 
        if args.speed_debug and (n_proc % args.log_every == 0):
            avg_build = (t_build / n_proc) if n_proc else 0.0
            avg_gen = (t_gen / n_proc) if n_proc else 0.0
            avg_write = (t_write / n_proc) if n_proc else 0.0
            print(f"[SPEED] n_proc={n_proc} avg_build={avg_build:.3f}s avg_gen={avg_gen:.3f}s avg_write={avg_write:.3f}s")

        batch_recs.clear()
        batch_msgs.clear()
        batch_metas.clear()

    with pred_path.open(file_mode, encoding="utf-8") as wf:
        pbar = tqdm(total=len(dataset), desc=f"Eval {args.task_name}", unit="sample", dynamic_ncols=True)

        for idx, item in enumerate(dataset):
            pbar.update(1)

            if idx in processed_indices:
                skipped += 1
                continue

            qid = item.get("Question_ID")
            if not _qid_pass(qid, qid_set, qid_regex):
                skipped += 1
                continue

            if int(item.get("Task_ID", -1)) != int(args.task_id):
                skipped += 1
                continue

            if not _item_matches_subtype(item, args.task_subtype_key, args.task_subtype_value):
                skipped += 1
                continue

            rng = random.Random(args.shuffle_seed + idx)

            prompt_tpl = item.get("Prompt_Template", {}) or {}
            if args.template_key not in prompt_tpl:
                skipped += 1
                continue

            options = item.get("Options", []) or []
            output_value = item.get("Output")
            question_type = item.get("Question_Type")

            if args.question_type != "open":
                if question_type != args.question_type:
                    skipped += 1
                    continue

            rec: Dict[str, Any] = {
                "idx": idx,
                "Question_ID": item.get("Question_ID"),
                "Task_ID": item.get("Task_ID"),
                "task_name": args.task_name,
                "mode": args.mode,
                "template_key": args.template_key,
                "question_type": question_type,
            }

            
            # ---- shuffle (ALL TASKS) ----
            correct_letter = None
            shuffled_options = options

 
            try:
                # ---- shuffle & GT record (ALL TASKS) ----
                if question_type == "multiple_choice":
                    if len(options) != 4:
                        raise ValueError(f"options_error: len(options)={len(options)} (expected 4)")

                    shuffle_map = build_shuffle_mapping(options, output_value, rng=rng)
                    shuffled_options = [options[i] for i in shuffle_map.perm]
                    correct_letter = shuffle_map.correct_letter

                    rec.update({
                        "gt_answer": correct_letter,                 # A/B/C/D
                        "gt_output": output_value,                   # GT
                        "gt_option_id": shuffle_map.correct_option_id,
                        "letter_to_option_id": shuffle_map.letter_to_option_id,
                        "shuffle_perm": shuffle_map.perm,
                    })

                elif question_type == "yes_or_no":
                    rec["gt_answer"] = str(output_value).upper()
                    rec["gt_output"] = output_value

                else:  # open
                    rec["gt_output"] = output_value

                # ---- prepare lists ----
                image_url_list, tag_list, text_list = [], [], []
                if shuffled_options:
                    image_url_list, tag_list, text_list = prepare_options_lists(shuffled_options)

                # ---- build messages ----
                tb0 = time.perf_counter()

                kw = dict(
                    item=item,
                    task_id=int(item.get("Task_ID")),
                    question_type=("open" if int(item.get("Task_ID")) == 4 else question_type),
                    mode=args.mode,
                    template_key=args.template_key,
                    image_url_list=image_url_list,
                    tag_list=tag_list,
                    text_list=text_list,
                    lang=args.lang,
                )
                sig = inspect.signature(build_messages_dispatch)
                kw = {k: v for k, v in kw.items() if k in sig.parameters}

                messages, modality_used = build_messages_dispatch(**kw)

                if args.localize_images:
                    messages = _localize_images_in_messages(messages)

                tb1 = time.perf_counter()
                t_build += (tb1 - tb0)
                n_build += 1

                if args.show_n_images:
                    n_img = _count_images_in_messages(messages)
                    print(f"[DEBUG] idx={idx} qid={qid} modality={modality_used} n_images_in_prompt={n_img}")

                rec["modality"] = modality_used

                if args.save_prompts:
                    rec["messages"] = messages
                    rec["image_urls_abcd"] = image_url_list
                    rec["tag_list_abcd"] = tag_list
                    rec["text_list_abcd"] = text_list

                # ---- batch enqueue ----
                batch_recs.append(rec)
                batch_msgs.append(messages)
                batch_metas.append({
                    "task_id": int(item.get("Task_ID")),
                    "question_type": ("open" if int(item.get("Task_ID")) == 4 else question_type),
                    "gt_output": output_value,
                    "correct_letter": correct_letter,
                })

                if len(batch_msgs) >= int(args.batch_size):
                    flush_batch(wf)

            except Exception as e:
                rec["error"] = f"inference_error: {repr(e)}"
                last_info = {"idx": idx, "qid": qid, "err": _short(repr(e), 70)}
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wf.flush()

            # Update postfix 
            postfix = {
                "acc": f"{correct}/{total}" if total else "0/0",
                "skipped": skipped,
                "bs": args.batch_size,
            }
            if last_info:
                if "err" in last_info:
                    postfix.update({"last_idx": last_info.get("idx"), "qid": last_info.get("qid"), "err": last_info.get("err")})
                else:
                    postfix.update({
                        "last_idx": last_info.get("idx"),
                        "qid": last_info.get("qid"),
                        "gt": _short(last_info.get("gt")),
                        "pred": _short(last_info.get("pred")),
                        "ok": last_info.get("ok"),
                    })
            pbar.set_postfix(postfix)

        # Process the remaining batch after the loop ends
        if batch_msgs:
            flush_batch(wf)
            pbar.set_postfix({
                "acc": f"{correct}/{total}" if total else "0/0",
                "skipped": skipped,
                "bs": args.batch_size,
            })

    summary: Dict[str, Any] = {
        "timestamp_kst": ts,
        "data_path": str(data_path),
        "model_id": args.model_id,
        "task_name": args.task_name,
        "task_id": args.task_id,
        "task_subtype_key": args.task_subtype_key,
        "task_subtype_value": args.task_subtype_value,
        "mode": args.mode,
        "template_key": args.template_key,
        "question_type": args.question_type,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "predictions_file": str(pred_path),
        "total": total,
        "correct": correct,
        "skipped": skipped,
        "accuracy": (correct / total) if total else None,
        "question_id_filter": {
            "question_ids": sorted(list(qid_set)) if qid_set else None,
            "question_id_regex": args.question_id_regex,
            "question_ids_file": args.question_ids_file,
        },
    }

    if args.speed_debug:
        summary["timing"] = {
            "t_build_sec": t_build,
            "t_gen_sec": t_gen,
            "t_write_sec": t_write,
            "n_build": n_build,
            "n_proc": n_proc,
            "avg_build_sec": (t_build / n_proc) if n_proc else None,
            "avg_gen_sec": (t_gen / n_proc) if n_proc else None,
            "avg_write_sec": (t_write / n_proc) if n_proc else None,
        }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    args = make_argparser().parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
