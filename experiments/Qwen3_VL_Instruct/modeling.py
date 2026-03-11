#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional, List
from pathlib import Path
import json
import time
import os

import torch
from transformers import AutoConfig, AutoProcessor, GenerationConfig
from transformers import Qwen3VLForConditionalGeneration


def load_qwen3vl(
    model_id: str,
    device_map: Any,
    *,
    # ✅ NEW: image token budget control
    min_pixels: Optional[str] = 448,
    max_pixels: Optional[str] = 448
):
    """
    [Prevent OOM] Passing min_pixels/max_pixels to the processor sets an internal upper bound on image resizing and tokenization.
    """
    proc_kwargs: Dict[str, Any] = dict(trust_remote_code=True, use_fast=True)
    if min_pixels is not None:
        proc_kwargs["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        proc_kwargs["max_pixels"] = int(max_pixels)

    
    cache_dir = os.path.join("../models/", model_id.split("/")[-1])
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True, **proc_kwargs)

    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True, trust_remote_code=True)
    if hasattr(config, "text_config") and not hasattr(config.text_config, "pad_token_id"):
        config.text_config.pad_token_id = getattr(config.text_config, "eos_token_id", 151643)

    # Turing (7.5) or lower: bf16 not supported → force fp16
    dtype = torch.float16
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        # Ampere (8.0) or higher: bf16 supported
        if major >= 8:
            dtype = torch.bfloat16

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        cache_dir=cache_dir, local_files_only=True, 
    ).eval()

    return model, processor


def load_generation_config(
    model_id: str,
    generation_config_path: Optional[str],
    overrides: Optional[Dict[str, Any]] = None,
) -> GenerationConfig:
    if generation_config_path:
        p = Path(generation_config_path)
        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            gen_cfg = GenerationConfig.from_dict(data)
        else:
            gen_cfg = GenerationConfig.from_pretrained(str(p) if p.exists() else generation_config_path)
    else:
        gen_cfg = GenerationConfig.from_pretrained(model_id)

    if overrides:
        for k, v in overrides.items():
            if v is None:
                continue
            setattr(gen_cfg, k, v)

    if not hasattr(gen_cfg, "use_cache") or gen_cfg.use_cache is None:
        gen_cfg.use_cache = True

    return gen_cfg


def _move_inputs_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    return inputs


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def run_generate(
    model: Any,
    processor: Any,
    messages: Any,
    generation_config: GenerationConfig,
    max_retries: int = 1,
    print_step_timing: bool = False,
) -> str:
    last_err = None
    for _ in range(max_retries + 1):
        try:
            _sync_cuda()
            t0 = time.perf_counter()

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            _sync_cuda()
            t1 = time.perf_counter()

            device = next(model.parameters()).device
            inputs = _move_inputs_to_device(inputs, device)

            _sync_cuda()
            t2 = time.perf_counter()

            generated_ids = model.generate(**inputs, generation_config=generation_config)

            _sync_cuda()
            t3 = time.perf_counter()

            in_len = inputs["input_ids"].shape[1]
            gen_only = generated_ids[:, in_len:]

            out_text = processor.batch_decode(
                gen_only,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            _sync_cuda()
            t4 = time.perf_counter()

            if print_step_timing:
                print(
                    f"[STEP] chat_template={t1-t0:.3f}s "
                    f"to_device={t2-t1:.3f}s "
                    f"generate={t3-t2:.3f}s "
                    f"decode={t4-t3:.3f}s "
                    f"total={t4-t0:.3f}s"
                )

            return out_text

        except Exception as e:
            last_err = e

    raise RuntimeError(f"[Fail] Generation Fail: {last_err}") from last_err


@torch.inference_mode()
def run_generate_batch(
    model: Any,
    processor: Any,
    batch_messages: List[Any],
    generation_config: GenerationConfig,
    max_retries: int = 1,
    print_step_timing: bool = False,
) -> List[str]:
    if not batch_messages:
        return []

    device = next(model.parameters()).device
    last_err = None

    for _ in range(max_retries + 1):
        try:
            _sync_cuda()
            t0 = time.perf_counter()

            try:
                inputs = processor.apply_chat_template(
                    batch_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                )
            except TypeError as e:
                raise TypeError(f"batch_apply_chat_template_not_supported: {e}") from e

            _sync_cuda()
            t1 = time.perf_counter()

            inputs = _move_inputs_to_device(inputs, device)

            _sync_cuda()
            t2 = time.perf_counter()

            generated_ids = model.generate(**inputs, generation_config=generation_config)

            _sync_cuda()
            t3 = time.perf_counter()

            in_len = inputs["input_ids"].shape[1]
            gen_only = generated_ids[:, in_len:]

            out_texts = processor.batch_decode(
                gen_only,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            _sync_cuda()
            t4 = time.perf_counter()

            if print_step_timing:
                print(
                    f"[BSTEP] chat_template={t1-t0:.3f}s "
                    f"to_device={t2-t1:.3f}s "
                    f"generate={t3-t2:.3f}s "
                    f"decode={t4-t3:.3f}s "
                    f"total={t4-t0:.3f}s "
                    f"(bs={len(batch_messages)})"
                )

            return list(out_texts)

        except TypeError:
            outs: List[str] = []
            for msgs in batch_messages:
                outs.append(
                    run_generate(
                        model=model,
                        processor=processor,
                        messages=msgs,
                        generation_config=generation_config,
                        max_retries=max_retries,
                        print_step_timing=print_step_timing,
                    )
                )
            return outs

        except Exception as e:
            last_err = e

    raise RuntimeError(f"[Fail] Batch Generation Fail: {last_err}") from last_err
