#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
from typing import List, Optional, Tuple

LETTER_LIST = ["A", "B", "C", "D"]

LETTER_RE = re.compile(r"(?:정답\s*[:：]?\s*)?([ABCD])", re.IGNORECASE)
YES_NO_RE = re.compile(r"(?:정답\s*[:：]?\s*)?(YES|NO)", re.IGNORECASE)
BBOX_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def parse_answer_abcd(model_text: str) -> Optional[str]:
    if not model_text:
        return None
    m = LETTER_RE.search(model_text.strip())
    if m:
        return m.group(1).upper()
    m2 = re.search(r"\b([ABCD])\b", model_text.strip(), re.IGNORECASE)
    return m2.group(1).upper() if m2 else None


def parse_answer_yn(model_text: str) -> Optional[str]:
    if not model_text:
        return None
    m = YES_NO_RE.search(model_text.strip())
    if m:
        return m.group(1).upper()
    m2 = re.search(r"\b(YES|NO)\b", model_text.strip(), re.IGNORECASE)
    return m2.group(1).upper() if m2 else None


def parse_bbox_4nums(text: str) -> Optional[List[float]]:
    if not text:
        return None

    bracket_blocks = re.findall(r"\[([^\]]+)\]", text)
    for blk in reversed(bracket_blocks):
        nums = BBOX_NUM_RE.findall(blk)
        if len(nums) >= 4:
            return list(map(float, nums[:4]))

    nums = BBOX_NUM_RE.findall(text)
    if len(nums) < 4:
        return None
    return list(map(float, nums[:4]))


def parse_answer_xywh(model_text: str) -> Optional[Tuple[float, float, float, float]]:
    vals = parse_bbox_4nums(model_text)
    if vals and len(vals) >= 4:
        return (vals[0], vals[1], vals[2], vals[3])
    return None
