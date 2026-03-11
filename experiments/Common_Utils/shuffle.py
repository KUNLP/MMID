#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
from dataclasses import dataclass
from typing import Any, Dict, List

LETTER_LIST = ["A", "B", "C", "D"]

@dataclass
class ShuffledOptions:
    perm: List[int]
    letter_to_option_id: Dict[str, str]
    option_id_to_letter: Dict[str, str]
    correct_letter: str
    correct_option_id: str

def build_shuffle_mapping(
    options: List[Dict[str, Any]],
    output_value: Any,
    rng: random.Random,
) -> ShuffledOptions:
    if len(options) != 4:
        raise ValueError(f"|Options| != 4: {len(options)}")

    correct_option_id = str(output_value)

    correct_idx = None
    for i, opt in enumerate(options):
        if str(opt.get("Option_ID")) == correct_option_id:
            correct_idx = i
            break

    if correct_idx is None:
        # fallback: output_value를 index로 해석 시도
        try:
            correct_idx = int(output_value)
            correct_option_id = str(options[correct_idx].get("Option_ID", output_value))
        except Exception as e:
            raise ValueError(f"The correct option could not be found. Output={output_value}") from e

    perm = list(range(4))
    rng.shuffle(perm)

    letter_to_option_id: Dict[str, str] = {}
    option_id_to_letter: Dict[str, str] = {}
    correct_letter = None

    for pos, orig_i in enumerate(perm):
        letter = LETTER_LIST[pos]
        opt_id = str(options[orig_i].get("Option_ID", orig_i))
        letter_to_option_id[letter] = opt_id
        option_id_to_letter[opt_id] = letter
        if orig_i == correct_idx:
            correct_letter = letter

    if correct_letter is None:
        raise RuntimeError("[Logic Error] Failed to determine correct_letter.")

    return ShuffledOptions(
        perm=perm,
        letter_to_option_id=letter_to_option_id,
        option_id_to_letter=option_id_to_letter,
        correct_letter=correct_letter,
        correct_option_id=correct_option_id,
    )
