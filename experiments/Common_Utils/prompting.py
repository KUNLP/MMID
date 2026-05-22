#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Optional

# 은/는 구별: grammer_1
# 이/가 구별: grammer_2
# 을/를 구별: grammer_3
def has_jongseong(word: str) -> bool:
    """
    Determine whether the last syllable of the word has a final consonant.
    """
    if not word:
        return False

    ch = word[-1]
    if ch == "1": ch="일"
    elif ch =="2": ch="이"
    code = ord(ch)

    # 한글 음절 범위
    if not (0xAC00 <= code <= 0xD7A3):
        return False

    jongseong_index = (code - 0xAC00) % 28
    return jongseong_index != 0


def grammer_1(word: str) -> str:
    has_final = has_jongseong(word)
    return "은" if has_final else "는"
def grammer_2(word: str) -> str:
    has_final = has_jongseong(word)
    return "이" if has_final else "가"
def grammer_3(word: str) -> str:
    has_final = has_jongseong(word)
    return "을" if has_final else "를"

def render_prompt(
    template: str,
    conversation_text: str,
    user_name: str,
    image_url_list: Optional[List[str]] = None,
    tag_list: Optional[List[str]] = None,
    text_list: Optional[List[str]] = None,
) -> str:
    if image_url_list is None:
        image_url_list = [""] * 4
    if tag_list is None:
        tag_list = [""] * 4
    if text_list is None:
        text_list = [""] * 4

    s = template
    s = s.replace("{Conversation}", conversation_text)
    s = s.replace("{conversation}", conversation_text)

    s = s.replace("{user}", user_name)
    s = s.replace("{grammer_1(user)}", grammer_1(user_name))
    s = s.replace("{grammer_2(user)}", grammer_2(user_name))

    for i in range(4):
        if i < len(image_url_list):
            s = s.replace(f"{{image_url_list[{i}]}}", image_url_list[i] or "")
        if i < len(tag_list):
            s = s.replace(f"{{tag_list[{i}]}}", tag_list[i] or "")
        if i < len(text_list):
            s = s.replace(f"{{text_list[{i}]}}", text_list[i] or "")
    return s

def extract_question_part(full_prompt: str) -> str:
    if not full_prompt:
        return ""
    for marker in ("Question:", "질문:"):
        idx = full_prompt.find(marker)
        if idx >= 0:
            return full_prompt[idx:].strip()
    return full_prompt.strip()
