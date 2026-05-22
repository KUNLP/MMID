#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any, Dict, List

from ..Common_Utils.image_cache import url_to_cached_path

LETTER_LIST = ["A", "B", "C", "D"]


# =============================================================================
# Language-specific section labels
# =============================================================================
_LABELS = {
    "ko": {
        "options":               "[보기]",
        "outfit_info":           "[의상 정보]",
        "utterance":             "[발화]",
        "dialogue_segments":     "[대화 단락]",
        "initial_dialogue":      "[초기 대화]",
        "order_of_outfit":       "[대화 내 의상 정보 등장 순서]",
        "order_of_outfit_image": "[대화 내 의상 이미지 등장 순서]",
        "last_segment_image":    "[마지막 단락의 의상 이미지]",
        "last_segment_info":     "[마지막 단락의 의상 정보]",
        "task8_type1_intro":     "아래 [대화 단락]은 user와 assistant 간의 대화를 무작위로 섞은 단락 집합이다.\n",
        "task8_type2_intro":     "아래 [초기 대화]는 user와 assistant 간의 첫 대화 내용이고, [대화 단락]은 [초기 대화] 이후 대화를 무작위로 섞은 단락 집합이다.\n",
        "conversation_intro":    "아래 [대화]는 user와 assistant 간의 대화 내용이다.\n",
        "conversation":          "[대화]",
        "image_label":           "이미지",
        "outfit_info_label":     "의상 정보",
    },
    "en": {
        "options":               "[Options]",
        "outfit_info":           "[Outfit Information]",
        "utterance":             "[Utterance]",
        "dialogue_segments":     "[Dialogue Segments]",
        "initial_dialogue":      "[Initial Dialogue]",
        "order_of_outfit":       "[Order of Outfit Information Appearance in Dialogue]",
        "order_of_outfit_image": "[Order of Outfit Image Appearance in Dialogue]",
        "last_segment_image":    "[Outfit Image of the Last Segment]",
        "last_segment_info":     "[Outfit Information of the Last Segment]",
        "task8_type1_intro":     "The following [Dialogue Segments] are a set of segments with the dialogue randomly shuffled.\n",
        "task8_type2_intro":     "The following [Initial Dialogue] is the first conversation between user and assistant, and [Dialogue Segments] are a set of segments with the subsequent dialogue randomly shuffled.\n",
        "conversation_intro":    "The following [Dialogue] is a conversation between user and assistant.\n",
        "conversation":          "[Dialogue]",
        "image_label":           "Image",
        "outfit_info_label":     "Outfit Information",
    },
}


def _L(lang: str, key: str) -> str:
    """언어별 섹션 레이블 반환. 미지원 lang은 ko로 fallback."""
    return _LABELS.get(lang, _LABELS["ko"])[key]


# -----------------------------
# Base message helpers
# -----------------------------
def _as_user_text(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _as_user_image(image_url: str) -> Dict[str, Any]:
    image_url = url_to_cached_path(image_url)
    return {"role": "user", "content": [{"type": "image", "image": image_url}]}


def _text_item(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def _image_item(image_url: str) -> Dict[str, Any]:
    image_url = url_to_cached_path(image_url)
    return {"type": "image", "image": image_url}


def _as_user_content_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"role": "user", "content": items}


def _safe_line(s: Any) -> str:
    return str(s)


def _append_text_as_lines(messages: List[Dict[str, Any]], text: str, *, suffix_newline: bool = True) -> None:
    if text is None:
        return
    s = str(text)
    for line in s.splitlines():
        messages.append(_as_user_text(line + ("\n" if suffix_newline else "")))


def _append_turns_as_lines(
    messages: List[Dict[str, Any]],
    turns: Any,
    *,
    include_turn_image_url_as_text: bool = True,
) -> None:
    if turns is None:
        return

    if isinstance(turns, str):
        _append_text_as_lines(messages, turns)
        return

    if isinstance(turns, list):
        for t in turns:
            if not isinstance(t, dict):
                messages.append(_as_user_text(str(t) + "\n"))
                continue

            image_info = t.get("image_info") or {}
            img_url = None
            if isinstance(image_info, dict):
                img_url = image_info.get("Url") or image_info.get("url")

            if include_turn_image_url_as_text and img_url:
                messages.append(_as_user_text(str(img_url) + "\n"))

            role = t.get("role", "unknown")
            content = str(t.get("content", ""))
            messages.append(_as_user_text(f"'{role}': '{content}'\n"))
        return

    _append_text_as_lines(messages, str(turns))


def _append_conversation_lines(
    messages: List[Dict[str, Any]],
    conversation: List[Dict[str, Any]],
    *,
    include_turn_image: bool,
    include_turn_tags_as_text: bool,
    lang: str = "ko",
) -> None:
    messages.append(_as_user_text(_L(lang, "conversation_intro")))
    messages.append(_as_user_text(_L(lang, "conversation") + "\n"))

    for t in conversation:
        role = t.get("role", "unknown")
        content = _safe_line(t.get("content", ""))

        image_info = t.get("image_info") or {}
        img_url = None
        tags = None
        if isinstance(image_info, dict):
            img_url = image_info.get("Url") or image_info.get("url")
            tags = image_info.get("Tags") or image_info.get("tags")

        if include_turn_image and img_url:
            messages.append(_as_user_image(str(img_url)))

        if (not include_turn_image) and include_turn_tags_as_text and tags:
            if isinstance(tags, list):
                tag_txt = ", ".join(map(str, tags))
            else:
                tag_txt = str(tags)
            messages.append(_as_user_text(tag_txt + "\n"))

        messages.append(_as_user_text(f"'{role}': '{content}'\n"))


# -----------------------------
# Task3 / Task4 : MCQ
# -----------------------------
def build_messages_task3_3_mcq_image(
    conversation: List[Dict[str, Any]],
    option_image_urls_abcd: List[str],
    question_text: str,
    *,
    include_turn_image: bool = False,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(
        messages,
        conversation,
        include_turn_image=include_turn_image,
        include_turn_tags_as_text=False,
        lang=lang,
    )

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    for i, letter in enumerate(LETTER_LIST):
        messages.append(_as_user_text(f"{letter}: "))
        if i < len(option_image_urls_abcd) and option_image_urls_abcd[i]:
            messages.append(_as_user_image(option_image_urls_abcd[i]))
        else:
            messages.append(_as_user_text("(missing image)\n"))

    messages.append(_as_user_text("\n" + question_text))
    return messages


def build_messages_task3_3_mcq_tags(
    conversation: List[Dict[str, Any]],
    option_tags_abcd: List[str],
    question_text: str,
    *,
    include_turn_tags_as_text: bool = False,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(
        messages,
        conversation,
        include_turn_image=False,
        include_turn_tags_as_text=include_turn_tags_as_text,
        lang=lang,
    )

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    for i, letter in enumerate(LETTER_LIST):
        tag_txt = option_tags_abcd[i] if i < len(option_tags_abcd) else ""
        messages.append(_as_user_text(f"{letter}: {tag_txt}\n"))

    messages.append(_as_user_text("\n" + question_text))
    return messages


# -----------------------------
# Task3 : Yes/No
# -----------------------------
def build_messages_task3_yn_image(
    conversation: List[Dict[str, Any]],
    option_image_url: str,
    statement_text_before_image: str,
    question_text_after_view: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=False, include_turn_tags_as_text=False, lang=lang)

    items: List[Dict[str, Any]] = []
    items.append(_text_item(_L(lang, "options") + "\n" + (statement_text_before_image or "")))
    if option_image_url:
        items.append(_image_item(option_image_url))
    items.append(_text_item("\n\n" + (question_text_after_view or "")))

    messages.append(_as_user_content_items(items))
    return messages


def build_messages_task3_yn_tags(
    conversation: List[Dict[str, Any]],
    option_tags_text: str,
    view_text: str,
    question_text_after_view: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=False, include_turn_tags_as_text=False, lang=lang)

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    messages.append(_as_user_text(view_text))
    messages.append(_as_user_text("\n\n" + question_text_after_view))
    return messages


# -----------------------------
# Task2 : MCQ / YesNo
# -----------------------------
def build_messages_task2_mcq_image(
    tag_text: str,
    option_image_urls_abcd: List[str],
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    messages.append(_as_user_text(_L(lang, "outfit_info") + "\n"))
    messages.append(_as_user_text(tag_text.strip() + "\n\n"))

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    for i, letter in enumerate(LETTER_LIST):
        messages.append(_as_user_text(f"{letter}: "))
        if i < len(option_image_urls_abcd) and option_image_urls_abcd[i]:
            messages.append(_as_user_image(option_image_urls_abcd[i]))
        else:
            messages.append(_as_user_text("(missing image)\n"))

    messages.append(_as_user_text("\n" + question_text))
    return messages


def build_messages_task2_yn_image(
    tag_text: str,
    option_image_url: str,
    view_statement_before_image: str,
    question_text_after_view: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    messages.append(_as_user_text(_L(lang, "outfit_info") + "\n"))
    messages.append(_as_user_text(tag_text.strip() + "\n\n"))

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    messages.append(_as_user_text(view_statement_before_image or ""))
    if option_image_url:
        messages.append(_as_user_image(option_image_url))
    messages.append(_as_user_text("\n\n" + (question_text_after_view or "")))

    return messages


# -----------------------------
# Task4 : Yes/No (one-content)
# -----------------------------
def build_messages_task4_yn_image(
    conversation: List[Dict[str, Any]],
    view_text: str,
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=True, include_turn_tags_as_text=False, lang=lang)

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    messages.append(_as_user_text((view_text or "").strip()))
    messages.append(_as_user_text("\n\n" + (question_text or "")))
    return messages


def build_messages_task4_yn_tags(
    conversation: List[Dict[str, Any]],
    view_text: str,
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=False, include_turn_tags_as_text=True, lang=lang)

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    messages.append(_as_user_text((view_text or "").strip()))
    messages.append(_as_user_text("\n\n" + (question_text or "")))
    return messages


# -----------------------------
# Task1 : Open / Image
# -----------------------------
def build_messages_task1_open_image(
    conversation: List[Dict[str, Any]],
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(
        messages,
        conversation,
        include_turn_image=True,
        include_turn_tags_as_text=False,
        lang=lang,
    )
    messages.append(_as_user_text("\n\n" + question_text))
    return messages


# -----------------------------
# Task5 : MCQ / YesNo (image/tags)
# -----------------------------
def build_messages_task5_mcq_image(
    conversation: List[Dict[str, Any]],
    option_image_urls_abcd: List[str],
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=True, include_turn_tags_as_text=False, lang=lang)

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    for i, letter in enumerate(LETTER_LIST):
        messages.append(_as_user_text(f"{letter}: "))
        if i < len(option_image_urls_abcd) and option_image_urls_abcd[i]:
            messages.append(_as_user_image(option_image_urls_abcd[i]))
        else:
            messages.append(_as_user_text("(missing image)\n"))

    messages.append(_as_user_text("\n" + question_text))
    return messages


def build_messages_task5_mcq_tags(
    conversation: List[Dict[str, Any]],
    option_tags_abcd: List[str],
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=False, include_turn_tags_as_text=True, lang=lang)

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    for i, letter in enumerate(LETTER_LIST):
        tag_txt = option_tags_abcd[i] if i < len(option_tags_abcd) else ""
        messages.append(_as_user_text(f"{letter}: {tag_txt}\n"))

    messages.append(_as_user_text("\n" + question_text))
    return messages


def build_messages_task5_yn_image(
    conversation: List[Dict[str, Any]],
    option_image_url: str,
    statement_text_before_image: str,
    question_text_after_view: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=True, include_turn_tags_as_text=False, lang=lang)

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    messages.append(_as_user_text((statement_text_before_image or "") + "\n"))
    if option_image_url:
        messages.append(_as_user_image(option_image_url))
    messages.append(_as_user_text("\n\n" + (question_text_after_view or "")))
    return messages


def build_messages_task5_yn_tags(
    conversation: List[Dict[str, Any]],
    option_tags_text: str,
    statement_text_before_tags: str,
    question_text_after_view: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=False, include_turn_tags_as_text=True, lang=lang)

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    messages.append(_as_user_text((statement_text_before_tags or "") + "\n"))
    messages.append(_as_user_text(option_tags_text or ""))
    messages.append(_as_user_text("\n\n" + (question_text_after_view or "")))
    return messages


# -----------------------------
# Task7 : MCQ / YesNo (image/tags)
# -----------------------------
def build_messages_task7_mcq_image(
    conversation: List[Dict[str, Any]],
    utterance_text: str,
    option_image_urls_abcd: List[str],
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=True, include_turn_tags_as_text=False, lang=lang)

    if utterance_text:
        messages.append(_as_user_text(_L(lang, "utterance") + "\n"))
        messages.append(_as_user_text(utterance_text + "\n\n"))

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    for i, letter in enumerate(LETTER_LIST):
        messages.append(_as_user_text(f"{letter}: "))
        if i < len(option_image_urls_abcd) and option_image_urls_abcd[i]:
            messages.append(_as_user_image(option_image_urls_abcd[i]))
        else:
            messages.append(_as_user_text("(missing image)\n"))

    messages.append(_as_user_text("\n" + question_text))
    return messages


def build_messages_task7_mcq_tags(
    conversation: List[Dict[str, Any]],
    utterance_text: str,
    option_tags_abcd: List[str],
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=False, include_turn_tags_as_text=True, lang=lang)

    if utterance_text:
        messages.append(_as_user_text(_L(lang, "utterance") + "\n"))
        messages.append(_as_user_text(utterance_text + "\n\n"))

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    for i, letter in enumerate(LETTER_LIST):
        tag_txt = option_tags_abcd[i] if i < len(option_tags_abcd) else ""
        messages.append(_as_user_text(f"{letter}: {tag_txt}\n"))

    messages.append(_as_user_text("\n" + question_text))
    return messages


def build_messages_task7_yn_image(
    conversation: List[Dict[str, Any]],
    utterance_text: str,
    option_image_url: str,
    statement_text_before_image: str,
    question_text_after_view: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=True, include_turn_tags_as_text=False, lang=lang)

    if utterance_text:
        messages.append(_as_user_text(_L(lang, "utterance") + "\n"))
        messages.append(_as_user_text(utterance_text + "\n\n"))

    items: List[Dict[str, Any]] = []
    items.append(_text_item(_L(lang, "options") + "\n"))
    items.append(_text_item((statement_text_before_image or "") + "\n"))
    if option_image_url:
        items.append(_image_item(option_image_url))
    items.append(_text_item("\n\n" + (question_text_after_view or "")))

    messages.append(_as_user_content_items(items))
    return messages


def build_messages_task7_yn_tags(
    conversation: List[Dict[str, Any]],
    utterance_text: str,
    option_tags_text: str,
    statement_text_before_tags: str,
    question_text_after_view: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=False, include_turn_tags_as_text=True, lang=lang)

    if utterance_text:
        messages.append(_as_user_text(_L(lang, "utterance") + "\n"))
        messages.append(_as_user_text(utterance_text + "\n\n"))

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    messages.append(_as_user_text((statement_text_before_tags or "") + "\n"))
    messages.append(_as_user_text(option_tags_text or ""))
    messages.append(_as_user_text("\n\n" + (question_text_after_view or "")))
    return messages


# -----------------------------
# Task6 : type1/type2 + tags variants
# -----------------------------
def build_messages_task6_type1_mcq_image(
    conversation: List[Dict[str, Any]],
    option_texts_abcd: List[str],
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=True, include_turn_tags_as_text=False, lang=lang)

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    for i, letter in enumerate(LETTER_LIST):
        opt_txt = option_texts_abcd[i] if i < len(option_texts_abcd) else ""
        messages.append(_as_user_text(f"{letter}: "))
        messages.append(_as_user_text(f"{opt_txt}"))

    messages.append(_as_user_text("\n" + question_text))
    return messages


def build_messages_task6_type1_mcq_tags(
    conversation: List[Dict[str, Any]],
    option_texts_abcd: List[str],
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=False, include_turn_tags_as_text=True, lang=lang)

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    for i, letter in enumerate(LETTER_LIST):
        opt_txt = option_texts_abcd[i] if i < len(option_texts_abcd) else ""
        messages.append(_as_user_text(f"{letter}: {opt_txt}\n"))

    messages.append(_as_user_text("\n" + question_text))
    return messages


def build_messages_task6_type1_yn_image(
    conversation: List[Dict[str, Any]],
    view_text: str,
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=True, include_turn_tags_as_text=False, lang=lang)

    combined = _L(lang, "options") + "\n" + (view_text or "").strip() + "\n\n" + (question_text or "")
    messages.append(_as_user_text(combined))
    return messages


def build_messages_task6_type1_yn_tags(
    conversation: List[Dict[str, Any]],
    view_text: str,
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    _append_conversation_lines(messages, conversation, include_turn_image=False, include_turn_tags_as_text=True, lang=lang)

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    messages.append(_as_user_text((view_text or "").strip()))
    messages.append(_as_user_text("\n\n" + (question_text or "")))
    return messages


def build_messages_task6_type2_mcq_image(
    tag_text: str,
    conversation: List[Dict[str, Any]],
    option_texts_abcd: List[str],
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    messages.append(_as_user_text(_L(lang, "outfit_info") + "\n"))
    messages.append(_as_user_text(tag_text.strip() + "\n\n"))

    if conversation:
        _append_conversation_lines(messages, conversation, include_turn_image=True, include_turn_tags_as_text=False, lang=lang)

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    for i, letter in enumerate(LETTER_LIST):
        opt_txt = option_texts_abcd[i] if i < len(option_texts_abcd) else ""
        messages.append(_as_user_text(f"{letter}: "))
        messages.append(_as_user_text(f"{opt_txt}"))

    messages.append(_as_user_text("\n" + question_text))
    return messages


def build_messages_task6_type2_mcq_tags(
    tag_text: str,
    conversation: List[Dict[str, Any]],
    option_texts_abcd: List[str],
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    messages.append(_as_user_text(_L(lang, "outfit_info") + "\n"))
    messages.append(_as_user_text(tag_text.strip() + "\n\n"))

    if conversation:
        _append_conversation_lines(messages, conversation, include_turn_image=False, include_turn_tags_as_text=True, lang=lang)

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    for i, letter in enumerate(LETTER_LIST):
        opt_txt = option_texts_abcd[i] if i < len(option_texts_abcd) else ""
        messages.append(_as_user_text(f"{letter}: {opt_txt}\n"))

    messages.append(_as_user_text("\n" + question_text))
    return messages


def build_messages_task6_type2_yn_image(
    tag_text: str,
    conversation: List[Dict[str, Any]],
    view_text: str,
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    messages.append(_as_user_text(_L(lang, "outfit_info") + "\n"))
    messages.append(_as_user_text(tag_text.strip() + "\n\n"))

    if conversation:
        _append_conversation_lines(messages, conversation, include_turn_image=True, include_turn_tags_as_text=False, lang=lang)

    combined = _L(lang, "options") + "\n" + (view_text or "").strip() + "\n\n" + (question_text or "")
    messages.append(_as_user_text(combined))
    return messages


def build_messages_task6_type2_yn_tags(
    tag_text: str,
    conversation: List[Dict[str, Any]],
    view_text: str,
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    messages.append(_as_user_text(_L(lang, "outfit_info") + "\n"))
    messages.append(_as_user_text(tag_text.strip() + "\n\n"))

    if conversation:
        _append_conversation_lines(messages, conversation, include_turn_image=False, include_turn_tags_as_text=True, lang=lang)

    messages.append(_as_user_text(_L(lang, "options") + "\n"))
    messages.append(_as_user_text((view_text or "").strip()))
    messages.append(_as_user_text("\n\n" + (question_text or "")))
    return messages


# -----------------------------
# Task8 : MCQ (Paragraph Ordering) - image/tags
# -----------------------------
def build_messages_task8_type1_mcq_image(
    conversation_a: Any,
    conversation_b: Any,
    conversation_c: Any,
    images: List[str],
    option_texts_abcd: List[str],
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []

    messages.append(_as_user_text(_L(lang, "task8_type1_intro")))
    if lang == "ko":
        question_text = question_text.replace("[Dialogue Segments] is a set of segments with the dialogue randomly shuffled.", "")

    messages.append(_as_user_text(_L(lang, "dialogue_segments") + "\n"))

    for label, turns in [("a", conversation_a), ("b", conversation_b), ("c", conversation_c)]:
        messages.append(_as_user_text(f"({label})\n"))
        _append_turns_as_lines(messages, turns, include_turn_image_url_as_text=False)
        messages.append(_as_user_text("\n"))

    messages.append(_as_user_text(_L(lang, "order_of_outfit_image") + "\n"))
    for idx, img_url in enumerate(images):
        if img_url:
            messages.append(_as_user_image(img_url))
        messages.append(_as_user_text("\n"))

    messages.append(_as_user_text("\n" + _L(lang, "options") + "\n"))
    for i, letter in enumerate(LETTER_LIST):
        opt_txt = option_texts_abcd[i] if i < len(option_texts_abcd) else ""
        messages.append(_as_user_text(f"{letter}: {opt_txt}\n"))

    messages.append(_as_user_text("\n" + question_text))
    return messages


def build_messages_task8_type1_mcq_tags(
    conversation_a: Any,
    conversation_b: Any,
    conversation_c: Any,
    image_tags: List[str],
    option_texts_abcd: List[str],
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []

    messages.append(_as_user_text(_L(lang, "task8_type1_intro")))
    if lang == "ko":
        question_text = question_text.replace("[대화 단락]은 user와 assistant 간의 대화를 무작위로 섞은 단락 집합이다.", "")

    messages.append(_as_user_text(_L(lang, "dialogue_segments") + "\n"))

    for label, turns in [("a", conversation_a), ("b", conversation_b), ("c", conversation_c)]:
        messages.append(_as_user_text(f"({label})\n"))
        _append_turns_as_lines(messages, turns, include_turn_image_url_as_text=False)
        messages.append(_as_user_text("\n"))

    messages.append(_as_user_text(_L(lang, "order_of_outfit") + "\n"))
    for idx, t in enumerate(image_tags):
        messages.append(_as_user_text(f"{_L(lang, 'outfit_info_label')} {idx + 1}: " + (str(t) if t else "") + "\n"))

    messages.append(_as_user_text("\n" + _L(lang, "options") + "\n"))
    for i, letter in enumerate(LETTER_LIST):
        opt_txt = option_texts_abcd[i] if i < len(option_texts_abcd) else ""
        messages.append(_as_user_text(f"{letter}: {opt_txt}\n"))

    messages.append(_as_user_text("\n" + question_text))
    return messages


def build_messages_task8_type2_mcq_image(
    conversation_init: Any,
    conversation_a: Any,
    conversation_b: Any,
    conversation_c: Any,
    last_image_url: str,
    option_texts_abcd: List[str],
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []

    messages.append(_as_user_text(_L(lang, "task8_type2_intro")))
    if lang == "ko":
        question_text = question_text.replace("[대화 단락]은 [초기 대화] 이후 대화를 무작위로 섞은 단락 집합이다.", "").strip()

    messages.append(_as_user_text(_L(lang, "initial_dialogue") + "\n"))
    _append_turns_as_lines(messages, conversation_init, include_turn_image_url_as_text=False)

    messages.append(_as_user_text("\n" + _L(lang, "dialogue_segments") + "\n"))
    for label, turns in [("a", conversation_a), ("b", conversation_b), ("c", conversation_c)]:
        messages.append(_as_user_text(f"({label})\n"))
        _append_turns_as_lines(messages, turns, include_turn_image_url_as_text=False)
        messages.append(_as_user_text("\n"))

    messages.append(_as_user_text(_L(lang, "last_segment_image") + "\n"))
    if last_image_url:
        messages.append(_as_user_image(last_image_url))

    messages.append(_as_user_text("\n" + _L(lang, "options") + "\n"))
    for i, letter in enumerate(LETTER_LIST):
        opt_txt = option_texts_abcd[i] if i < len(option_texts_abcd) else ""
        messages.append(_as_user_text(f"{letter}: {opt_txt}\n"))

    messages.append(_as_user_text("\n" + question_text))
    return messages


def build_messages_task8_type2_mcq_tags(
    conversation_init: Any,
    conversation_a: Any,
    conversation_b: Any,
    conversation_c: Any,
    last_image_tags: str,
    option_texts_abcd: List[str],
    question_text: str,
    lang: str = "ko",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []

    messages.append(_as_user_text(_L(lang, "task8_type2_intro")))
    if lang == "ko":
        question_text = question_text.replace("[대화 단락]은 [초기 대화] 이후 대화를 무작위로 섞은 단락 집합이다.", "").strip()

    messages.append(_as_user_text(_L(lang, "initial_dialogue") + "\n"))
    _append_turns_as_lines(messages, conversation_init, include_turn_image_url_as_text=False)

    messages.append(_as_user_text("\n" + _L(lang, "dialogue_segments") + "\n"))
    for label, turns in [("a", conversation_a), ("b", conversation_b), ("c", conversation_c)]:
        messages.append(_as_user_text(f"({label})\n"))
        _append_turns_as_lines(messages, turns, include_turn_image_url_as_text=False)
        messages.append(_as_user_text("\n"))

    messages.append(_as_user_text(_L(lang, "last_segment_info") + "\n"))
    if last_image_tags:
        messages.append(_as_user_text(str(last_image_tags) + "\n"))

    messages.append(_as_user_text("\n" + _L(lang, "options") + "\n"))
    for i, letter in enumerate(LETTER_LIST):
        opt_txt = option_texts_abcd[i] if i < len(option_texts_abcd) else ""
        messages.append(_as_user_text(f"{letter}: {opt_txt}\n"))

    messages.append(_as_user_text("\n" + question_text))
    return messages
