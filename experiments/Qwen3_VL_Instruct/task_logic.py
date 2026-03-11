#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

from .messages import (

    build_messages_task1_open_image,

    build_messages_task2_mcq_image,
    build_messages_task2_yn_image,

    build_messages_task3_3_mcq_image,
    build_messages_task3_3_mcq_tags,
    build_messages_task3_yn_image,
    build_messages_task3_yn_tags,

    build_messages_task4_yn_image,
    build_messages_task4_yn_tags,

    build_messages_task5_mcq_image,
    build_messages_task5_mcq_tags,
    build_messages_task5_yn_image,
    build_messages_task5_yn_tags,

    build_messages_task6_type1_mcq_image,
    build_messages_task6_type1_mcq_tags,
    build_messages_task6_type1_yn_image,
    build_messages_task6_type1_yn_tags,

    build_messages_task6_type2_mcq_image,
    build_messages_task6_type2_mcq_tags,
    build_messages_task6_type2_yn_image,
    build_messages_task6_type2_yn_tags,

    build_messages_task7_mcq_image,
    build_messages_task7_mcq_tags,
    build_messages_task7_yn_image,
    build_messages_task7_yn_tags,


    build_messages_task8_type1_mcq_image,
    build_messages_task8_type1_mcq_tags,
    build_messages_task8_type2_mcq_image,
    build_messages_task8_type2_mcq_tags,
)

LETTER_LIST = ["A", "B", "C", "D"]


# -----------------------------
# Korean particle helpers
# -----------------------------

def _is_hangul_syllable(ch: str) -> bool:
    o = ord(ch)
    return 0xAC00 <= o <= 0xD7A3


def _has_final_consonant(kor_syllable: str) -> bool:
    code = ord(kor_syllable) - 0xAC00
    jong = code % 28
    return jong != 0


def grammer_1(name: str) -> str:
    if not name:
        return "는"
    last = name[-1]
    if _is_hangul_syllable(last):
        return "은" if _has_final_consonant(last) else "는"
    return "는"


def grammer_2(name: str) -> str:
    if not name:
        return "가"
    last = name[-1]
    if _is_hangul_syllable(last):
        return "이" if _has_final_consonant(last) else "가"
    return "가"


# -----------------------------
# Option list helpers
# -----------------------------

def prepare_options_lists(options: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    image_urls: List[str] = []
    tags: List[str] = []
    texts: List[str] = []

    for opt in options:
        image_urls.append(str(opt.get("Url", "") or opt.get("url", "") or ""))

        t = opt.get("Tags") or opt.get("tags") or []
        if isinstance(t, list):
            tags.append(", ".join(map(str, t)))
        else:
            tags.append(str(t))

        texts.append(str(opt.get("Text", "") or opt.get("text", "") or ""))

    return image_urls, tags, texts


def _render_conversation_as_lines(conversation: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for t in conversation:
        role = t.get("role", "unknown")
        content = str(t.get("content", ""))
        lines.append(f"'{role}': '{content}'")
    return "\n".join(lines)


def render_prompt(
    template: str,
    *,
    conversation_text: str,
    user_name: str,
    image_url_list: List[str],
    tag_list: List[str],
    text_list: List[str],
) -> str:
    s = template
    s = s.replace("{user}", user_name)
    s = s.replace("{grammer_1(user)}", grammer_1(user_name))
    s = s.replace("{grammer_2(user)}", grammer_2(user_name))
    s = s.replace("{conversation}", conversation_text)
    s = s.replace("{Conversation}", conversation_text)

    for i in range(4):
        s = s.replace(f"{{image_url_list[{i}]}}", image_url_list[i] if i < len(image_url_list) else "")
        s = s.replace(f"{{tag_list[{i}]}}", tag_list[i] if i < len(tag_list) else "")
        s = s.replace(f"{{text_list[{i}]}}", text_list[i] if i < len(text_list) else "")

    return s


def extract_question_part(full_prompt: str) -> str:
    idx = full_prompt.find("질문:")
    if idx < 0:
        return full_prompt.strip()
    return full_prompt[idx:].strip()


def _normalize_text_with_user_grammar(s: str, user_name: str) -> str:
    s = (s or "")
    s = s.replace("{user}", user_name)
    s = s.replace("{grammer_1(user)}", grammer_1(user_name))
    s = s.replace("{grammer_2(user)}", grammer_2(user_name))
    return s


def _infer_subtype(item: Dict[str, Any]) -> int:
    """
    task subtype detection:
    - supports keys like: subtype / Subtype / sub_type / Task_Subtype / Sub_Type
    - default: 1
    """
    for k in ("subtype", "Subtype", "sub_type", "Task_Subtype", "Sub_Type", "task_subtype"):
        if k in item and item.get(k) is not None:
            try:
                return int(item.get(k))
            except Exception:
                pass
    return 1


def _normalize_question_type(qt: str) -> str:
    qt = (qt or "").strip().lower()
    if qt in ("yes_no", "yesno", "yes/no", "yn", "binary"):
        return "yes_or_no"
    if qt in ("multiple_choice", "multiplechoice", "mcq", "choice"):
        return "multiple_choice"
    return qt

def _normalize_mode(m: str) -> str:
    m = (m or "").strip().lower()
    if m in ("tag", "tags", "text"):
        return "tags"
    if m in ("image", "img", "vision"):
        return "image"
    return m


# -----------------------------
# Message dispatch
# -----------------------------

def build_messages_dispatch(
    *,
    item: Dict[str, Any],
    task_id: int,
    question_type: str,
    mode: str,
    template_key: str,
    image_url_list: List[str],
    tag_list: List[str],
    text_list: List[str],
) -> Tuple[List[Dict[str, Any]], str]:
    
    question_type = _normalize_question_type(question_type)
    mode = _normalize_mode(mode)
    
    prompt_tpl = item.get("Prompt_Template", {}) or {}
    if template_key not in prompt_tpl:
        raise KeyError(f"Prompt_Template missing key: {template_key}")

    tpl = str(prompt_tpl[template_key])
    conversation = item.get("Conversation", []) or item.get("conversation", []) or []
    user_name = str(item.get("user", "user"))

    conversation_text = _render_conversation_as_lines(conversation)
    full_prompt = render_prompt(
        template=tpl,
        conversation_text=conversation_text,
        user_name=user_name,
        image_url_list=image_url_list,
        tag_list=tag_list,
        text_list=text_list,
    )
    question_text = extract_question_part(full_prompt)

    # ---- Per Task ----
    if task_id == 1:
        msgs = build_messages_task1_open_image(conversation=conversation, question_text=question_text)
        return msgs, "image"

    if task_id == 2:
        if question_type == "multiple_choice":
            output_idx = item.get("Output", 0)
            tag_text = tag_list[output_idx] if tag_list else ""
            msgs = build_messages_task2_mcq_image(
                tag_text=tag_text,
                option_image_urls_abcd=image_url_list,
                question_text=question_text,
            )
            return msgs, "image"
        if question_type == "yes_or_no":
            tag_text = tag_list[0] if tag_list else ""
            marker = image_url_list[0] if image_url_list else ""

            view_before = "[의상 정보]에 해당하는 가장 적합한 의상 이미지는 다음과 같다. "
            msgs = build_messages_task2_yn_image(
                tag_text=tag_text,
                option_image_url=marker,
                view_statement_before_image=view_before,
                question_text_after_view=question_text,
            )
            return msgs, "image"

    if task_id == 3:
        if question_type == "multiple_choice":
            if mode == "image":
                msgs = build_messages_task3_3_mcq_image(
                    conversation=conversation,
                    option_image_urls_abcd=image_url_list,
                    question_text=question_text,
                    include_turn_image=False,
                )
                return msgs, "image"
            else:
                msgs = build_messages_task3_3_mcq_tags(
                    conversation=conversation,
                    option_tags_abcd=tag_list,
                    question_text=question_text,
                    include_turn_tags_as_text=False,
                )
                return msgs, "tags"

        if question_type == "yes_or_no":
            if mode == "image":
                marker = image_url_list[0] if image_url_list else ""
                before = full_prompt.split("[보기]")[1]
                if marker and marker in before:
                    statement_before_image = before.split(marker)[0]
                else:
                    statement_before_image = f"{user_name}{grammer_2(user_name)} 선호하는 의상을 다음과 같다. "
                msgs = build_messages_task3_yn_image(
                    conversation=conversation,
                    option_image_url=marker,
                    statement_text_before_image=statement_before_image,
                    question_text_after_view=question_text,
                )
                return msgs, "image"
            else:
                after_view = full_prompt.split("[보기]")[1]
                view_part = after_view.split("질문:")[0].strip("\n")

                img_marker = image_url_list[0] if image_url_list else ""
                tag_marker = tag_list[0] if tag_list else ""
                if img_marker and img_marker in view_part:
                    view_part = view_part.replace(img_marker, tag_marker)

                msgs = build_messages_task3_yn_tags(
                    conversation=conversation,
                    option_tags_text=tag_marker,
                    view_text=view_part,
                    question_text_after_view=question_text,
                )
                return msgs, "tags"

    if task_id == 4:
        conversation = item.get("Conversation", []) or []
        user_name = str(item.get("user", "user"))
        conv_mode = mode

        if question_type == "multiple_choice":
            from .messages import _as_user_text, _as_user_image 

            msgs: List[Dict[str, Any]] = []
            msgs.append(_as_user_text("아래 [대화]는 user와 assistant 간의 대화 내용이다.\n"))
            msgs.append(_as_user_text("[대화]\n"))

            for t in conversation:
                role = t.get("role", "unknown")
                content = str(t.get("content", ""))

                image_info = t.get("image_info") or {}
                img_url = None
                tags = None
                if isinstance(image_info, dict):
                    img_url = image_info.get("Url") or image_info.get("url")
                    tags = image_info.get("Tags") or image_info.get("tags")

                if conv_mode == "image" and img_url:
                    msgs.append(_as_user_image(str(img_url)))
                elif conv_mode == "tags" and tags:
                    tag_txt = ", ".join(map(str, tags)) if isinstance(tags, list) else str(tags)
                    msgs.append(_as_user_text(tag_txt + "\n"))

                msgs.append(_as_user_text(f"'{role}': '{content}'\n"))

            msgs.append(_as_user_text("[보기]\n"))
            for i, letter in enumerate(LETTER_LIST):
                opt_txt = text_list[i] if i < len(text_list) else ""
                opt_txt = _normalize_text_with_user_grammar(opt_txt, user_name)
                msgs.append(_as_user_text(f"{letter}: {opt_txt}\n"))

            msgs.append(_as_user_text("\n" + question_text))
            return msgs, ("image" if conv_mode == "image" else "tags")

        if question_type == "yes_or_no":
            view_text = text_list[0] if len(text_list) > 0 else ""
            view_text = _normalize_text_with_user_grammar(view_text, user_name)

            if conv_mode == "image":
                msgs = build_messages_task4_yn_image(
                    conversation=conversation,
                    view_text=view_text,
                    question_text=question_text,
                )
                return msgs, "image"
            else:
                msgs = build_messages_task4_yn_tags(
                    conversation=conversation,
                    view_text=view_text,
                    question_text=question_text,
                )
                return msgs, "tags"

        raise ValueError(f"Task3 unsupported question_type: {question_type}")


    if task_id == 5:
        if question_type == "multiple_choice":
            if mode == "image":
                msgs = build_messages_task5_mcq_image(
                    conversation=conversation,
                    option_image_urls_abcd=image_url_list,
                    question_text=question_text,
                )
                return msgs, "image"
            else:
                msgs = build_messages_task5_mcq_tags(
                    conversation=conversation,
                    option_tags_abcd=tag_list,
                    question_text=question_text,
                )
                return msgs, "tags"

        if question_type == "yes_or_no":
            after_view = full_prompt.split("[보기]")[1]
            before_question = after_view.split("질문:")[0]

            if mode == "image":
                marker = image_url_list[0] if image_url_list else ""
                if marker and marker in before_question:
                    statement_before = before_question.split(marker)[0].strip("\n")
                else:
                    statement_before = "[대화] 내 user의 마지막 발화를 기반으로 user가 선호하는 의상 이미지는 다음과 같다."
                msgs = build_messages_task5_yn_image(
                    conversation=conversation,
                    option_image_url=marker,
                    statement_text_before_image=statement_before,
                    question_text_after_view=question_text,
                )
                return msgs, "image"
            else:
                tag_marker = tag_list[0] if tag_list else ""
                if tag_marker and tag_marker in before_question:
                    statement_before = before_question.split(tag_marker)[0].strip("\n")
                else:
                    statement_before = "[대화] 내 user의 마지막 발화를 기반으로 user가 선호하는 의상 정보는 다음과 같다."
                msgs = build_messages_task5_yn_tags(
                    conversation=conversation,
                    option_tags_text=tag_marker,
                    statement_text_before_tags=statement_before,
                    question_text_after_view=question_text,
                )
                return msgs, "tags"

    if task_id == 6:
        subtype = _infer_subtype(item)
        option_texts = [_normalize_text_with_user_grammar(t, user_name) for t in (text_list or [])]
        view_text = _normalize_text_with_user_grammar(text_list[0] if text_list else "", user_name)

        if subtype == 1:
            if question_type == "multiple_choice":
                if mode == "image":
                    msgs = build_messages_task6_type1_mcq_image(
                        conversation=conversation,
                        option_texts_abcd=option_texts,
                        question_text=question_text,
                    )
                    return msgs, "image"
                else:
                    msgs = build_messages_task6_type1_mcq_tags(
                        conversation=conversation,
                        option_texts_abcd=option_texts,
                        question_text=question_text,
                    )
                    return msgs, "tags"

            if question_type == "yes_or_no":
                if mode == "image":
                    msgs = build_messages_task6_type1_yn_image(
                        conversation=conversation,
                        view_text=view_text,
                        question_text=question_text,
                    )
                    return msgs, "image"
                else:
                    msgs = build_messages_task6_type1_yn_tags(
                        conversation=conversation,
                        view_text=view_text,
                        question_text=question_text,
                    )
                    return msgs, "tags"

        else:
            tag_text = tag_list[0] if tag_list else ""
            if question_type == "multiple_choice":
                if mode == "image":
                    msgs = build_messages_task6_type2_mcq_image(
                        tag_text=tag_text,
                        conversation=conversation,
                        option_texts_abcd=option_texts,
                        question_text=question_text,
                    )
                    return msgs, "image"
                else:
                    msgs = build_messages_task6_type2_mcq_tags(
                        tag_text=tag_text,
                        conversation=conversation,
                        option_texts_abcd=option_texts,
                        question_text=question_text,
                    )
                    return msgs, "tags"

            if question_type == "yes_or_no":
                if mode == "image":
                    msgs = build_messages_task6_type2_yn_image(
                        tag_text=tag_text,
                        conversation=conversation,
                        view_text=view_text,
                        question_text=question_text,
                    )
                    return msgs, "image"
                else:
                    msgs = build_messages_task6_type2_yn_tags(
                        tag_text=tag_text,
                        conversation=conversation,
                        view_text=view_text,
                        question_text=question_text,
                    )
                    return msgs, "tags"

    if task_id == 7:
        utterance_text = full_prompt.split("[발화]:")[1].split("[보기]")[0].strip()
        special_item = '\"'+utterance_text.split("**")[1] +'\"'
        question_text = question_text.replace("강조된 옷", special_item)
        
        if question_type == "multiple_choice":
            if mode == "image":
                msgs = build_messages_task7_mcq_image(
                    conversation=conversation,
                    utterance_text=utterance_text,
                    option_image_urls_abcd=image_url_list,
                    question_text=question_text,
                )
                return msgs, "image"
            else:
                msgs = build_messages_task7_mcq_tags(
                    conversation=conversation,
                    utterance_text=utterance_text,
                    option_tags_abcd=tag_list,
                    question_text=question_text,
                )
                return msgs, "tags"

        if question_type == "yes_or_no":
            after_view = full_prompt.split("[보기]")[1]
            before_question = after_view.split("질문:")[0].strip("\n")

            if mode == "image":
                marker = image_url_list[0] if image_url_list else ""
                if marker and marker in before_question:
                    statement_before = before_question.split(marker)[0].strip("\n")
                else:
                    statement_before = "[대화]를 바탕으로 [발화]에서 강조된 의상 이미지는 다음과 같다."
                msgs = build_messages_task7_yn_image(
                    conversation=conversation,
                    utterance_text=utterance_text,
                    option_image_url=marker,
                    statement_text_before_image=statement_before,
                    question_text_after_view=question_text,
                )
                return msgs, "image"
            else:
                tag_marker = tag_list[0] if tag_list else ""
                if tag_marker and tag_marker in before_question:
                    statement_before = before_question.split(tag_marker)[0].strip("\n")
                else:
                    statement_before = "[대화]를 바탕으로 [발화]에서 강조된 의상 정보는 다음과 같다."
                msgs = build_messages_task7_yn_tags(
                    conversation=conversation,
                    utterance_text=utterance_text,
                    option_tags_text=tag_marker,
                    statement_text_before_tags=statement_before,
                    question_text_after_view=question_text,
                )
                return msgs, "tags"

    if task_id == 8:
        conv_a = item.get("Conversation_a", []) or []
        conv_b = item.get("Conversation_b", []) or []
        conv_c = item.get("Conversation_c", []) or []
        conv_init = item.get("Conversation_init", []) or []
        images_list = item.get("Images", []) or []

        def _img_to_url_and_tags(x: Any) -> Tuple[str, str]:
            if isinstance(x, dict):
                url = str(x.get("Url") or x.get("url") or "")
                tags = x.get("Tags") or x.get("tags")
                if isinstance(tags, list):
                    tag_txt = ", ".join(map(str, tags))
                else:
                    tag_txt = str(tags) if tags else ""
                return url, tag_txt
            return str(x or ""), ""

        question_text = question_text if mode == "image" else question_text.replace("의상 이미지", "의상 정보")

        if conv_init:
            # type2
            last_img = images_list[0] if images_list else ""
            last_url, last_tags = _img_to_url_and_tags(last_img)
            if mode == "image":
                msgs = build_messages_task8_type2_mcq_image(
                    conversation_init=conv_init,
                    conversation_a=conv_a,
                    conversation_b=conv_b,
                    conversation_c=conv_c,
                    last_image_url=last_url,
                    option_texts_abcd=text_list,
                    question_text=question_text,
                )
                return msgs, "image"
            else:
                msgs = build_messages_task8_type2_mcq_tags(
                    conversation_init=conv_init,
                    conversation_a=conv_a,
                    conversation_b=conv_b,
                    conversation_c=conv_c,
                    last_image_tags=last_tags, 
                    option_texts_abcd=text_list,
                    question_text=question_text,
                )
                return msgs, "tags"

        else:
            # type1
            img_urls: List[str] = []
            img_tags: List[str] = []
            for img in images_list:
                u, t = _img_to_url_and_tags(img)
                img_urls.append(u)
                img_tags.append(t)

            if mode == "image":
                msgs = build_messages_task8_type1_mcq_image(
                    conversation_a=conv_a,
                    conversation_b=conv_b,
                    conversation_c=conv_c,
                    images=img_urls,
                    option_texts_abcd=text_list,
                    question_text=question_text,
                )
                return msgs, "image"
            else:
                msgs = build_messages_task8_type1_mcq_tags(
                    conversation_a=conv_a,
                    conversation_b=conv_b,
                    conversation_c=conv_c,
                    image_tags=img_tags,
                    option_texts_abcd=text_list,
                    question_text=question_text,
                )
                return msgs, "tags"

    raise ValueError(f"Unsupported dispatch: task_id={task_id}, question_type={question_type}, mode={mode}")


def score_prediction(
    task_id: int,
    question_type: str,
    response_text: str,
    gt_output: Any,
    correct_letter: Optional[str],
) -> Dict[str, Any]:
    pred_text = (response_text or "").strip()

    rec: Dict[str, Any] = {
        "raw_response": response_text,
        "pred_text": pred_text,
        "gt_output": gt_output,
    }

    if question_type == "multiple_choice":
        ans = None
        for L in LETTER_LIST:
            if f"정답: {L}" in pred_text:
                ans = L
                break
        if ans is None and pred_text in LETTER_LIST:
            ans = pred_text
        rec["pred_answer"] = ans
        rec["is_correct"] = (ans == correct_letter) if (ans and correct_letter) else False
        return rec

    if question_type == "yes_or_no":
        up = pred_text.upper()
        pred = "YES" if "YES" in up else ("NO" if "NO" in up else None)
        rec["pred_answer"] = pred
        rec["is_correct"] = (pred == str(gt_output).upper()) if pred else False
        return rec

    rec["pred_answer"] = pred_text
    rec["is_correct"] = None
    return rec
