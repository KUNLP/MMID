import json
import os
import re
import time
import random
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


def load_samples(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("The input file must be in JSON format.")
    return data


def build_dialog_text(conversation: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for turn in conversation:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            prefix = "사용자"
        elif role == "system":
            prefix = "시스템"
        elif role == "assistant":
            prefix = "어시스턴트"
        else:
            prefix = role or "기타"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def collect_images(conversation: List[Dict[str, Any]]) -> List[str]:
    """Collect image URLs that appear in the conversation"""
    urls: List[str] = []
    for turn in conversation:
        image_info = turn.get("image_info", [])
        if isinstance(image_info, list):
            for url in image_info:
                if isinstance(url, str):
                    urls.append(url)
        elif isinstance(image_info, dict):
            url = image_info.get("Url")
            if isinstance(url, str):
                urls.append(url)
    seen = set()
    unique_urls: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return unique_urls


def format_prompt(sample: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Input: Prompt_Template with the conversation
    Return: list of image URLs
    """
    prompt_template: str = sample.get("Prompt_Templete", "")
    conversation: List[Dict[str, Any]] = sample.get("Conversation", [])

    dialog_text = build_dialog_text(conversation)
    prompt_text = prompt_template.replace("{dialog}", dialog_text)

    images = collect_images(conversation)
    return prompt_text, images


def call_gpt4o_for_bbox(
    client: OpenAI,
    prompt_text: str,
    images: List[str],
    max_retries: int = 3,
) -> Optional[str]:
    """
    Input: Prompt_Template with the conversation and the images
    Return: bounding box
    """
    for attempt in range(max_retries):
        try:
            content: List[Dict[str, Any]] = []
            content.append(
                {
                    "type": "text",
                    "text": prompt_text,
                }
            )
            for url in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": url},
                    }
                )

            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]

            resp = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=messages,
                temperature=0.0,
                max_tokens=64,
            )
            answer = (resp.choices[0].message.content or "").strip()
            return answer
        except Exception as e:
            print(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                return None
    return None


def parse_bbox(text: str) -> Optional[List[float]]:
    """Extract four numbers in the format [x, y, w, h] from the text."""
    if not text:
        return None
    nums = re.findall(r"-?\d+\.?\d*", text)
    if len(nums) < 4:
        return None
    try:
        vals = [float(n) for n in nums[:4]]
        return vals
    except ValueError:
        return None


def bbox_to_xyxy(bbox: List[float]) -> Tuple[float, float, float, float]:
    """[x, y, w, h] -> (x1, y1, x2, y2)"""
    x, y, w, h = bbox
    return x, y, x + w, y + h


def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Calculate the center coordinates (cx, cy) of the bounding box [x, y, w, h]"""
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def bbox_area(bbox: List[float]) -> float:
    """Calculate the area of the bounding box [x, y, w, h]"""
    _, _, w, h = bbox
    return max(0.0, w) * max(0.0, h)


def center_distance(b1: List[float], b2: List[float]) -> float:
    """Calculate the Euclidean distance between the centers of the two bounding boxes"""
    c1x, c1y = bbox_center(b1)
    c2x, c2y = bbox_center(b2)
    return ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5


def area_ratio(pred: List[float], gold: List[float]) -> Optional[float]:
    """Calculate the area ratio of the predicted and gold bounding boxes: |B_pred| / |B_gold|"""
    gold_area = bbox_area(gold)
    pred_area = bbox_area(pred)
    if gold_area <= 0:
        return None
    return pred_area / gold_area


def compute_iou(b1: List[float], b2: List[float]) -> float:
    """Calculate the IoU of the two bounding box [x, y, w, h]"""
    x1_min, y1_min, x1_max, y1_max = bbox_to_xyxy(b1)
    x2_min, y2_min, x2_max, y2_max = bbox_to_xyxy(b2)

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def evaluate(
    samples: List[Dict[str, Any]],
    client: OpenAI,
    max_samples: Optional[int] = None,
    delay_sec: float = 1.0,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    if max_samples is not None:
        samples = samples[:max_samples]

    for idx, sample in enumerate(samples, start=1):
        qid = sample.get("Question_ID", "")
        gt_str: str = sample.get("Output", "")

        print(f"\n[{idx}/{len(samples)}] Question_ID: {qid}")

        gt_bbox = parse_bbox(gt_str)
        if gt_bbox is None:
            print(f"  [Warning] Failed to parse the GT box: {gt_str}")

        prompt_text, images = format_prompt(sample)
        print(f"  Number of images: {len(images)}")

        print("  Calling GPT-4o...")
        pred_raw = call_gpt4o_for_bbox(client, prompt_text, images)

        if pred_raw is None:
            print("  [Fail] API call failed...")
            result_item = {
                "question_id": qid,
                "gt_output": gt_str,
                "gt_bbox": gt_bbox,
                "pred_raw": None,
                "pred_bbox": None,
                "iou": None,
            }
            results.append(result_item)
            time.sleep(delay_sec)
            continue

        print(f"  Output: {pred_raw}")
        pred_bbox = parse_bbox(pred_raw)
        if pred_bbox is None:
            print("  [Fail] Failed to parse the predicted bounding box...")
            iou = None
            dist = None
            ratio = None
        elif gt_bbox is None:
            iou = None
            dist = None
            ratio = None
        else:
            iou = compute_iou(gt_bbox, pred_bbox)
            dist = center_distance(gt_bbox, pred_bbox)
            ratio = area_ratio(pred_bbox, gt_bbox)
            print(f"  IoU: {iou:.4f}")
            if dist is not None:
                print(f"  Center Distance: {dist:.4f}")
            if ratio is not None:
                print(f"  Area Ratio (|B_pred|/|B_gold|): {ratio:.4f}")

        result_item = {
            "question_id": qid,
            "gt_output": gt_str,
            "gt_bbox": gt_bbox,
            "pred_raw": pred_raw,
            "pred_bbox": pred_bbox,
            "iou": iou,
            "center_distance": dist,
            "area_ratio": ratio,
        }
        results.append(result_item)

        time.sleep(delay_sec)

    ious = [r["iou"] for r in results if isinstance(r.get("iou"), (int, float))]
    dists = [
        r["center_distance"]
        for r in results
        if isinstance(r.get("center_distance"), (int, float))
    ]
    ratios = [
        r["area_ratio"]
        for r in results
        if isinstance(r.get("area_ratio"), (int, float))
    ]

    mean_iou = float(sum(ious) / len(ious)) if ious else 0.0
    acc_05 = float(sum(1 for v in ious if v >= 0.5) / len(ious)) if ious else 0.0
    acc_07 = float(sum(1 for v in ious if v >= 0.7) / len(ious)) if ious else 0.0
    mean_dist = float(sum(dists) / len(dists)) if dists else 0.0
    mean_ratio = float(sum(ratios) / len(ratios)) if ratios else 0.0

    summary = {
        "num_samples": len(samples),
        "num_valid_iou": len(ious),
        "num_valid_center_distance": len(dists),
        "num_valid_area_ratio": len(ratios),
        "mean_iou": mean_iou,
        "acc_iou_0.5": acc_05,
        "acc_iou_0.7": acc_07,
        "mean_center_distance": mean_dist,
        "mean_area_ratio": mean_ratio,
    }

    print("\n==============================")
    print("Task 4 Object Grounding Evaluation Summary")
    print("==============================")
    print(f"Number of samples: {summary['num_samples']}")
    print(f"Number of samples with valid IoU: {summary['num_valid_iou']}")
    print(f"Mean IoU: {summary['mean_iou']:.4f}")
    print(f"Percentage with IoU ≥ 0.5: {summary['acc_iou_0.5']*100:.2f}%")
    print(f"Percentage with IoU ≥ 0.7: {summary['acc_iou_0.7']*100:.2f}%")
    print(f"Number of samples with valid center distance: {summary['num_valid_center_distance']}")
    print(f"Mean center distance: {summary['mean_center_distance']:.4f}")
    print(f"Number of samples with valid area ratio: {summary['num_valid_area_ratio']}")
    print(f"Mean area ratio (|B_pred|/|B_gold|): {summary['mean_area_ratio']:.4f}")

    return {
        "summary": summary,
        "details": results,
    }


def main(API_KEY):
    INPUT_FILE = "/home/MMID/object_grounding/task4_dataset_samples.json"
    OUTPUT_FILE = "/home/MMID/object_grounding/task4_gpt4o_evaluation_results.json"

    MAX_SAMPLES: Optional[int] = 10
    DELAY_SEC: float = 1.0
    client = OpenAI(api_key=API_KEY)

    print("=" * 60)
    print("Task4 Object Grounding - Evaluation using GPT-4o")
    print("=" * 60)

    print(f"\n1. Load the samples: {INPUT_FILE}")
    samples = load_samples(INPUT_FILE)
    print(f"   Total number of samples: {len(samples)}개")

    # MAX_SAMPLES 개수만큼 랜덤 샘플링 (여기서는 10개)
    if MAX_SAMPLES is not None and len(samples) > MAX_SAMPLES:
        random.seed(42)
        samples = random.sample(samples, MAX_SAMPLES)
        print(f"   Number of randomly sampled samples: {len(samples)} (MAX_SAMPLES={MAX_SAMPLES})")

    print("\n2. Evaluate using GPT-4o...")
    results = evaluate(samples, client, max_samples=None, delay_sec=DELAY_SEC)

    print(f"\n3. Save : {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nFinish!")
    print("=" * 60)


if __name__ == "__main__":
    API_KEY = "Your API Key"
    main(API_KEY)


