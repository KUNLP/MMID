#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Any, Dict, List, Set

def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Check the format...")
        return data

    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    raise ValueError("Supported Extensions: 'jsonl' or 'json'")

def load_processed_indices(predictions_jsonl: Path) -> Set[int]:
    done: Set[int] = set()
    with predictions_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "idx" in rec:
                    done.add(int(rec["idx"]))
            except Exception:
                continue
    return done
