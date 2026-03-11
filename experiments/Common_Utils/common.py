#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch

KST = timezone(timedelta(hours=9))

def kst_timestamp() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_device_map(x: str):
    x = (x or "auto").lower()
    if x == "auto":
        return "auto"
    if x == "cuda":
        return {"": 0}
    return x
