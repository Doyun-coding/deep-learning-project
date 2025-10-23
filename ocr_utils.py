import os, json, random, re
from typing import List, Tuple, Dict, Callable, Optional
from collections import defaultdict
import numpy as np
from PIL import Image, ImageOps

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

def list_json_basenames(dir_path: str, json_ext: str = ".json") -> List[str]:
    files = [f for f in os.listdir(dir_path) if f.endswith(json_ext) and "_4PR_" in f]
    return [os.path.splitext(f)[0] for f in files]

def safe_open_gray(path: str) -> Image.Image:
    return Image.open(path).convert("L")

HANGUL_ALNUM_PATTERN = re.compile(
    r"^[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3A-Za-z0-9]$"
)
HANGUL_ONLY_PATTERN = re.compile(
    r"^[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]$"
)

def default_char_filter(ch: str) -> bool:
    return bool(HANGUL_ALNUM_PATTERN.match(ch))

def HANGUL_ONLY(ch: str) -> bool:
    return bool(HANGUL_ONLY_PATTERN.match(ch))

def otsu_threshold(np_img: np.ndarray) -> int:
    hist = np.bincount(np_img.flatten(), minlength=256).astype(np.float64)
    total = np_img.size
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    return int(np.nanargmax(sigma_b2))

def crop_from_polygon(img: Image.Image, xs: List[int], ys: List[int], pad: int = 2) -> Image.Image:
    xmin, xmax = max(0, min(xs)), min(img.width, max(xs))
    ymin, ymax = max(0, min(ys)), min(img.height, max(ys))
    xmin, ymin = max(0, xmin - pad), max(0, ymin - pad)
    xmax, ymax = min(img.width, xmax + pad), min(img.height, ymax + pad)
    if xmin >= xmax or ymin >= ymax:
        return Image.new("L", (1, 1), 255)
    return img.crop((xmin, ymin, xmax, ymax))

def normalize_and_resize(np_img: np.ndarray, out_size: int = 32) -> np.ndarray:
    t = otsu_threshold(np_img)
    bin_img = (np_img < t).astype(np.uint8) * 255  # 글자=255(검), 배경=0
    pil = Image.fromarray(bin_img, mode="L")
    w, h = pil.size
    m = max(w, h)
    pad_l = (m - w) // 2
    pad_r = m - w - pad_l
    pad_t = (m - h) // 2
    pad_b = m - h - pad_t
    pil = ImageOps.expand(pil, border=(pad_l, pad_t, pad_r, pad_b), fill=0)
    pil = pil.resize((out_size, out_size), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    return arr

def load_split(
    split_dir: str,
    img_ext: str = ".png",
    json_ext: str = ".json",
    max_per_image: int = None,
    image_size: int = 32,
    filter_to_train_labels: Dict[str, int] = None,
    char_filter_fn: Optional[Callable[[str], bool]] = default_char_filter,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str], Dict[str, int]]:
    bases = list_json_basenames(split_dir, json_ext=json_ext)
    X, y = [], []
    label2idx: Dict[str, int] = {} if filter_to_train_labels is None else dict(filter_to_train_labels)
    stats = defaultdict(int)

    for bi, base in enumerate(bases, 1):
        jp = os.path.join(split_dir, base + json_ext)
        ip = os.path.join(split_dir, base + img_ext)
        if not (os.path.exists(jp) and os.path.exists(ip)):
            stats["skipped_missing"] += 1
            continue
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            img = safe_open_gray(ip)
        except Exception:
            stats["skipped_io"] += 1
            continue

        used_from_this = 0
        for bb in data.get("bbox", []):
            text = str(bb.get("data", "")).strip()
            xs, ys = bb.get("x", []), bb.get("y", [])
            if not text or len(text) != 1 or not xs or not ys:
                stats["skipped_non_single"] += 1
                continue
            if char_filter_fn is not None and not char_filter_fn(text):
                stats["skipped_char_filter"] += 1
                continue
            if filter_to_train_labels is not None and text not in filter_to_train_labels:
                stats["skipped_unknown_label"] += 1
                continue

            patch = crop_from_polygon(img, xs, ys, pad=2)
            patch_np = np.array(patch, dtype=np.uint8)
            norm = normalize_and_resize(patch_np, out_size=image_size)
            X.append(norm.flatten())

            if filter_to_train_labels is None:
                if text not in label2idx:
                    label2idx[text] = len(label2idx)
                y.append(label2idx[text])
            else:
                y.append(filter_to_train_labels[text])

            stats["used"] += 1
            used_from_this += 1
            if max_per_image is not None and used_from_this >= max_per_image:
                break

        if bi % 100 == 0:
            print(f"[{split_dir}] {bi}/{len(bases)} used={stats['used']}")

    if not X:
        raise RuntimeError(f"{split_dir}: usable samples not found (check PNG/JSON pairs & filters).")
    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.int32)
    idx2label = {v: k for k, v in label2idx.items()}
    return X, y, label2idx, idx2label, stats

def split_train_valid(X, y, valid_ratio=0.2, seed=42):
    set_seed(seed)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    n_val = int(len(X) * valid_ratio)
    vidx, tidx = idx[:n_val], idx[n_val:]
    return X[tidx], y[tidx], X[vidx], y[vidx]

def iterate_minibatches(X, y, batch_size=128, shuffle=True):
    N = len(X)
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, N, batch_size):
        sel = idx[i:i+batch_size]
        yield X[sel], y[sel]
