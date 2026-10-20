"""
Usage:
    uv run main.py equation.jpg            # full pipeline
    uv run main.py --latex "x^2+2x-8=0"    # skip OCR, test parse + solve only
    uv run main.py equation.jpg --debug    # also saves the preprocessed image
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import sympy as sp
from PIL import Image
from sympy.parsing.latex import parse_latex


# Preprocessing: clean up a phone photo so the OCR model sees clean ink

def preprocess(path: str, max_side: int = 1024) -> Image.Image:
  gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  if gray is None:
      raise FileNotFoundError(f"Could not read image: {path}")
  h, w = gray.shape
  s = max_side / max(h, w)
  gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
  h, w = gray.shape

  # ink = 255 (inverted threshold)
  gray = cv2.GaussianBlur(gray, (3, 3), 0)
  ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 31, 15)

  # 1) remove thin lines (grid)
  disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, disk)

  # 2) keep only big blobs (merge fragments first so broken strokes count as one)
  merged = cv2.dilate(opened, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
  n, lab, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
  keep = np.zeros_like(opened)
  for i in range(1, n):
      if max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= 40:
          keep[lab == i] = 255
  seed = cv2.bitwise_and(opened, keep)

  # keep only the horizontal band where most ink is (the equation line)
  ys, xs = np.where(seed > 0)
  if len(ys) == 0:
      raise ValueError("No handwriting found after grid removal")
  mid, half = int(np.median(ys)), int(0.15 * h)
  band = np.zeros_like(seed)
  band[max(mid - half, 0): mid + half] = 255
  seed = cv2.bitwise_and(seed, band)

  # 3) grow strokes back into the original ink to repair gaps
  cur, k3 = seed.copy(), np.ones((3, 3), np.uint8)
  for _ in range(7):
      cur = cv2.bitwise_and(cv2.dilate(cur, k3), ink)

  # crop + pad, return black ink on white
  ys, xs = np.where(cur > 0)
  pad = 30
  y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, h)
  x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, w)
  out = 255 - cur[y0:y1, x0:x1]
  out = cv2.copyMakeBorder(out, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
  return Image.fromarray(out).convert("RGB")

# Recognition: image -> LaTeX string (swap this function to change models)
_ocr_model = None

VLM_ID = "Qwen/Qwen2-VL-2B-Instruct"
PROMPT = ("Transcribe the handwritten math in this image to LaTeX. "
          "Output only the LaTeX, with no explanation and no code fences.")
_vlm = None


def image_to_latex(img: Image.Image) -> str:
    global _vlm
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if _vlm is None:
        proc = AutoProcessor.from_pretrained(
            VLM_ID, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28
        )
        model = AutoModelForImageTextToText.from_pretrained(
            VLM_ID, dtype=torch.bfloat16, device_map="cuda"
        )
        _vlm = (proc, model)
    proc, model = _vlm

    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    out = out[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(out, skip_special_tokens=True)[0]

# Cleanup + PARSE: LaTeX string -> SymPy object
def clean_latex(s: str) -> str:
  s = re.sub(r"```(?:latex)?", "", s)
  s = re.sub(r"\\\[|\\\]|\\\(|\\\)", "", s)
  s = s.strip().strip("$").strip()
  s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
  s = re.sub(r"\\(left|right)\s*", "", s)
  s = re.sub(r"\\[,;:!]|\\ ", " ", s)
  s = s.rstrip(".,")
  return s


def parse(latex: str):
    # lark backend avoids the antlr4 version pin headache
    return parse_latex(latex, backend="lark")


# Solve: SymPy does the real math (exact), numpy only for numeric extras
def pick_variable(expr):
    syms = sorted(expr.free_symbols, key=str)
    for s in syms:
        if str(s) == "x":
            return s
    return syms[0] if syms else None


def solve(expr) -> dict:
    # Equation: solve for a variable
    if isinstance(expr, sp.Equality):
        var = pick_variable(expr)
        if var is None:
          return {"kind": "truth check", "result": sp.simplify(expr.lhs - expr.rhs) == 0} # type: ignore[operator]
        sols = sp.solve(expr, var)
        return {
            "kind": "equation",
            "variable": var,
            "solutions": sols,
            "numeric": [sp.N(s) for s in sols],
        }

    # Inequality
    if isinstance(expr, sp.core.relational.Relational):
        var = pick_variable(expr)
        return {"kind": "inequality", "result": sp.solve_univariate_inequality(expr, var)}

    # Plain expression: simplify / evaluate
    result = {"kind": "expression", "simplified": sp.simplify(expr)}
    if not expr.free_symbols:
        result["numeric"] = sp.N(expr)
    return result


# Pipeline glue
def run(latex: str) -> None:
    latex = clean_latex(latex)
    print(f"LaTeX      : {latex}")
    if len(latex) > 120:
        print("Recognition looks like garbage (too long).")
        return
    try:
        expr = parse(latex)
    except Exception as e:
        print(f"Could not parse that LaTeX ({type(e).__name__}). Try --latex to fix it by hand.")
        return
    print(f"SymPy      : {expr}")
    for k, v in solve(expr).items():
        print(f"{k:<11}: {v}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="path to image with one equation")
    ap.add_argument("--latex", help="skip OCR and use this LaTeX directly")
    ap.add_argument("--debug", action="store_true", help="save preprocessed image")
    args = ap.parse_args()

    if args.latex:
        run(args.latex)
        return
    if not args.image:
        ap.error("give an image path or --latex")

    img = preprocess(args.image)
    if args.debug:
        out = Path(args.image).with_suffix(".pre.png")
        img.save(out)
        print(f"Saved preprocessed image -> {out}")

    run(image_to_latex(img))


if __name__ == "__main__":
    main()
