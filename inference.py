#!/usr/bin/env python3
"""Run BBBC006 nucleus-count inference with the fine-tuned MedGemma adapter."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MODEL = "yayamomt/medgemma-4b-it-bbbc006-nuclei-count-lora"
INSTRUCTION = (
    "Count the Hoechst-stained U2OS cell nuclei in this fluorescence microscopy "
    "image. Return only a JSON object with one integer field named nuclei_count."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images",
        type=Path,
        nargs="+",
        help="One or more z=16, w1 Hoechst/DAPI TIFF images.",
    )
    parser.add_argument(
        "--model_name_or_path",
        "--model",
        dest="model_name_or_path",
        default=DEFAULT_MODEL,
        help=(
            "Hugging Face adapter ID or local adapter directory "
            f"(default: {DEFAULT_MODEL})."
        ),
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Optional path at which to save all predictions as JSON.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32)
    return parser.parse_args()


def load_rgb_16bit(path: Path) -> Image.Image:
    """Apply the percentile scaling used for training and evaluation."""
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.casefold() not in {".tif", ".tiff"}:
        raise ValueError(f"Expected a TIFF image, received: {path}")

    with Image.open(path) as image:
        pixels = np.asarray(image, dtype=np.float32)
    if pixels.ndim != 2:
        raise ValueError(f"Expected a single-channel grayscale image: {path}")

    lower, upper = np.percentile(pixels, (1.0, 99.5))
    if upper <= lower:
        raise ValueError(
            f"Image has no usable intensity range after percentile scaling: {path}"
        )

    scaled = np.clip(
        (pixels - lower) * 255.0 / (upper - lower), 0, 255
    ).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def parse_prediction(text: str) -> int | None:
    """Parse the model's JSON response without hiding its raw output."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned).get("nuclei_count")
    except (json.JSONDecodeError, AttributeError):
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < 0:
        return None
    if not numeric_value.is_integer():
        return None
    return int(numeric_value)


def predict(
    model: Any,
    processor: Any,
    image_path: Path,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": INSTRUCTION},
                {"type": "image"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=prompt,
        images=load_rgb_16bit(image_path),
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )

    raw_response = processor.tokenizer.decode(
        generated[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()
    return {
        "image_path": str(image_path),
        "raw_response": raw_response,
        "predicted_nuclei_count": parse_prediction(raw_response),
    }


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be at least 1")

    # Validate and preprocess every input before downloading or loading the model.
    for image_path in args.images:
        load_rgb_16bit(image_path)

    os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"
    import torch
    from unsloth import FastVisionModel

    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA GPU is required for inference.")

    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_name_or_path,
        load_in_4bit=True,
        attn_implementation="sdpa",
    )
    processor.tokenizer.padding_side = "right"
    FastVisionModel.for_inference(model)
    model.eval()

    results = [
        predict(model, processor, image_path, args.max_new_tokens)
        for image_path in args.images
    ]
    for result in results:
        print(json.dumps(result, ensure_ascii=False), flush=True)

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(
            json.dumps(
                {
                    "model_name_or_path": args.model_name_or_path,
                    "instruction": INSTRUCTION,
                    "predictions": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved predictions to {args.output_file}", flush=True)


if __name__ == "__main__":
    main()
