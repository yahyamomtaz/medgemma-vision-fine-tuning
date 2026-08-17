#!/usr/bin/env python3
"""Fine-tune base MedGemma 4B to count nuclei in BBBC006 v1 images.

BBBC006 contains two Hoechst-stained fields of view from each of 384 wells.
The supplied counts were generated automatically at the optimal focal plane.
This script uses the z=00 images present locally and keeps both sites from a
well in the same split to prevent site-level leakage.

The model is trained from the instruction-tuned
``unsloth/medgemma-4b-it-bnb-4bit`` checkpoint by default; it does not load or
modify the text-only V7 adapter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import fine_tune_medgemma_vision as vision


INSTRUCTION = (
    "Count the Hoechst-stained U2OS cell nuclei in this fluorescence microscopy "
    "image. Return only a JSON object with one integer field named nuclei_count."
)
REQUIRED_COLUMNS = {
    "Image_Count_Nuclei",
    "Image_FileName_OrigDAPI",
    "Image_Metadata_Site",
    "Image_Metadata_Well",
}
IMAGE_KEY_RE = re.compile(r"_([a-p]\d{2})_s([12])_w1[0-9a-f-]+\.tif$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="unsloth/medgemma-4b-it-bnb-4bit")
    parser.add_argument("--images_dir", type=Path, default=Path("data/BBBC006"))
    parser.add_argument("--counts_file", type=Path, default=Path("data/BBBC006.csv"))
    parser.add_argument("--manifest_dir", type=Path, default=Path("data/bbbc006_vision"))
    parser.add_argument("--output_dir", type=Path, default=Path("models/medgemma-4b-it-sft-lora-bbbc006"))
    parser.add_argument("--eval_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--finetune_vision_layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune_language_layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_samples(images_dir: Path, counts_file: Path) -> list[vision.CountSample]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not counts_file.is_file():
        raise FileNotFoundError(f"Counts file not found: {counts_file}")

    # z=00 filenames have image-plane-specific UUIDs, whereas the CSV lists the
    # z=16 filename UUIDs. Join on the stable well/site pair instead. Only w1
    # is Hoechst/DAPI; w2 is the phalloidin channel and is intentionally ignored.
    image_paths: dict[tuple[str, str], Path] = {}
    for path in images_dir.glob("*.tif"):
        match = IMAGE_KEY_RE.search(path.name)
        if not match:
            continue
        key = (match.group(1).casefold(), match.group(2))
        if key in image_paths:
            raise ValueError(f"Duplicate z=00 Hoechst image for well/site {key}: {path}")
        image_paths[key] = path
    samples: list[vision.CountSample] = []
    seen_keys: set[tuple[str, str]] = set()
    with counts_file.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Counts file missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            filename = (row["Image_FileName_OrigDAPI"] or "").strip()
            well = (row["Image_Metadata_Well"] or "").strip().upper()
            site = (row["Image_Metadata_Site"] or "").strip()
            key = (well.casefold(), site)
            image_path = image_paths.get(key)
            if image_path is None:
                raise FileNotFoundError(f"No z=00 Hoechst TIFF for counts row {row_number}: {well} site {site} ({filename})")
            if key in seen_keys:
                raise ValueError(f"Duplicate counts row for well/site: {key}")
            seen_keys.add(key)
            count = int(row["Image_Count_Nuclei"])
            if not well or not site:
                raise ValueError(f"Missing well/site metadata on counts row {row_number}")
            with Image.open(image_path) as image:
                width, height, mode = image.width, image.height, image.mode
                image.verify()
            samples.append(vision.CountSample(
                image_id=image_path.stem,
                image_path=str(image_path),
                group=well,  # group is the well; both sites stay together.
                human_counter_1=count,
                human_counter_2=count,
                consensus_count=float(count),
                width=width,
                height=height,
                mode=mode,
            ))
    if set(image_paths) != seen_keys:
        extra = sorted(set(image_paths) - seen_keys)
        raise ValueError(f"Images without count labels: {extra[:5]}")
    return sorted(samples, key=lambda sample: sample.image_id.casefold())


def split_by_well(samples: list[vision.CountSample], eval_ratio: float, seed: int) -> tuple[list[vision.CountSample], list[vision.CountSample]]:
    if not 0 < eval_ratio < 1:
        raise ValueError("--eval_ratio must be between 0 and 1")
    wells = sorted({sample.group for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(wells)
    evaluation_wells = set(wells[:round(len(wells) * eval_ratio)])
    train = [sample for sample in samples if sample.group not in evaluation_wells]
    evaluation = [sample for sample in samples if sample.group in evaluation_wells]
    if not train or not evaluation:
        raise ValueError("Split produced an empty train or evaluation set")
    return train, evaluation


def load_rgb_16bit(path: str) -> Image.Image:
    """Percentile-scale a 16-bit fluorescence image to RGB for MedGemma."""
    with Image.open(path) as image:
        pixels = np.asarray(image, dtype=np.float32)
    lower, upper = np.percentile(pixels, (1.0, 99.5))
    if upper <= lower:
        scaled = np.zeros_like(pixels, dtype=np.uint8)
    else:
        scaled = np.clip((pixels - lower) * 255.0 / (upper - lower), 0, 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def write_manifests(manifest_dir: Path, train: list[vision.CountSample], evaluation: list[vision.CountSample], args: argparse.Namespace) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split, samples in (("train", train), ("eval", evaluation)):
        with (manifest_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for sample in samples:
                row = asdict(sample)
                row.update({
                    "instruction": INSTRUCTION,
                    "answer": json.dumps({"nuclei_count": int(sample.consensus_count)}, separators=(",", ":")),
                    "well": sample.group,
                })
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": "BBBC006 v1, z=00 Hoechst images",
        "target": "Image_Count_Nuclei supplied by BBBC006; automated count at optimal focus, applied to the same field of view",
        "preprocessing": "per-image 1st–99.5th percentile scaling from 16-bit grayscale TIFF to 8-bit RGB",
        "split": "well-disjoint random split; both sites from a well remain in one split",
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "train_images": len(train),
        "eval_images": len(evaluation),
        "train_wells": len({sample.group for sample in train}),
        "eval_wells": len({sample.group for sample in evaluation}),
    }
    (manifest_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def as_conversation(sample: vision.CountSample) -> dict[str, Any]:
    """Build the exact image/text/answer format used by BBBC006 training.

    This cannot reuse the BBBC002 helper because its target key is ``cell_count``.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": INSTRUCTION},
                    {"type": "image", "image": load_rgb_16bit(sample.image_path)},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"nuclei_count": int(sample.consensus_count)},
                            separators=(",", ":"),
                        ),
                    }
                ],
            },
        ]
    }


def parse_prediction(text: str) -> float | None:
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        value = json.loads(cleaned).get("nuclei_count")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    except (json.JSONDecodeError, AttributeError):
        return None
    return None


def evaluate_model(model: Any, processor: Any, samples: list[vision.CountSample], output_dir: Path, max_new_tokens: int) -> dict[str, Any]:
    import torch
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)
    results = []
    for index, sample in enumerate(samples, start=1):
        image = load_rgb_16bit(sample.image_path)
        prompt = processor.apply_chat_template([{"role": "user", "content": [
            {"type": "text", "text": INSTRUCTION}, {"type": "image"}
        ]}], add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, images=image, add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)
        prediction = processor.tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        value = parse_prediction(prediction)
        results.append({**asdict(sample), "prediction": prediction, "predicted_nuclei_count": value})
        print(f"  eval [{index}/{len(samples)}] {sample.image_id}: target={sample.consensus_count:.0f} prediction={value}", flush=True)
    valid = [row for row in results if row["predicted_nuclei_count"] is not None]
    errors = [row["predicted_nuclei_count"] - row["consensus_count"] for row in valid]
    metrics = {
        "samples": len(results), "valid_predictions": len(valid),
        "mae": round(statistics.mean(abs(error) for error in errors), 4) if errors else None,
        "rmse": round(math.sqrt(statistics.mean(error * error for error in errors)), 4) if errors else None,
        "mape": round(100 * statistics.mean(abs(error) / row["consensus_count"] for error, row in zip(errors, valid)), 4) if errors else None,
        "note": "Held-out, well-disjoint development split. Counts are automated BBBC006 reference labels, not independent manual annotations.",
    }
    (output_dir / "eval_predictions.json").write_text(json.dumps({"metrics": metrics, "results": results}, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    samples = load_samples(args.images_dir, args.counts_file)
    train, evaluation = split_by_well(samples, args.eval_ratio, args.seed)
    write_manifests(args.manifest_dir, train, evaluation, args)
    print(json.dumps({"images": len(samples), "train": len(train), "eval": len(evaluation), "train_wells": len({s.group for s in train}), "eval_wells": len({s.group for s in evaluation})}, indent=2))
    if args.dry_run:
        return
    # Reuse the battle-tested Unsloth trainer, but replace BBBC002-specific prompt,
    # TIFF conversion, and evaluation with BBBC006-specific implementations.
    vision.INSTRUCTION = INSTRUCTION
    vision.load_rgb = load_rgb_16bit
    vision.as_conversation = as_conversation
    vision.evaluate_model = evaluate_model
    vision.train_model(args, train, evaluation)


if __name__ == "__main__":
    main()
