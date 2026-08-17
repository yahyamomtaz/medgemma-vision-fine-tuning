#!/usr/bin/env python3
"""Evaluate a MedGemma model on the held-out BBBC006 z=16 split.

By default, this script generates predictions for all 154 images in the
well-disjoint evaluation manifest created by ``fine_tune_medgemma_vision.py``.
It prints every target and raw response, then saves full per-image results and
aggregate counting metrics. The model may be a local fine-tuned adapter or a
Hugging Face model ID such as the original base model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

from fine_tune_medgemma_vision import load_rgb_16bit, parse_prediction


REQUIRED_FIELDS = {
    "image_id",
    "image_path",
    "well",
    "site",
    "reference_count",
    "instruction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name_or_path",
        "--model_dir",
        dest="model_name_or_path",
        default="models/medgemma-4b-it-sft-lora-bbbc006-z16",
        help="Local adapter directory or Hugging Face model ID.",
    )
    parser.add_argument(
        "--eval_file",
        type=Path,
        default=Path("data/bbbc006_z16_vision/eval.jsonl"),
    )
    parser.add_argument(
        "--train_file",
        type=Path,
        default=Path("data/bbbc006_z16_vision/train.jsonl"),
        help="Used only to calculate the training-mean baseline.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Defaults beside a local adapter; remote models require this option.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional prefix limit for debugging; omit to evaluate all images.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate manifests and image paths without loading the model.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(
                    f"Missing fields in {path} at line {line_number}: "
                    f"{sorted(missing)}"
                )
            image_path = Path(row["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image from {path} line {line_number} not found: {image_path}"
                )
            reference_count = row["reference_count"]
            if not isinstance(reference_count, int) or reference_count < 0:
                raise ValueError(
                    f"Invalid reference_count in {path} line {line_number}: "
                    f"{reference_count!r}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest contains no samples: {path}")
    image_ids = [row["image_id"] for row in rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"Duplicate image_id values in manifest: {path}")
    return rows


def training_mean(train_file: Path) -> float:
    rows = load_manifest(train_file)
    return statistics.mean(row["reference_count"] for row in rows)


def compute_metrics(
    results: list[dict[str, Any]], train_reference_mean: float
) -> dict[str, Any]:
    valid = [row for row in results if row["predicted_nuclei_count"] is not None]
    errors = [
        row["predicted_nuclei_count"] - row["reference_count"] for row in valid
    ]
    references = [row["reference_count"] for row in valid]
    predictions = [row["predicted_nuclei_count"] for row in valid]

    mae = statistics.mean(abs(error) for error in errors) if errors else None
    rmse = (
        math.sqrt(statistics.mean(error * error for error in errors))
        if errors
        else None
    )
    nonzero_rows = [row for row in valid if row["reference_count"] > 0]
    mape = (
        100
        * statistics.mean(
            abs(row["predicted_nuclei_count"] - row["reference_count"])
            / row["reference_count"]
            for row in nonzero_rows
        )
        if nonzero_rows
        else None
    )

    if len(references) > 1:
        reference_mean = statistics.mean(references)
        total_sum_squares = sum(
            (reference - reference_mean) ** 2 for reference in references
        )
        residual_sum_squares = sum(error * error for error in errors)
        r_squared = (
            1.0 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0
            else None
        )
    else:
        r_squared = None

    all_references = [row["reference_count"] for row in results]
    baseline_errors = [
        train_reference_mean - reference for reference in all_references
    ]
    return {
        "samples": len(results),
        "valid_predictions": len(valid),
        "valid_prediction_rate": round(len(valid) / len(results), 6),
        "mae": round(mae, 4) if mae is not None else None,
        "rmse": round(rmse, 4) if rmse is not None else None,
        "mape": round(mape, 4) if mape is not None else None,
        "mean_error_bias": (
            round(statistics.mean(errors), 4) if errors else None
        ),
        "median_absolute_error": (
            round(statistics.median(abs(error) for error in errors), 4)
            if errors
            else None
        ),
        "r_squared": round(r_squared, 6) if r_squared is not None else None,
        "within_5_nuclei": sum(abs(error) <= 5 for error in errors),
        "within_10_nuclei": sum(abs(error) <= 10 for error in errors),
        "mean_reference_count": (
            round(statistics.mean(references), 4) if references else None
        ),
        "mean_predicted_count": (
            round(statistics.mean(predictions), 4) if predictions else None
        ),
        "train_mean_baseline": {
            "prediction": round(train_reference_mean, 4),
            "mae": round(
                statistics.mean(abs(error) for error in baseline_errors), 4
            ),
            "rmse": round(
                math.sqrt(statistics.mean(error * error for error in baseline_errors)),
                4,
            ),
        },
        "note": (
            "Held-out, well-disjoint BBBC006 z=16 development split. "
            "Counts are automated reference labels."
        ),
    }


def evaluate(
    model: Any,
    processor: Any,
    samples: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        image = load_rgb_16bit(sample["image_path"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["instruction"]},
                    {"type": "image"},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=prompt,
            images=image,
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
        raw_prediction = processor.tokenizer.decode(
            generated[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()
        predicted_count = parse_prediction(raw_prediction)
        absolute_error = (
            None
            if predicted_count is None
            else abs(predicted_count - sample["reference_count"])
        )
        result = {
            **sample,
            "prediction": raw_prediction,
            "predicted_nuclei_count": predicted_count,
            "absolute_error": absolute_error,
        }
        results.append(result)

        print(f"\n[{index}/{len(samples)}] {sample['image_id']}", flush=True)
        print(
            f"  target : {{\"nuclei_count\":{sample['reference_count']}}}",
            flush=True,
        )
        print(f"  output : {raw_prediction}", flush=True)
        print(f"  parsed : {predicted_count}", flush=True)
        print(f"  abs_err: {absolute_error}", flush=True)

    print(
        f"\nGeneration completed in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return results


def main() -> None:
    args = parse_args()
    samples = load_manifest(args.eval_file)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max_samples must be at least 1")
        samples = samples[: args.max_samples]
    train_reference_mean = training_mean(args.train_file)
    model_path = Path(args.model_name_or_path)
    if args.output_file is not None:
        output_file = args.output_file
    elif args.model_name_or_path.startswith(("models/", "./", "/")):
        output_file = model_path / "full_eval_predictions.json"
    else:
        raise ValueError(
            "--output_file is required when evaluating a Hugging Face model ID"
        )

    print("=" * 72, flush=True)
    print("MedGemma BBBC006 z=16 standalone evaluation", flush=True)
    print(f"Model          : {args.model_name_or_path}", flush=True)
    print(f"Evaluation file: {args.eval_file}", flush=True)
    print(f"Images         : {len(samples)}", flush=True)
    print(f"Wells          : {len({row['well'] for row in samples})}", flush=True)
    print(f"Output         : {output_file}", flush=True)
    print(f"Train mean     : {train_reference_mean:.4f}", flush=True)
    print("=" * 72, flush=True)

    # Exercise the exact preprocessing path during a CPU-only dry run.
    preview = load_rgb_16bit(samples[0]["image_path"])
    if preview.mode != "RGB" or preview.size != (696, 520):
        raise ValueError(
            f"Unexpected normalized image properties: {preview.mode} {preview.size}"
        )
    if args.dry_run:
        print("Dry run complete; manifests and preprocessing are valid.", flush=True)
        return

    if args.model_name_or_path.startswith(("models/", "./", "/")):
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Local model directory not found: {args.model_name_or_path}"
            )
    os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"
    import unsloth  # noqa: F401
    import torch
    from unsloth import FastVisionModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MedGemma vision evaluation.")
    print(f"GPU            : {torch.cuda.get_device_name(0)}", flush=True)
    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_name_or_path,
        load_in_4bit=True,
        attn_implementation="sdpa",
    )
    processor.tokenizer.padding_side = "right"
    FastVisionModel.for_inference(model)
    model.eval()

    results = evaluate(model, processor, samples, args.max_new_tokens)
    metrics = compute_metrics(results, train_reference_mean)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(
            {
                "model_name_or_path": args.model_name_or_path,
                "eval_file": str(args.eval_file),
                "metrics": metrics,
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n" + "=" * 72, flush=True)
    print("FULL EVALUATION METRICS", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved full results to {output_file}", flush=True)


if __name__ == "__main__":
    main()
