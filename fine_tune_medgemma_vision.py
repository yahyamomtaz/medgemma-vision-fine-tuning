#!/usr/bin/env python3
"""Fine-tune MedGemma 4B to count nuclei in BBBC006 z=16 images.

The BBBC006 count CSV and image archives use different UUID suffixes. Samples
are joined with the stable (well, site) identifiers, while only the w1
Hoechst/DAPI channel is used. Both sites from a well remain in the same split.

After training, deterministic generation runs on 10 held-out images by default.
Every raw output is printed and saved to ``eval_predictions.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


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
IMAGE_KEY_RE = re.compile(
    r"_([a-p]\d{2})_s([12])_w1[0-9a-f-]+\.tiff?$", re.IGNORECASE
)


@dataclass(frozen=True)
class CountSample:
    image_id: str
    image_path: str
    well: str
    site: str
    reference_count: int
    width: int
    height: int
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name", default="unsloth/medgemma-4b-it-bnb-4bit"
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path("data/BBBC006_v1_images_z_16"),
    )
    parser.add_argument(
        "--counts_file",
        type=Path,
        default=Path("data/BBBC006_v1_counts.csv"),
    )
    parser.add_argument(
        "--manifest_dir", type=Path, default=Path("data/bbbc006_z16_vision")
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models/medgemma-4b-it-sft-lora-bbbc006-z16"),
    )
    parser.add_argument("--eval_ratio", type=float, default=0.2)
    parser.add_argument("--prediction_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--finetune_vision_layers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--finetune_language_layers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_samples(images_dir: Path, counts_file: Path) -> list[CountSample]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not counts_file.is_file():
        raise FileNotFoundError(f"Counts file not found: {counts_file}")

    # UUID suffixes differ between the archive and CSV. Match the stable well
    # and site fields instead. w2 is phalloidin and is intentionally ignored.
    image_paths: dict[tuple[str, str], Path] = {}
    for path in sorted(images_dir.iterdir()):
        if path.suffix.casefold() not in {".tif", ".tiff"}:
            continue
        match = IMAGE_KEY_RE.search(path.name)
        if not match:
            continue
        key = (match.group(1).casefold(), match.group(2))
        if key in image_paths:
            raise ValueError(
                f"Duplicate z=16 w1 image for well/site {key}: "
                f"{image_paths[key]} and {path}"
            )
        image_paths[key] = path

    samples: list[CountSample] = []
    seen_keys: set[tuple[str, str]] = set()
    with counts_file.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Counts file missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            source_filename = (row["Image_FileName_OrigDAPI"] or "").strip()
            well = (row["Image_Metadata_Well"] or "").strip().upper()
            site = (row["Image_Metadata_Site"] or "").strip()
            if not well or not site:
                raise ValueError(f"Missing well/site metadata on row {row_number}")

            key = (well.casefold(), site)
            image_path = image_paths.get(key)
            if image_path is None:
                raise FileNotFoundError(
                    f"No z=16 w1 TIFF for row {row_number}: {well} site {site} "
                    f"(CSV source filename: {source_filename})"
                )
            if key in seen_keys:
                raise ValueError(f"Duplicate count row for well/site: {key}")
            seen_keys.add(key)

            count = int(row["Image_Count_Nuclei"])
            if count < 0:
                raise ValueError(f"Negative count on row {row_number}: {count}")
            with Image.open(image_path) as image:
                width, height, mode = image.width, image.height, image.mode
                image.verify()
            if (width, height) != (696, 520):
                raise ValueError(
                    f"Unexpected image dimensions for {image_path}: {width}x{height}"
                )

            samples.append(
                CountSample(
                    image_id=image_path.stem,
                    image_path=str(image_path),
                    well=well,
                    site=site,
                    reference_count=count,
                    width=width,
                    height=height,
                    mode=mode,
                )
            )

    missing_labels = sorted(set(image_paths) - seen_keys)
    if missing_labels:
        raise ValueError(f"z=16 w1 images without count labels: {missing_labels[:5]}")
    if not samples:
        raise ValueError("No matched BBBC006 z=16 w1 samples were loaded.")
    return sorted(samples, key=lambda sample: sample.image_id.casefold())


def split_by_well(
    samples: list[CountSample], eval_ratio: float, seed: int
) -> tuple[list[CountSample], list[CountSample]]:
    if not 0 < eval_ratio < 1:
        raise ValueError("--eval_ratio must be between 0 and 1")
    wells = sorted({sample.well for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(wells)
    evaluation_wells = set(wells[: round(len(wells) * eval_ratio)])
    train = [sample for sample in samples if sample.well not in evaluation_wells]
    evaluation = [sample for sample in samples if sample.well in evaluation_wells]
    if not train or not evaluation:
        raise ValueError("Split produced an empty train or evaluation set")
    return train, evaluation


def select_prediction_samples(
    evaluation: list[CountSample], sample_count: int, seed: int
) -> list[CountSample]:
    if sample_count < 1:
        raise ValueError("--prediction_samples must be at least 1")
    if sample_count > len(evaluation):
        raise ValueError(
            f"--prediction_samples ({sample_count}) exceeds the held-out set "
            f"({len(evaluation)})"
        )
    selected = list(evaluation)
    random.Random(seed + 1).shuffle(selected)
    return selected[:sample_count]


def load_rgb_16bit(path: str) -> Image.Image:
    """Percentile-scale one 16-bit fluorescence image to RGB."""
    with Image.open(path) as image:
        pixels = np.asarray(image, dtype=np.float32)
    lower, upper = np.percentile(pixels, (1.0, 99.5))
    if upper <= lower:
        raise ValueError(
            f"Image has no usable intensity range after percentile scaling: {path}"
        )
    scaled = np.clip(
        (pixels - lower) * 255.0 / (upper - lower), 0, 255
    ).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def manifest_record(sample: CountSample) -> dict[str, Any]:
    record = asdict(sample)
    record.update(
        {
            "instruction": INSTRUCTION,
            "answer": json.dumps(
                {"nuclei_count": sample.reference_count}, separators=(",", ":")
            ),
        }
    )
    return record


def write_manifests(
    manifest_dir: Path,
    train: list[CountSample],
    evaluation: list[CountSample],
    prediction_samples: list[CountSample],
    args: argparse.Namespace,
) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split, samples in (
        ("train", train),
        ("eval", evaluation),
        ("prediction_samples", prediction_samples),
    ):
        with (manifest_dir / f"{split}.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for sample in samples:
                handle.write(json.dumps(manifest_record(sample)) + "\n")

    summary = {
        "source": "BBBC006 v1, z=16 w1 Hoechst images",
        "target": "Image_Count_Nuclei from the official BBBC006 count CSV",
        "join_key": "(Image_Metadata_Well, Image_Metadata_Site); UUIDs differ",
        "preprocessing": (
            "per-image 1st-99.5th percentile scaling from 16-bit grayscale "
            "TIFF to 8-bit RGB"
        ),
        "split": "well-disjoint; both sites from a well remain in one split",
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "train_images": len(train),
        "eval_images": len(evaluation),
        "post_training_prediction_images": len(prediction_samples),
        "train_wells": len({sample.well for sample in train}),
        "eval_wells": len({sample.well for sample in evaluation}),
    }
    (manifest_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def as_conversation(sample: CountSample) -> dict[str, Any]:
    answer = json.dumps(
        {"nuclei_count": sample.reference_count}, separators=(",", ":")
    )
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
                "content": [{"type": "text", "text": answer}],
            },
        ]
    }


def parse_prediction(text: str) -> float | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned).get("nuclei_count")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    except (json.JSONDecodeError, AttributeError):
        return None
    return None


def evaluate_model(
    model: Any,
    processor: Any,
    samples: list[CountSample],
    output_dir: Path,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)
    results: list[dict[str, Any]] = []
    print("\n" + "=" * 72, flush=True)
    print(f"POST-TRAINING GENERATION ON {len(samples)} HELD-OUT IMAGES", flush=True)
    print("=" * 72, flush=True)

    for index, sample in enumerate(samples, start=1):
        image = load_rgb_16bit(sample.image_path)
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
        prediction_text = processor.tokenizer.decode(
            generated[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        ).strip()
        predicted_count = parse_prediction(prediction_text)
        absolute_error = (
            None
            if predicted_count is None
            else abs(predicted_count - sample.reference_count)
        )
        results.append(
            {
                **asdict(sample),
                "prediction": prediction_text,
                "predicted_nuclei_count": predicted_count,
                "absolute_error": absolute_error,
            }
        )
        print(f"\n[{index}/{len(samples)}] {sample.image_id}", flush=True)
        print(f"  target : {{\"nuclei_count\":{sample.reference_count}}}", flush=True)
        print(f"  output : {prediction_text}", flush=True)
        print(f"  parsed : {predicted_count}", flush=True)
        print(f"  abs_err: {absolute_error}", flush=True)

    valid = [row for row in results if row["predicted_nuclei_count"] is not None]
    errors = [
        row["predicted_nuclei_count"] - row["reference_count"] for row in valid
    ]
    metrics = {
        "samples": len(results),
        "valid_predictions": len(valid),
        "mae": (
            round(statistics.mean(abs(error) for error in errors), 4)
            if errors
            else None
        ),
        "rmse": (
            round(math.sqrt(statistics.mean(error * error for error in errors)), 4)
            if errors
            else None
        ),
        "mape": (
            round(
                100
                * statistics.mean(
                    abs(row["predicted_nuclei_count"] - row["reference_count"])
                    / row["reference_count"]
                    for row in valid
                    if row["reference_count"] > 0
                ),
                4,
            )
            if valid
            else None
        ),
        "note": (
            "Metrics cover the deterministic post-training subset of the "
            "held-out, well-disjoint development split."
        ),
    }
    result_path = output_dir / "eval_predictions.json"
    result_path.write_text(
        json.dumps({"metrics": metrics, "results": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n" + json.dumps(metrics, indent=2), flush=True)
    print(f"Saved prediction details to {result_path}", flush=True)
    return metrics


def train_model(
    args: argparse.Namespace,
    train_samples: list[CountSample],
    eval_samples: list[CountSample],
    prediction_samples: list[CountSample],
) -> None:
    # MedGemma's SigLIP head_dim=72 is incompatible with Flex Attention.
    os.environ["UNSLOTH_ENABLE_FLEX_ATTENTION"] = "0"

    # Unsloth must be imported before torch/transformers/trl.
    import unsloth  # noqa: F401
    import torch
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MedGemma vision fine-tuning.")

    print("=" * 72, flush=True)
    print("MedGemma 4B BBBC006 z=16 nuclei-counting fine-tuning", flush=True)
    print(f"GPU                 : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Model               : {args.model_name}", flush=True)
    print("Attention           : SDPA", flush=True)
    print(f"Train / eval        : {len(train_samples)} / {len(eval_samples)}", flush=True)
    print(f"Post-train examples : {len(prediction_samples)}", flush=True)
    print(f"Output              : {args.output_dir}", flush=True)
    print("=" * 72, flush=True)

    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        attn_implementation="sdpa",
    )
    processor.tokenizer.padding_side = "right"
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )

    train_dataset = [as_conversation(sample) for sample in train_samples]
    eval_dataset = [as_conversation(sample) for sample in eval_samples]
    collator = UnslothVisionDataCollator(
        model,
        processor,
        max_seq_length=args.max_seq_length,
        resize="min",
        train_on_responses_only=True,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
        completion_only_loss=True,
    )
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=0.3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        seed=args.seed,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_seq_length,
    )

    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor.tokenizer,
        data_collator=collator,
        args=training_args,
    )
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print(f"Training completed in {time.perf_counter() - started:.1f}s", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    evaluate_model(
        model,
        processor,
        prediction_samples,
        args.output_dir,
        args.max_new_tokens,
    )


def main() -> None:
    args = parse_args()
    samples = load_samples(args.images_dir, args.counts_file)
    train_samples, eval_samples = split_by_well(samples, args.eval_ratio, args.seed)
    prediction_samples = select_prediction_samples(
        eval_samples, args.prediction_samples, args.seed
    )
    write_manifests(
        args.manifest_dir,
        train_samples,
        eval_samples,
        prediction_samples,
        args,
    )

    summary = {
        "matched_z16_w1_images": len(samples),
        "train_images": len(train_samples),
        "eval_images": len(eval_samples),
        "prediction_images": len(prediction_samples),
        "train_wells": len({sample.well for sample in train_samples}),
        "eval_wells": len({sample.well for sample in eval_samples}),
    }
    print(json.dumps(summary, indent=2), flush=True)
    print(
        "Post-training prediction sample IDs:\n  "
        + "\n  ".join(sample.image_id for sample in prediction_samples),
        flush=True,
    )
    if args.dry_run:
        print("Dry run complete; model was not loaded.", flush=True)
        return
    train_model(args, train_samples, eval_samples, prediction_samples)


if __name__ == "__main__":
    main()
