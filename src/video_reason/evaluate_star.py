"""Evaluate a local Qwen3.5 checkpoint on STAR video multiple-choice questions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from video_reason.common.qwen import generate_answer, load_model
from video_reason.common.star import (
    CATEGORIES,
    category,
    gold_letter,
    parse_prediction,
    prompt,
    selected_rows,
)
from video_reason.common.video import decode_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", type=Path, default=Path("/data/wyt/STAR/STAR_val.json")
    )
    parser.add_argument(
        "--videos", type=Path, default=Path("/data/wyt/charades/Charades_v1_480")
    )
    parser.add_argument(
        "--model", type=Path, default=Path("/data/wyt/Qwen/Qwen3.5-0.8B")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/star_baseline.jsonl")
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Round-robin sample across four categories; 0 means all",
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument(
        "--image-size", type=int, default=336, help="Maximum frame width or height"
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check labels and decode frames without loading model",
    )
    args = parser.parse_args()
    if (
        args.limit < 0
        or args.frames < 1
        or args.image_size < 32
        or args.max_new_tokens < 1
        or args.num_shards < 1
        or not 0 <= args.shard_index < args.num_shards
    ):
        parser.error(
            "invalid limit, frames, image-size, max-new-tokens, or shard settings"
        )
    return args


def main() -> None:
    args = parse_args()
    rows = selected_rows(args.annotations, args.limit)
    rows = rows[args.shard_index :: args.num_shards]
    if not rows:
        raise ValueError("No questions selected")
    model = processor = device = None
    if not args.validate_only:
        model, processor, device = load_model(args.model, args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(lambda: {"correct": 0, "total": 0, "labeled": 0, "invalid": 0})
    output_file = (
        args.output.open("w", encoding="utf-8") if not args.validate_only else None
    )
    try:
        for index, row in enumerate(rows, 1):
            gold = gold_letter(row) if "answer" in row else None
            video_path = args.videos / f"{row['video_id']}.mp4"
            frames, timestamps, clip_duration = decode_clip(
                video_path,
                float(row["start"]),
                float(row["end"]),
                args.frames,
                args.image_size,
            )
            response = prediction = None
            elapsed = None
            if model is not None:
                started = time.perf_counter()
                response = generate_answer(
                    model,
                    processor,
                    device,
                    frames,
                    clip_duration,
                    prompt(row),
                    args.max_new_tokens,
                )
                elapsed = round(time.perf_counter() - started, 3)
                prediction = parse_prediction(response)
            group = category(row)
            counts[group]["total"] += 1
            if gold is not None:
                counts[group]["labeled"] += 1
            if prediction is not None and gold is not None:
                counts[group]["correct"] += int(prediction == gold)
            if prediction is None and model is not None:
                counts[group]["invalid"] += 1
            record = {
                "question_id": row["question_id"],
                "video_id": row["video_id"],
                "category": group,
                "start": row["start"],
                "end": row["end"],
                "frame_times": [round(t, 3) for t in timestamps],
                "question": row["question"],
                "choices": [x["choice"] for x in row["choices"]],
                "gold": gold,
                "response": response,
                "prediction": prediction,
                "correct": prediction == gold
                if model is not None and gold is not None
                else None,
                "seconds": elapsed,
            }
            if output_file:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
            print(
                f"{index}/{len(rows)} {row['question_id']} gold={gold} pred={prediction} time={elapsed}",
                flush=True,
            )
    finally:
        if output_file:
            output_file.close()
    if model is not None:
        total = sum(value["total"] for value in counts.values())
        labeled = sum(value["labeled"] for value in counts.values())
        correct = sum(value["correct"] for value in counts.values())
        invalid = sum(value["invalid"] for value in counts.values())
        print(f"Completed: {total}; invalid={invalid}")
        if labeled:
            print(f"Overall accuracy: {correct}/{labeled} = {correct / labeled:.1%}")
        for name in CATEGORIES:
            value = counts[name]
            if value["total"]:
                if value["labeled"]:
                    print(
                        f"{name}: {value['correct']}/{value['labeled']} = {value['correct'] / value['labeled']:.1%}; invalid={value['invalid']}"
                    )
                else:
                    print(
                        f"{name}: completed={value['total']}; invalid={value['invalid']}"
                    )
        print(f"Predictions: {args.output}")
    else:
        print(
            f"Validated {len(rows)} annotations and decoded {args.frames} frames per clip"
        )


if __name__ == "__main__":
    main()
