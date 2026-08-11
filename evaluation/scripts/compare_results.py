#!/usr/bin/env python3
"""Combine FID, CLIP, and FER JSON files into CSV and Markdown."""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json


def main():
    """Parse arguments, combine metrics from multiple runs, and export to CSV and Markdown."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        nargs=4,
        action="append",
        metavar=("NAME", "FID_JSON", "CLIP_JSON", "FER_JSON"),
        required=True,
        help="Add a run with its FID, CLIP, and FER JSON files"
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=Path,
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--markdown",
        dest="md_path",
        type=Path,
        required=True,
        help="Output Markdown file path"
    )
    args = parser.parse_args()

    rows = []
    for name, fid_path, clip_path, fer_path in args.run:
        fid = load_json(Path(fid_path))
        clip = load_json(Path(clip_path))
        fer = load_json(Path(fer_path))
        rows.append({
            "run": name,
            "FID ↓": fid.get("overall_fid", ""),
            "CLIP agreement ↑": clip.get("zero_shot_accuracy", ""),
            "CLIP target cosine ×100 ↑": clip.get("mean_target_cosine_x100", ""),
            "FER accuracy ↑": fer.get("accuracy", ""),
            "FER macro F1 ↑": fer.get("macro_f1", ""),
            "generated images": clip.get("image_count", "")
        })

    # Write CSV
    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Write Markdown
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for row in rows:
        formatted_values = []
        for h in headers:
            value = row[h]
            if isinstance(value, float):
                formatted_values.append(f"{value:.4f}")
            else:
                formatted_values.append(str(value))
        lines.append("| " + " | ".join(formatted_values) + " |")

    args.md_path.parent.mkdir(parents=True, exist_ok=True)
    args.md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved CSV: {args.csv_path}")
    print(f"Saved Markdown: {args.md_path}")


if __name__ == "__main__":
    main()
