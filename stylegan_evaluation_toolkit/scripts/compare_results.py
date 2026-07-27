#!/usr/bin/env python3
"""Combine FID, CLIP, and FER JSON files into CSV and Markdown."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_json

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run",nargs=4,action="append",metavar=("NAME","FID_JSON","CLIP_JSON","FER_JSON"),required=True); p.add_argument("--csv",dest="csv_path",type=Path,required=True); p.add_argument("--markdown",dest="md_path",type=Path,required=True); args=p.parse_args()
    rows=[]
    for name,fid_path,clip_path,fer_path in args.run:
        fid,clip,fer=load_json(Path(fid_path)),load_json(Path(clip_path)),load_json(Path(fer_path))
        rows.append({"run":name,"FID ↓":fid.get("overall_fid",""),"CLIP agreement ↑":clip.get("zero_shot_accuracy",""),"CLIP target cosine ×100 ↑":clip.get("mean_target_cosine_x100",""),"FER accuracy ↑":fer.get("accuracy",""),"FER macro F1 ↑":fer.get("macro_f1",""),"generated images":clip.get("image_count","")})
    args.csv_path.parent.mkdir(parents=True,exist_ok=True)
    with args.csv_path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    headers=list(rows[0]); lines=["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]
    for row in rows: lines.append("| "+" | ".join(f"{row[h]:.4f}" if isinstance(row[h],float) else str(row[h]) for h in headers)+" |")
    args.md_path.parent.mkdir(parents=True,exist_ok=True); args.md_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Saved CSV: {args.csv_path}"); print(f"Saved Markdown: {args.md_path}")
if __name__ == "__main__": main()
