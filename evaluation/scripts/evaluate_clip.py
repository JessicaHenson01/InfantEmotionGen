#!/usr/bin/env python3
"""Evaluate generated classes with OpenCLIP image-text similarity."""
from __future__ import annotations
import argparse, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, open_clip, torch
from PIL import Image
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import class_name_from_path, image_paths, load_json, save_json, select_torch_device

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--model", default="ViT-B-32")
    p.add_argument("--pretrained", default="laion2b_s34b_b79k")
    return p.parse_args()

def main():
    args = parse_args(); device = select_torch_device(args.device)
    config = load_json(args.prompts); paths = image_paths(args.generated); class_names = sorted(config)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.model); model.eval()
    flat, owners = [], []
    for i, name in enumerate(class_names):
        for prompt in config[name]: flat.append(prompt); owners.append(i)
    with torch.inference_mode():
        pf = model.encode_text(tokenizer(flat).to(device)); pf = pf / pf.norm(dim=-1, keepdim=True)
        class_features=[]
        for i in range(len(class_names)):
            feature=pf[[j for j,o in enumerate(owners) if o==i]].mean(dim=0); class_features.append(feature/feature.norm())
        tf = torch.stack(class_features)
    records=[]; target_scores=defaultdict(list); margins=defaultdict(list); confusion=Counter()
    for start in tqdm(range(0,len(paths),args.batch_size), desc="CLIP evaluation"):
        batch_paths=paths[start:start+args.batch_size]; tensors=[]
        for path in batch_paths:
            with Image.open(path) as image: tensors.append(preprocess(image.convert("RGB")))
        with torch.inference_mode():
            feats=model.encode_image(torch.stack(tensors).to(device)); feats=feats/feats.norm(dim=-1,keepdim=True)
            cosine=feats@tf.T; probs=(100*cosine).softmax(dim=-1)
        cosine, probs = cosine.float().cpu().numpy(), probs.float().cpu().numpy()
        for row,path in enumerate(batch_paths):
            target=class_name_from_path(path,args.generated); ti=class_names.index(target); pi=int(np.argmax(probs[row])); pred=class_names[pi]
            score=float(cosine[row,ti]*100); ordered=np.sort(cosine[row]); margin=float((ordered[-1]-ordered[-2])*100)
            confusion[(target,pred)]+=1; target_scores[target].append(score); margins[target].append(margin)
            records.append({"image":str(path),"target_class":target,"predicted_class":pred,"correct":target==pred,"target_cosine_x100":score,"top1_margin_x100":margin})
    result={"metric":"OpenCLIP emotion alignment","model":args.model,"pretrained":args.pretrained,"image_count":len(records),"zero_shot_accuracy":sum(r["correct"] for r in records)/len(records),"mean_target_cosine_x100":float(np.mean([r["target_cosine_x100"] for r in records])),"per_class":{},"confusion":{f"{a}->{b}":n for (a,b),n in sorted(confusion.items())},"records":records}
    for name in class_names:
        subset=[r for r in records if r["target_class"]==name]
        if subset: result["per_class"][name]={"count":len(subset),"zero_shot_accuracy":sum(r["correct"] for r in subset)/len(subset),"mean_target_cosine_x100":float(np.mean(target_scores[name])),"mean_top1_margin_x100":float(np.mean(margins[name]))}
    save_json(result,args.output); print(f"CLIP agreement: {result['zero_shot_accuracy']:.4f}"); print(f"Saved: {args.output}")
if __name__ == "__main__": main()
