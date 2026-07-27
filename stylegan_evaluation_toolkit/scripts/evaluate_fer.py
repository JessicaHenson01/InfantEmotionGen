#!/usr/bin/env python3
"""Evaluate generated emotion classes with a Hugging Face classifier."""
from __future__ import annotations
import argparse, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import class_name_from_path, image_paths, load_json, normalize_label, save_json, select_torch_device

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--generated",type=Path,required=True); p.add_argument("--model-id",required=True); p.add_argument("--mapping",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",default="auto"); p.add_argument("--batch-size",type=int,default=16); return p.parse_args()

def main():
    args=parse_args(); device_name=select_torch_device(args.device); device=torch.device(device_name); mapping=load_json(args.mapping); paths=image_paths(args.generated)
    processor=AutoImageProcessor.from_pretrained(args.model_id); model=AutoModelForImageClassification.from_pretrained(args.model_id).eval().to(device); id_to_label={int(i):str(v) for i,v in model.config.id2label.items()}
    target_by_folder={}; accepted_to_target={}
    for folder,cfg in mapping.items():
        target=str(cfg["target_name"]); target_by_folder[folder]=target
        for label in cfg["accepted_model_labels"]: accepted_to_target[normalize_label(str(label))]=target
    y_true=[]; y_pred=[]; records=[]; unmapped=Counter()
    for start in tqdm(range(0,len(paths),args.batch_size),desc="FER evaluation"):
        batch_paths=paths[start:start+args.batch_size]; images=[]
        for path in batch_paths:
            with Image.open(path) as image: images.append(image.convert("RGB").copy())
        inputs={k:v.to(device) for k,v in processor(images=images,return_tensors="pt").items()}
        with torch.inference_mode(): probs=model(**inputs).logits.softmax(dim=-1).float().cpu().numpy()
        for row,path in enumerate(batch_paths):
            folder=class_name_from_path(path,args.generated); target=target_by_folder[folder]; pred_id=int(np.argmax(probs[row])); raw=id_to_label[pred_id]; mapped=accepted_to_target.get(normalize_label(raw))
            if mapped is None: mapped=f"UNMAPPED:{raw}"; unmapped[raw]+=1
            y_true.append(target); y_pred.append(mapped); records.append({"image":str(path),"target_class":target,"predicted_model_label":raw,"mapped_prediction":mapped,"correct":target==mapped,"confidence":float(probs[row,pred_id])})
    labels=sorted(set(target_by_folder.values())); report=classification_report(y_true,y_pred,labels=labels,output_dict=True,zero_division=0); matrix=confusion_matrix(y_true,y_pred,labels=labels)
    result={"metric":"FER classifier agreement","model_id":args.model_id,"image_count":len(records),"accuracy":accuracy_score(y_true,y_pred),"macro_f1":report["macro avg"]["f1-score"],"weighted_f1":report["weighted avg"]["f1-score"],"classification_report":report,"confusion_matrix":{"labels":labels,"matrix":matrix.tolist()},"unmapped_predictions":dict(unmapped),"records":records}
    save_json(result,args.output); print(f"FER accuracy: {result['accuracy']:.4f}"); print(f"FER macro F1: {result['macro_f1']:.4f}"); print(f"Saved: {args.output}")
if __name__ == "__main__": main()
