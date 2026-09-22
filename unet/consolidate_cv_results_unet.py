"""Consolidate Phase 4 (Cross-Validation) results into CSV+JSON.

Este script lê os N ficheiros results.csv gerados pelos Folds da Fase 4,
extrai as métricas baseadas no melhor 'val_custom_iou' e calcula a média
e desvio padrão para construir a tabela final do artigo.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

METRIC_KEYS = {
    "train_loss": "loss",
    "train_acc": "accuracy",
    "train_iou": "custom_iou", 
    "train_dice": "custom_dice",
    "val_loss": "val_loss",
    "val_acc": "val_accuracy",
    "val_iou": "val_custom_iou",
    "val_dice": "val_custom_dice"
}
BEST_EPOCH_KEY = "val_custom_iou"

def parse_args():
    p = argparse.ArgumentParser(description="Consolida resultados CV da U-Net.")
    p.add_argument("--project", default="/workspace/logs/pipeline_unet_v1")
    p.add_argument("--cv_version", default="cv_v1")
    p.add_argument("--k_folds", type=int, default=5)
    return p.parse_args()

def get_actual_metric_name(headers: list, target: str) -> str:
    for h in headers:
        if target in h: return h
    return target

def parse_best_epoch(results_csv: Path) -> dict:
    rows = []
    with results_csv.open() as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})
            
    actual_best_key = get_actual_metric_name(headers, BEST_EPOCH_KEY)
    
    def get_float_safe(row, key):
        try: return float(row.get(key, "-inf") or "-inf")
        except ValueError: return float('-inf')

    # Na U-Net (Fase 3/4) maximizamos o IoU, pelo que procuramos o 'max'
    best_row = max(rows, key=lambda r: get_float_safe(r, actual_best_key))

    out = {}
    for short_name, full_name in METRIC_KEYS.items():
        actual_name = get_actual_metric_name(headers, full_name)
        out[short_name] = float(best_row.get(actual_name, "0") or 0.0)
    out["best_epoch"] = float(best_row.get("epoch", "0") or 0.0)
    return out

def main():
    args = parse_args()
    project = Path(args.project).resolve()
    cv_root = project / "cv" / args.cv_version / "unet_cv_isic_2018"
    out_dir = project / "pipeline_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold_metrics = []
    
    for k in range(args.k_folds):
        csv_path = cv_root / f"fold_{k}" / "results.csv"
        if not csv_path.exists():
            print(f"[ERRO] CSV do fold {k} não encontrado: {csv_path}")
            return 1
        metrics = parse_best_epoch(csv_path)
        metrics["fold"] = k
        per_fold_metrics.append(metrics)

    # Cálculo da Média e Desvio Padrão
    summary = {}
    metric_names = list(METRIC_KEYS.keys()) + ["best_epoch"]
    
    for key in metric_names:
        values = [m[key] for m in per_fold_metrics]
        summary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0
        }

    # Gerar JSON
    payload = {
        "model": "unet",
        "n_folds": args.k_folds,
        "per_fold": per_fold_metrics,
        "summary": summary
    }
    
    with open(out_dir / "cv_consolidated.json", "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        
    # Gerar CSV
    csv_headers = ["model", "n_folds"]
    for k in metric_names:
        csv_headers.extend([f"{k}_mean", f"{k}_std"])
        
    with open(out_dir / "cv_consolidated.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_headers)
        w.writeheader()
        row = {"model": "unet", "n_folds": args.k_folds}
        for k in metric_names:
            row[f"{k}_mean"] = summary[k]["mean"]
            row[f"{k}_std"] = summary[k]["std"]
        w.writerow(row)

    print(f"\n[SUCESSO] Artefatos CV Consolidados salvos em: {out_dir}")
    print(f"  Val IoU (Jaccard): {summary['val_iou']['mean']:.4f} ± {summary['val_iou']['std']:.4f}")
    print(f"  Val Dice (DSC): {summary['val_dice']['mean']:.4f} ± {summary['val_dice']['std']:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())