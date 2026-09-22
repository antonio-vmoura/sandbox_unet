"""Consolidate single-split Phase 1 / Phase 3 metrics into CSV + JSON for U-Net.

Para uma fase específica ('baseline' para a Fase 1 ou 'optimized' para a Fase 3),
este script percorre o arquivo results.csv gerado pelo Keras, seleciona a melhor época
baseada na menor `val_loss`, e grava um CSV + JSON.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_ORDER = ["unet"]

# Atualizado com as novas métricas customizadas
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

BEST_EPOCH_KEY = "val_loss"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coleta as métricas do Keras.")
    p.add_argument("--phase", choices=["baseline", "optimized"], required=True)
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER)
    p.add_argument("--project", default="/workspace/logs/pipeline_unet_v1")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()

def results_csv_path(project: Path, phase: str, model: str) -> Path:
    if phase == "baseline":
        return project / "phase1_baseline" / f"{model}_baseline" / "results.csv"
    return project / "phase3_optimized" / f"{model}_optimized" / "results.csv"

def get_actual_metric_name(headers: list, target: str) -> str:
    for h in headers:
        if target in h:
            return h
    return target

def parse_best_epoch_metrics(results_csv: Path) -> dict:
    rows = []
    with results_csv.open() as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})
            
    if not rows:
        raise ValueError(f"O results.csv está vazio: {results_csv}")

    actual_best_key = get_actual_metric_name(headers, BEST_EPOCH_KEY)
    
    def get_float_safe(row, key):
        try:
            return float(row.get(key, "inf") or "inf")
        except ValueError:
            return float('inf')

    best_row = min(rows, key=lambda r: get_float_safe(r, actual_best_key))

    out = {}
    for short_name, full_name in METRIC_KEYS.items():
        actual_name = get_actual_metric_name(headers, full_name)
        out[short_name] = float(best_row.get(actual_name, "0") or 0.0)
        
    out["best_epoch"] = float(best_row.get("epoch", "0") or 0.0)
    return out

def _collect_model_row(project: Path, phase: str, model: str) -> dict:
    csv_path = results_csv_path(project, phase, model)
    if not csv_path.exists():
        print(f"  [warn] results.csv não encontrado para {model}: {csv_path}")
        return None
    try:
        metrics = parse_best_epoch_metrics(csv_path)
    except Exception as e:
        print(f"  [error] Falha ao ler {csv_path}: {e}")
        return None
        
    print(
        f"  {model:<8} : "
        f"Melhor Época={metrics['best_epoch']:.0f} | "
        f"Val Loss={metrics['val_loss']:.4f} | "
        f"Val IoU={metrics['val_iou']:.4f} | "
        f"Val Dice={metrics['val_dice']:.4f}"
    )
    return {"model": model, "results_csv": str(csv_path), **metrics}

def _write_artifacts(phase: str, per_model: list, missing: list, out_dir: Path):
    csv_out = out_dir / f"{phase}_metrics.csv"
    json_out = out_dir / f"{phase}_metrics.json"

    if per_model:
        fieldnames = ["model", "best_epoch"] + list(METRIC_KEYS.keys()) + ["results_csv"]
        with csv_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in per_model:
                w.writerow(row)

    with json_out.open("w") as f:
        json.dump({"phase": phase, "models": per_model, "missing": missing}, f, indent=2, sort_keys=True)
    return csv_out, json_out

def main():
    args = parse_args()
    project = Path(args.project).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else project / "pipeline_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_model = []
    missing = []

    print(f"Coletando métricas da fase: {args.phase.upper()}")
    for m in args.models:
        row = _collect_model_row(project, args.phase, m)
        if row is None:
            missing.append(m)
            continue
        per_model.append(row)

    if per_model:
        csv_out, json_out = _write_artifacts(args.phase, per_model, missing, out_dir)
        print("\nArtefatos gerados com sucesso!")
        
    if missing:
        print(f"  [warn] Nenhum resultado para: {missing}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())