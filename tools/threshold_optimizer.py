import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_score(s: pd.Series) -> pd.Series:
    if s.min() < 0 or s.max() > 1:
        denom = s.max() - s.min()
        if denom == 0:
            return s.fillna(0.0)
        return (s - s.min()) / denom
    return s


def evaluate_thresholds(scores, labels, t1, t2):
    N = len(scores)
    low = scores < t1
    med = (scores >= t1) & (scores < t2)
    high = scores >= t2

    prop_low = low.mean()
    prop_med = med.mean()
    prop_high = high.mean()

    total_crit = labels.sum()
    total_non = (~labels).sum()
    crit_in_high = labels[high].sum()
    non_in_high = (~labels[high]).sum()

    recall_high = crit_in_high / total_crit if total_crit > 0 else 0.0
    fp_rate_high = non_in_high / total_non if total_non > 0 else 0.0

    return {
        "t1": float(t1),
        "t2": float(t2),
        "prop_low": float(prop_low),
        "prop_med": float(prop_med),
        "prop_high": float(prop_high),
        "recall_high": float(recall_high),
        "fp_rate_high": float(fp_rate_high),
        "crit_in_high": int(crit_in_high),
        "non_in_high": int(non_in_high),
    }


def grid_search(scores, labels, p1_grid, p2_grid, prop_target=(0.7, 0.2, 0.1), tol=0.02, fp_max=0.15):
    results = []
    s = np.asarray(scores)
    for p1 in p1_grid:
        for p2 in p2_grid:
            if p1 >= p2:
                continue
            t1 = np.percentile(s, p1)
            t2 = np.percentile(s, p2)
            res = evaluate_thresholds(scores, labels, t1, t2)
            # Verificar proporciones
            prop_ok = (
                abs(res["prop_low"] - prop_target[0]) <= tol
                and abs(res["prop_med"] - prop_target[1]) <= tol
                and abs(res["prop_high"] - prop_target[2]) <= tol
            )
            fp_ok = res["fp_rate_high"] <= fp_max
            res.update({"prop_ok": prop_ok, "fp_ok": fp_ok, "p1": p1, "p2": p2})
            results.append(res)
    return results


def pick_best(results, fp_max=0.15):
    # Preferir factibles (prop_ok & fp_ok); si no, preferir fp_ok con mayor recall
    feasible = [r for r in results if r["prop_ok"] and r["fp_ok"]]
    if feasible:
        best = max(feasible, key=lambda x: x["recall_high"])
        best["reason"] = "feasible_prop_fp"
        return best
    fp_ok = [r for r in results if r["fp_ok"]]
    if fp_ok:
        best = max(fp_ok, key=lambda x: x["recall_high"])
        best["reason"] = "fp_ok_only"
        return best
    # Si no, elegir el mejor recall global, pero advertir
    best = max(results, key=lambda x: x["recall_high"])
    best["reason"] = "best_overall_no_fp"
    return best


def run(input_path, score_col, label_col, out_dir, **kwargs):
    df = pd.read_csv(input_path)
    if score_col not in df.columns:
        raise SystemExit(f"Score column '{score_col}' not found in {input_path}")
    if label_col not in df.columns:
        raise SystemExit(f"Label column '{label_col}' not found in {input_path}")

    scores = df[score_col].astype(float)
    # Normalizar si es necesario
    scores = normalize_score(scores)

    labels = df[label_col].map({True: True, False: False}).fillna(df[label_col]).astype(bool)

    p1_grid = list(range(kwargs.get("p1_min", 50), kwargs.get("p1_max", 81), kwargs.get("p1_step", 1)))
    p2_grid = list(range(kwargs.get("p2_min", 85), kwargs.get("p2_max", 100), kwargs.get("p2_step", 1)))

    results = grid_search(
        scores,
        labels,
        p1_grid,
        p2_grid,
        prop_target=kwargs.get("prop_target", (0.7, 0.2, 0.1)),
        tol=kwargs.get("tol", 0.02),
        fp_max=kwargs.get("fp_max", 0.15),
    )

    best = pick_best(results, fp_max=kwargs.get("fp_max", 0.15))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guardar el grid completo
    pd.DataFrame(results).to_csv(out_dir / "threshold_grid_results.csv", index=False)

    # Guardar el mejor resultado
    with open(out_dir / "threshold_best.json", "w", encoding="utf8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    # Anotar el input con la categoría
    t1 = best["t1"]
    t2 = best["t2"]

    def cat(s):
        if s < t1:
            return "bajo"
        if s < t2:
            return "medio"
        return "alto"

    df["score_norm"] = scores
    df["category"] = df["score_norm"].map(cat)
    df.to_csv(out_dir / "annotated_scores.csv", index=False)

    print("Best thresholds:")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    print(f"Annotated CSV saved to {out_dir / 'annotated_scores.csv'}")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--score-col", default="risk_score")
    parser.add_argument("--label-col", default="is_critical")
    parser.add_argument("--out", default="files/output")
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--fp-max", type=float, default=0.15)
    parser.add_argument("--p1-min", type=int, default=50)
    parser.add_argument("--p1-max", type=int, default=81)
    parser.add_argument("--p1-step", type=int, default=1)
    parser.add_argument("--p2-min", type=int, default=85)
    parser.add_argument("--p2-max", type=int, default=100)
    parser.add_argument("--p2-step", type=int, default=1)
    args = parser.parse_args()

    run(
        args.input,
        args.score_col,
        args.label_col,
        args.out,
        tol=args.tol,
        fp_max=args.fp_max,
        p1_min=args.p1_min,
        p1_max=args.p1_max,
        p1_step=args.p1_step,
        p2_min=args.p2_min,
        p2_max=args.p2_max,
        p2_step=args.p2_step,
    )


if __name__ == "__main__":
    cli()
