import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path("files/output")
OUT.mkdir(parents=True, exist_ok=True)

# Load artifacts
grid_path = Path("files/output/threshold_grid_results.csv")
best_path = Path("files/output/threshold_best.json")
annotated_path = Path("files/output/annotated_scores.csv")

if not grid_path.exists() or not best_path.exists() or not annotated_path.exists():
    raise SystemExit("Required files not found in files/output. Run threshold optimizer first.")

grid = pd.read_csv(grid_path)
with open(best_path, "r", encoding="utf8") as f:
    best = json.load(f)
ann = pd.read_csv(annotated_path)

# Trade-off scatter
plt.figure(figsize=(8, 6))
sc = plt.scatter(grid['fp_rate_high'], grid['recall_high'],
                 c=grid['prop_ok'].astype(int), cmap='viridis', alpha=0.8)
plt.colorbar(sc, label='prop_ok (1=ok,0=not)')
plt.xlabel('FP rate (alto)')
plt.ylabel('Recall (alto)')
plt.title('Trade-off: Recall vs FP rate (grid)')
# annotate best
plt.scatter([best['fp_rate_high']], [best['recall_high']], c='red', s=80, label='best')
plt.legend()
tradeoff_file = OUT / 'tradeoff_scatter.png'
plt.tight_layout()
plt.savefig(tradeoff_file)
plt.close()

# Proportions bar for best
props = [best['prop_low'], best['prop_med'], best['prop_high']]
labels = ['bajo', 'medio', 'alto']
plt.figure(figsize=(6, 4))
plt.bar(labels, props, color=['#4CAF50', '#FFC107', '#F44336'])
plt.ylim(0, 1)
plt.title('Proporciones bajo/medio/alto (umbral seleccionado)')
prop_file = OUT / 'proportions_best.png'
plt.tight_layout()
plt.savefig(prop_file)
plt.close()

# Bootstrap for chosen thresholds
t1 = best['t1']
t2 = best['t2']
labels = ann['is_critical'].astype(bool)
scores = ann['score_norm'].astype(float)

n_boot = 500
rng = np.random.default_rng(0)
recalls = []
fprs = []
N = len(ann)
for _ in range(n_boot):
    idx = rng.integers(0, N, N)
    s = scores.values[idx]
    y = labels.values[idx]
    high = s >= t2
    crit_in_high = int(((y) & (high)).sum())
    non_in_high = int((~y & high).sum())
    total_crit = int(y.sum())
    total_non = int((~y).sum())
    recall_high = crit_in_high / total_crit if total_crit > 0 else 0.0
    fp_rate_high = non_in_high / total_non if total_non > 0 else 0.0
    recalls.append(recall_high)
    fprs.append(fp_rate_high)

recalls = np.array(recalls)
fprs = np.array(fprs)

ci_rec = np.percentile(recalls, [2.5, 97.5])
ci_fp = np.percentile(fprs, [2.5, 97.5])

# Save histograms
plt.figure(figsize=(8, 4))
plt.hist(recalls, bins=30, color='C0', alpha=0.8)
plt.axvline(best['recall_high'], color='red', linestyle='--', label='best')
plt.title('Bootstrap recall_high')
plt.xlabel('Recall (alto)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(OUT / 'bootstrap_recall.png')
plt.close()

plt.figure(figsize=(8, 4))
plt.hist(fprs, bins=30, color='C1', alpha=0.8)
plt.axvline(best['fp_rate_high'], color='red', linestyle='--', label='best')
plt.title('Bootstrap fp_rate_high')
plt.xlabel('FP rate (alto)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(OUT / 'bootstrap_fp.png')
plt.close()

# Write simple report
report = {
    'best': best,
    'bootstrap': {
        'recall_mean': float(recalls.mean()),
        'recall_ci_2.5': float(ci_rec[0]),
        'recall_ci_97.5': float(ci_rec[1]),
        'fp_mean': float(fprs.mean()),
        'fp_ci_2.5': float(ci_fp[0]),
        'fp_ci_97.5': float(ci_fp[1]),
    }
}
with open(OUT / 'threshold_report.json', 'w', encoding='utf8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print('Saved:', tradeoff_file, prop_file, 'bootstrap images, and threshold_report.json')
print(json.dumps(report, indent=2, ensure_ascii=False))
