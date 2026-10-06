import json
import csv
import numpy as np
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

with open('results/stage4_longrun_fixed/summary.json') as f:
    d = json.load(f)

q = d['sample_rate']
steps = d['total_dp_steps']
sigma = d['sigma']

alphas = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
rdp = compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=alphas)
eps, best_alpha = get_privacy_spent(orders=alphas, rdp=rdp, delta=1e-5)

print(f"Computed eps: {eps:.4f}, best_alpha: {best_alpha}")
print(f"Artifact eps: {d['epsilon']:.4f}, best_alpha: {d['best_alpha']}")

with open('results/stage4_longrun_fixed/rounds.csv') as f:
    r = list(csv.DictReader(f))
print(f"Rounds in CSV: {len(r)}")
for row in r:
    print(f"Round {row['round']}: Acc {row['test_accuracy']}")
