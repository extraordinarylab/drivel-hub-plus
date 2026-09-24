"""Does the self-reuse ordering survive the choice of n? If not, n=3 is cherry-picked."""
import numpy as np
from scipy.stats import spearmanr

from scripts.plot_storyscope import build, recycled_share

labels, _, raw = build()
table = {n: {l: float(np.median(recycled_share(raw[l], n=n))) for l in labels}
         for n in (2, 3, 4, 5)}

print(f"{'source':24}" + "".join(f"{'n=' + str(n):>9}" for n in table))
for l in sorted(labels, key=lambda l: table[3][l]):
    print(f"  {l:22}" + "".join(f"{table[n][l]:8.1f}%" for n in table))

print("\nrank correlation of the source ordering, against n=3:")
base = [table[3][l] for l in labels]
for n in table:
    print(f"  n={n}: rho = {spearmanr([table[n][l] for l in labels], base).statistic:.3f}")

print("\nare both humans below every model?")
for n in table:
    hum = max(table[n][l] for l in labels if l.startswith("Human"))
    mod = min(table[n][l] for l in labels if not l.startswith("Human"))
    print(f"  n={n}: max human {hum:5.1f}%  min model {mod:5.1f}%  "
          f"{'yes' if hum < mod else 'NO'}")
