"""Synthetic checks for the torch probe: content signal within turn is recovered when present,
and within-turn AUC ~0.5 when the label depends only on the turn."""
import numpy as np
from prism_content_probe import LogisticRegression, PCA, within_turn, auc

rng = np.random.default_rng(0)
n, d = 3000, 40
turn = rng.integers(1, 8, n)
H = rng.normal(size=(n, d))
oh = np.eye(13)[turn]
for signal in (True, False):
    logit = 0.6 * (turn - 4) + (2.0 * H[:, 0] if signal else 0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    tr, va = slice(0, 2000), slice(2000, None)
    pca = PCA(d).fit(H[tr])   # keep all dims: isotropic synthetic data has no preferred PCs
    Z = pca.transform(H)
    clf = LogisticRegression(C=1.0).fit(np.hstack([oh[tr], Z[tr]]), y[tr])
    content = Z[va] @ clf.coef_[0][13:]
    wt, pairs = within_turn(content, y[va], turn[va])
    print("signal" if signal else "no-signal", round(wt, 3), pairs)
    assert (wt > 0.75) if signal else (0.4 < wt < 0.6)
print("probe ok")
