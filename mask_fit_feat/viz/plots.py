from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def feature_importance(model, names, path="feature_importance.png") -> None:
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(idx)), importances[idx], tick_label=np.array(names)[idx])
    plt.xticks(rotation=90)
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def pred_vs_true(y_true, y_pred, path="pred_vs_true.png") -> None:
    plt.figure(figsize=(4, 4))
    plt.scatter(y_true, y_pred, s=10, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
