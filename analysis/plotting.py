import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from typing import Optional, Literal


DATASETS = {"metr": "METR-LA", "pems": "PEMS-BAY"}

def plot_screeplot(
    vals: np.ndarray, 
    decomp: Literal["svd", "eig"], 
    mat_type: Literal["adj", "lap"] = "lap", 
    dataset: str = None, 
    symmetrized: bool = False,
):
    contribution = np.round(np.cumsum(vals) / vals.sum(), 2)
    quantiles = {p: np.argmax(contribution >= p) for p in {0.5, 0.75, 0.9}}
    ranges = {
        0.5: range(quantiles[0.5] + 1), 
        0.75: range(quantiles[0.5], quantiles[0.75] + 1), 
        0.9: range(quantiles[0.75], quantiles[0.9] + 1)
    }
    alphas = {0.5: 0.5, 0.75: 0.25, 0.9: 0.15}
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    plt.tight_layout()
    ax.plot(range(vals.shape[0]), vals)
    ax.set_xlim(-5, vals.shape[0] + 5)
    ax.set_ylim(-0.15, vals.max() + 0.15)
    ax.set_ylabel("Singular value" if decomp == "svd" else "Eigenvalue")
    ax.set_xlabel("Order")
    for p, range_p in ranges.items():
        ax.fill_between(
            x=ranges[p], 
            y1=0, 
            y2=vals[range_p], 
            alpha=alphas[p], 
            label=f"{int(p * 100): d}% explained"
        )
    ax.legend()
    title = "Scree Plot"
    if decomp == "svd":
        title = "SVD " + title
    if dataset is not None:
        mat_type = "Adjacency Matrix" if mat_type == "adj" else "Laplacian"
        if symmetrized:
            mat_type = "Symmetrized " + mat_type
        title += f" for the {mat_type} of {DATASETS[dataset]}"
    ax.set_title(title)


def plot_loss(losses: dict[str, np.ndarray], title: Optional[str] = None):
    num_epochs = losses["train"].shape[0]
    assert num_epochs == losses["valid"].shape[0]
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    plt.tight_layout()
    ax.plot(range(1, num_epochs + 1), losses["train"], label="Training loss")
    ax.plot(range(1, num_epochs + 1), losses["valid"], label="Validation loss")
    ax.legend(loc=1)
    ax.set_xlim(0.5, num_epochs + 0.5)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    if title is not None:
        ax.set_title(title)