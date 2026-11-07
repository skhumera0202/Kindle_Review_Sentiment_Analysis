import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load GaussianNB comparison results
results = pd.read_csv(
    PROJECT_ROOT / "results" / "gaussian_nb_comparison.csv"
)

# Create figure with two confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models = [
    ("Bag of Words (GaussianNB)", "TN", "FP", "FN", "TP"),
    ("TF-IDF (GaussianNB)", "TN", "FP", "FN", "TP")
]

for ax, (_, row) in zip(axes, results.iterrows()):
    cm = [
        [row["TN"], row["FP"]],
        [row["FN"], row["TP"]]
    ]

    image = ax.imshow(cm, cmap="Blues")

    ax.set_title(row["Model"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Negative", "Positive"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, int(cm[i][j]),
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)

plt.tight_layout()

# Save visualization
output_path = (
    PROJECT_ROOT / "results" / "gaussian_nb_confusion_matrices.png"
)

plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("Confusion matrix comparison saved successfully!")
print(f"Saved at: {output_path}")

plt.show()