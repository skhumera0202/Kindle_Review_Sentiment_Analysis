
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load model comparison results
results_df = pd.read_csv(
    PROJECT_ROOT / "results" / "model_comparison.csv"
)

# Select evaluation metrics
metrics = [
    "Accuracy",
    "F1 Score",
    "Negative Recall"
]

# Create comparison chart
ax = results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 7),
    width=0.75
)

# Chart formatting
plt.title(
    "Kindle Review Sentiment - Model Comparison",
    fontsize=14
)

plt.xlabel("Machine Learning Models")
plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.legend(title="Evaluation Metrics")

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

# Display values on bars
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        fontsize=8
    )

plt.tight_layout()

# Create results folder if needed
results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)

# Save chart
chart_path = results_dir / "model_comparison.png"

plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight"
)

print("Model comparison chart saved successfully!")
print("Saved at:", chart_path)

plt.show()
