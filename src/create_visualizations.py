
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

sns.set_theme(style="whitegrid")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load prepared training data
    X = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y = pd.read_csv(PROCESSED_DIR / "y_train.csv")["label"]

    # Add labels for charting only
    df = X.copy()
    df["traffic_type"] = y.map({0: "Normal", 1: "Attack"})

    # 1. Traffic class distribution
    plt.figure(figsize=(7, 5))
    counts = df["traffic_type"].value_counts()
    sns.barplot(x=counts.index, y=counts.values)

    plt.title("Normal vs Attack Traffic")
    plt.xlabel("Traffic Type")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "traffic_class_distribution.png", dpi=300)
    plt.close()

    # 2. Top 10 protocols
    plt.figure(figsize=(9, 6))
    protocol_counts = df["proto"].value_counts().head(10)
    sns.barplot(x=protocol_counts.values, y=protocol_counts.index)

    plt.title("Top 10 Network Protocols")
    plt.xlabel("Number of Records")
    plt.ylabel("Protocol")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "protocol_distribution.png", dpi=300)
    plt.close()

    # 3. Top 10 services
    plt.figure(figsize=(9, 6))
    service_counts = df["service"].value_counts().head(10)
    sns.barplot(x=service_counts.values, y=service_counts.index)

    plt.title("Top 10 Network Services")
    plt.xlabel("Number of Records")
    plt.ylabel("Service")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "service_distribution.png", dpi=300)
    plt.close()

    # 4. Compare selected numerical features
    selected_features = ["rate", "sbytes", "dbytes", "spkts"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for feature, ax in zip(selected_features, axes.flatten()):
        sns.boxplot(
            data=df,
            x="traffic_type",
            y=feature,
            ax=ax
        )
        ax.set_title(f"{feature}: Normal vs Attack")
        ax.set_xlabel("Traffic Type")
        ax.set_ylabel(feature)

    fig.suptitle("Network Feature Distribution by Traffic Type")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "feature_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("Visualization creation completed.")
    print("Charts saved to:", FIGURES_DIR)

    print("\nCreated files:")
    for file_path in sorted(FIGURES_DIR.glob("*.png")):
        print(file_path.name)


if __name__ == "__main__":
    main()
