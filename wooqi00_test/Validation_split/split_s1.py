import os
import argparse
import numpy as np
import pandas as pd


# -------------------------------------------------
# Argument parser
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="S1: Difficulty-stratified validation split"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata.csv"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory (e.g., splits/s1)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation ratio per difficulty bin (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    return parser.parse_args()


# -------------------------------------------------
# Utility
# -------------------------------------------------
def rank_normalize(x: pd.Series) -> pd.Series:
    """Rank-based normalization to [0, 1]."""
    return x.rank(method="average") / len(x)


def compute_s1_difficulty(df: pd.DataFrame) -> pd.Series:
    """
    S1 difficulty score:
    rank-normalized sum of
    - overlap_pixels
    - boundary_adj_pixels
    - small_mask_outlier_count
    """
    return (
        rank_normalize(df["overlap_pixels"]) +
        rank_normalize(df["boundary_adj_pixels"]) +
        rank_normalize(df["small_mask_outlier_count"])
    )


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):
    np.random.seed(args.seed)

    df = pd.read_csv(args.metadata)

    # --- compute difficulty score ---
    df["difficulty_score"] = compute_s1_difficulty(df)

    # --- bin into easy / mid / hard ---
    df["difficulty_bin"] = pd.qcut(
        df["difficulty_score"],
        q=3,
        labels=["easy", "mid", "hard"]
    )

    val_ids = []

    for bin_name in ["easy", "mid", "hard"]:
        bin_df = df[df["difficulty_bin"] == bin_name]
        n_val = int(len(bin_df) * args.val_ratio)

        sampled = bin_df.sample(
            n=n_val,
            random_state=args.seed
        )
        val_ids.extend(sampled["image_id"].tolist())
    
    # --- derive train ids ---
    all_ids = set(df["image_id"].tolist())
    val_ids = set(val_ids)
    train_ids = sorted(all_ids - val_ids)


    # --- save ---
    os.makedirs(args.out_dir, exist_ok=True)
    val_path = os.path.join(args.out_dir, "val.txt")
    train_path = os.path.join(args.out_dir, "train.txt")

    with open(val_path, "w") as f:
        for image_id in sorted(val_ids):
            f.write(f"{image_id}\n")

    with open(train_path, "w") as f:
        for image_id in train_ids:
            f.write(f"{image_id}\n")


    print(f"[OK] S1 split saved to: {args.out_dir}")
    print(f"Train images: {len(train_ids)} | Val images: {len(val_ids)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)