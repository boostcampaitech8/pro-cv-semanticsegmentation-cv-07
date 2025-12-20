import os
import argparse
import pandas as pd


# -------------------------------------------------
# Argument parser
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="S4: Stress-boosted validation split (based on S2 val)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata.csv"
    )
    parser.add_argument(
        "--s2_val",
        type=str,
        required=True,
        help="Path to S2 val.txt (baseline val set to start from)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for S4 split"
    )
    parser.add_argument(
        "--target_stress_ratio",
        type=float,
        default=0.5,
        help="Target stress ratio in validation set (e.g., 0.5)"
    )
    parser.add_argument(
        "--outlier_thr",
        type=int,
        default=2,
        help="Stress if small_mask_outlier_count >= this value"
    )
    parser.add_argument(
        "--ratio_quantile",
        type=float,
        default=0.2,
        help="Quantile threshold for small_class_pixel_ratio (lower is more stress)"
    )
    parser.add_argument(
        "--min_pixel_quantile",
        type=float,
        default=0.2,
        help="Quantile threshold for min_small_class_pixels (lower is more stress)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling during swaps"
    )
    return parser.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):
    df = pd.read_csv(args.metadata)

    # required columns check (fail fast)
    required = [
        "image_id",
        "small_mask_outlier_count",
        "small_class_pixel_ratio",
        "min_small_class_pixels",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in metadata: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # -------------------------------------------------
    # Load S2 validation ids
    # -------------------------------------------------
    with open(args.s2_val, "r") as f:
        s2_val_ids = set(line.strip() for line in f if line.strip())

    df["is_val"] = df["image_id"].isin(s2_val_ids)

    # -------------------------------------------------
    # Define stress condition (S3-A + S3-B + S3-C)
    # -------------------------------------------------
    ratio_thr = df["small_class_pixel_ratio"].quantile(args.ratio_quantile)
    minpix_thr = df["min_small_class_pixels"].quantile(args.min_pixel_quantile)

    df["is_stress"] = (
        (df["small_class_pixel_ratio"] < ratio_thr)
        | (df["min_small_class_pixels"] < minpix_thr)
        | (df["small_mask_outlier_count"] >= args.outlier_thr)
    )

    val_df = df[df["is_val"]]
    train_df = df[~df["is_val"]]

    current_ratio = float(val_df["is_stress"].mean())
    target_ratio = float(args.target_stress_ratio)

    print(f"[INFO] Current S2 val stress ratio: {current_ratio:.3f}")
    print(f"[INFO] Target S4 val stress ratio: {target_ratio:.3f}")
    print(f"[INFO] S2 val size: {len(val_df)} / train size: {len(train_df)}")
    print(f"[INFO] Stress thresholds: ratio<{ratio_thr:.6f}, minpix<{minpix_thr:.1f}, outlier>={args.outlier_thr}")

    # -------------------------------------------------
    # Stress-boost swap (keep val size constant)
    # -------------------------------------------------
    new_val_ids = set(s2_val_ids)

    if current_ratio < target_ratio:
        # how many stress samples we need to add into validation
        current_stress = int(val_df["is_stress"].sum())
        target_stress = int(round(target_ratio * len(val_df)))
        need = max(0, target_stress - current_stress)

        stress_train = train_df[train_df["is_stress"]]
        nonstress_val = val_df[~val_df["is_stress"]]

        swap_n = min(len(stress_train), len(nonstress_val), need)

        print(f"[INFO] Need +{need} stress in val -> swapping {swap_n} samples")

        if swap_n > 0:
            swap_in = stress_train.sample(swap_n, random_state=args.seed)
            swap_out = nonstress_val.sample(swap_n, random_state=args.seed)

            new_val_ids |= set(swap_in["image_id"])
            new_val_ids -= set(swap_out["image_id"])
        else:
            print("[WARN] Not enough candidates to swap. Keeping S2 split as-is.")
    else:
        print("[INFO] Validation already meets/exceeds target stress ratio. Keeping S2 split.")

    # -------------------------------------------------
    # Save split
    # -------------------------------------------------
    all_ids = set(df["image_id"].astype(str).tolist())
    val_ids = sorted(new_val_ids)
    train_ids = sorted(all_ids - set(new_val_ids))

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_ids))

    with open(os.path.join(args.out_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_ids))

    # -------------------------------------------------
    # Final report
    # -------------------------------------------------
    final_val_df = df[df["image_id"].isin(new_val_ids)]
    final_ratio = float(final_val_df["is_stress"].mean())

    print(f"[OK] Saved S4 split to: {args.out_dir}")
    print(f"[OK] Train size: {len(train_ids)} / Val size: {len(val_ids)}")
    print(f"[OK] Final S4 val stress ratio: {final_ratio:.3f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)