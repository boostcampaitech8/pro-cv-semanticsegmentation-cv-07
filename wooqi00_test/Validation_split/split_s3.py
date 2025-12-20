import os
import argparse
import numpy as np
import pandas as pd


# -------------------------------------------------
# Argument parser
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="S3: S2 split + small/rare structure stress test"
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
        help="Path to S2 val.txt"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory (e.g., splits/s3)"
    )
    parser.add_argument(
        "--low_ratio_percentile",
        type=float,
        default=20.0,
        help="Percentile defining low small-class pixel ratio (default: 20)"
    )
    parser.add_argument(
        "--min_outlier_count",
        type=int,
        default=2,
        help="Minimum small-mask outlier count to consider stress case"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    return parser.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):
    np.random.seed(args.seed)

    df = pd.read_csv(args.metadata)

    # --- load S2 validation ids ---
    with open(args.s2_val) as f:
        s2_val_ids = set(line.strip() for line in f)

    df["is_s2_val"] = df["image_id"].isin(s2_val_ids)

    # -------------------------------------------------
    # 1. Compute small-class pixel ratio
    #    (already stored as total_bone_pixels + small_mask_outlier_count logic)
    # -------------------------------------------------
    if "small_class_pixel_ratio" not in df.columns:
        df["small_class_pixel_ratio"] = (
            df["small_class_pixels"] / df["total_bone_pixels"]
        )

    # low-ratio threshold
    ratio_thr = np.percentile(
        df["small_class_pixel_ratio"],
        args.low_ratio_percentile
    )

    df["is_low_small_ratio"] = df["small_class_pixel_ratio"] <= ratio_thr
    df["is_outlier_heavy"] = df["small_mask_outlier_count"] >= args.min_outlier_count

    df["is_small_stress"] = (
        df["is_low_small_ratio"] | df["is_outlier_heavy"]
    )

    # -------------------------------------------------
    # 2. Desired stress ratio in validation
    # -------------------------------------------------
    overall_stress_ratio = df["is_small_stress"].mean()

    val_df = df[df["is_s2_val"]]
    val_stress_ratio = val_df["is_small_stress"].mean()

    print(f"Overall small-stress ratio: {overall_stress_ratio:.3f}")
    print(f"S2 val small-stress ratio: {val_stress_ratio:.3f}")

    tolerance = 0.03
    new_val_ids = set(s2_val_ids)

    if abs(val_stress_ratio - overall_stress_ratio) < tolerance:
        print("[INFO] Small-structure stress already balanced. Keeping S2 split.")
    else:
        n_val = len(val_df)
        target_stress = int(round(overall_stress_ratio * n_val))

        current_stress_ids = set(
            val_df[val_df["is_small_stress"]]["image_id"]
        )
        current_normal_ids = set(
            val_df[~val_df["is_small_stress"]]["image_id"]
        )

        pool_stress = set(
            df[(~df["is_s2_val"]) & (df["is_small_stress"])]["image_id"]
        )
        pool_normal = set(
            df[(~df["is_s2_val"]) & (~df["is_small_stress"])]["image_id"]
        )

        if len(current_stress_ids) < target_stress:
            to_add = list(pool_stress)[: target_stress - len(current_stress_ids)]
            to_remove = list(current_normal_ids)[: len(to_add)]
        else:
            to_remove = list(current_stress_ids)[: len(current_stress_ids) - target_stress]
            to_add = list(pool_normal)[: len(to_remove)]

        for r, a in zip(to_remove, to_add):
            new_val_ids.remove(r)
            new_val_ids.add(a)

        print(f"[INFO] Swapped {len(to_add)} samples for small-structure stress balance.")

    # -------------------------------------------------
    # 3. Save
    # -------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "val.txt")

    with open(out_path, "w") as f:
        for image_id in sorted(new_val_ids):
            f.write(f"{image_id}\n")

    print(f"[OK] S3 validation split saved to: {out_path}")
    print(f"Total val images: {len(new_val_ids)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)