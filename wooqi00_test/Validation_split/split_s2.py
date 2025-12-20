import os
import argparse
import numpy as np
import pandas as pd


# -------------------------------------------------
# Argument parser
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="S2: S1 split + boundary-heavy distribution adjustment"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata.csv"
    )
    parser.add_argument(
        "--s1_val",
        type=str,
        required=True,
        help="Path to S1 val.txt"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory (e.g., splits/s2)"
    )
    parser.add_argument(
        "--boundary_percentile",
        type=float,
        default=75.0,
        help="Percentile to define boundary-heavy images (default: 75)"
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

    # --- load S1 validation ids ---
    with open(args.s1_val) as f:
        s1_val_ids = set(line.strip() for line in f)

    df["is_s1_val"] = df["image_id"].isin(s1_val_ids)

    # --- define boundary-heavy threshold ---
    thr = np.percentile(
        df["boundary_adj_pixels"],
        args.boundary_percentile
    )

    df["is_boundary_heavy"] = df["boundary_adj_pixels"] >= thr

    # --- overall boundary-heavy ratio ---
    overall_ratio = df["is_boundary_heavy"].mean()

    # --- S1 val boundary-heavy ratio ---
    val_df = df[df["is_s1_val"]]
    val_ratio = val_df["is_boundary_heavy"].mean()

    print(f"Overall boundary-heavy ratio: {overall_ratio:.3f}")
    print(f"S1 val boundary-heavy ratio: {val_ratio:.3f}")

    # --- if already close enough, keep S1 ---
    tolerance = 0.02
    if abs(val_ratio - overall_ratio) < tolerance:
        print("[INFO] Boundary-heavy ratio already balanced. Keeping S1 split.")
        new_val_ids = s1_val_ids
    else:
        # need to swap
        n_val = len(val_df)
        target_heavy = int(round(overall_ratio * n_val))

        current_heavy_ids = set(
            val_df[val_df["is_boundary_heavy"]]["image_id"]
        )
        current_light_ids = set(
            val_df[~val_df["is_boundary_heavy"]]["image_id"]
        )

        pool_heavy = set(
            df[(~df["is_s1_val"]) & (df["is_boundary_heavy"])]["image_id"]
        )
        pool_light = set(
            df[(~df["is_s1_val"]) & (~df["is_boundary_heavy"])]["image_id"]
        )

        new_val_ids = set(s1_val_ids)

        if len(current_heavy_ids) > target_heavy:
            # too many heavy → swap out heavy
            to_remove = list(current_heavy_ids)[: len(current_heavy_ids) - target_heavy]
            to_add = list(pool_light)[: len(to_remove)]
        else:
            # too few heavy → swap in heavy
            to_add = list(pool_heavy)[: target_heavy - len(current_heavy_ids)]
            to_remove = list(current_light_ids)[: len(to_add)]

        for r, a in zip(to_remove, to_add):
            new_val_ids.remove(r)
            new_val_ids.add(a)

        print(f"[INFO] Swapped {len(to_add)} samples to balance boundary-heavy ratio.")

    # --- save ---
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "val.txt")

    with open(out_path, "w") as f:
        for image_id in sorted(new_val_ids):
            f.write(f"{image_id}\n")

    print(f"[OK] S2 validation split saved to: {out_path}")
    print(f"Total val images: {len(new_val_ids)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)