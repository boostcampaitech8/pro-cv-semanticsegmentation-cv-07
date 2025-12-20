import os
import argparse
from glob import glob
import pandas as pd


# -------------------------------------------------
# Argument parser
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build base flags.csv for manual inspection"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Path to data/train/DCM"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output flags csv path"
    )
    return parser.parse_args()

# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):
    rows = []

    subject_ids = sorted(os.listdir(args.image_root))
    for sid in subject_ids:
        img_dir = os.path.join(args.image_root, sid)
        if not os.path.isdir(img_dir):
            continue

        img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
        for img_path in img_paths:
            image_id = os.path.basename(img_path)

            rows.append({
                # --- identifiers ---
                "subject_id": sid,
                "image_id": image_id,

                # --- optional stored attribute (NOT used by default) --- L/R
                "hand_side": "",

                # --- core manual flags (1 / 0 / blank) ---
                "has_wrist_deviation": "",
                "has_metal_artifact": "",
                "has_implant": "",
                "has_nail_art": "",

                # --- optional free-text note ---
                "note": ""
            })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"[OK] Base flags.csv created at: {args.out}")
    print("hand_side is stored for reference only (not used in split/train).")
    print("Fill 1 / 0 for flags, leave blank if uncertain.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
