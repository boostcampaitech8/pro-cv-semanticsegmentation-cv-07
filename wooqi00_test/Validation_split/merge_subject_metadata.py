import argparse
import pandas as pd


# -------------------------------------------------
# Argument parser
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge image-level metadata with subject-level metadata"
    )
    parser.add_argument(
        "--image_metadata",
        type=str,
        required=True,
        help="Path to image-level metadata.csv"
    )
    parser.add_argument(
        "--subject_metadata",
        type=str,
        required=True,
        help="Path to subject metadata file (xlsx or csv)"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output merged metadata csv path"
    )
    return parser.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):
    # --- load image-level metadata ---
    img_df = pd.read_csv(args.image_metadata)

    # --- load subject-level metadata ---
    if args.subject_metadata.endswith(".xlsx"):
        subj_df = pd.read_excel(args.subject_metadata)
    else:
        subj_df = pd.read_csv(args.subject_metadata)

    # -------------------------------------------------
    # 1. Rename columns (Korean -> English)
    # -------------------------------------------------
    rename_map = {
        "ID": "subject_id",
        "나이": "age",
        "성별": "sex",
        "체중(몸무게)": "weight",
        "키(신장)": "height",
    }

    missing_cols = set(rename_map.keys()) - set(subj_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing expected columns in subject metadata: {missing_cols}"
        )

    subj_df = subj_df.rename(columns=rename_map)

    # -------------------------------------------------
    # 2. Normalize subject_id format
    #    001 -> ID001
    # -------------------------------------------------
    subj_df["subject_id"] = (
        subj_df["subject_id"]
        .astype(str)
        .str.zfill(3)
        .apply(lambda x: f"ID{x}")
    )

    # ensure string type for safe merge
    img_df["subject_id"] = img_df["subject_id"].astype(str)

    # -------------------------------------------------
    # 3. Merge
    # -------------------------------------------------
    merged_df = img_df.merge(
        subj_df,
        on="subject_id",
        how="left"
    )

    # -------------------------------------------------
    # 4. Save
    # -------------------------------------------------
    merged_df.to_csv(args.out, index=False)

    print(f"[OK] Merged metadata saved to: {args.out}")
    print(f"Total rows: {len(merged_df)}")

    missing_subject = merged_df["age"].isna().sum()
    if missing_subject > 0:
        print(
            f"[WARN] {missing_subject} rows have missing subject metadata"
        )
    else:
        print("[OK] All rows successfully matched with subject metadata")


if __name__ == "__main__":
    args = parse_args()
    main(args)