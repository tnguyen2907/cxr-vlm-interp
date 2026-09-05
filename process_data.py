"""Create the same CheXpert+ study manifest as the original notebook."""

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from experiment_utils import RANDOM_STATE, TARGET_LABELS, resolve_path, run_log

CXR_DIR = Path("/opt/gpudata/cxr")
TRAIN_STUDIES_N = 20_000
TEST_STUDIES_N = 5_000


def prepare_manifest(cxr_dir=CXR_DIR):
    chexpert_dir = cxr_dir / "chexpertplus"
    split_df = pd.read_csv(chexpert_dir / "split.csv")
    meta_df = pd.read_csv(chexpert_dir / "metadata.csv")
    label_df = pd.read_csv(cxr_dir / "derived/chexpertplus-findings-labels-chexbert.csv")
    for name, frame in [("split", split_df), ("metadata", meta_df), ("labels", label_df)]:
        print(name, frame.shape)
        print(frame.head().to_string(index=False))

    rows = []
    for path in tqdm(sorted((chexpert_dir / "PNG").rglob("*.png")), desc="Scan images"):
        patient_id, study, dicom = path.with_suffix("").parts[-3:]
        rows.append({"subject_id": patient_id, "study_id": f"{patient_id}_{study}",
                     "dicom_id": f"{patient_id}_{study}_{dicom}", "image_path": str(path)})
    image_df = pd.DataFrame(rows)
    # patient32368 has an image-loading problem.
    image_df = image_df[image_df["subject_id"] != "patient32368"].reset_index(drop=True)
    meta_small = meta_df[["subject_id", "study_id", "dicom_id", "ViewPosition"]].copy()
    meta_small["view"] = meta_small["ViewPosition"].fillna("").str.upper()
    data = (
        split_df[["study_id", "split"]].drop_duplicates("study_id")
        .merge(meta_small, on="study_id", how="inner")
        .merge(image_df, on=["subject_id", "study_id", "dicom_id"], how="left")
        .merge(label_df[["study_id"] + TARGET_LABELS], on="study_id", how="inner")
    )
    print("Merged rows/studies/patients", len(data), data.study_id.nunique(), data.subject_id.nunique())
    print(data["view"].value_counts(dropna=False).to_string())
    frontal = data[data["view"].isin(["PA", "AP"]) & data["image_path"].notna()].copy()
    frontal["view_rank"] = frontal["view"].map({"PA": 0, "AP": 1})
    manifest = (
        frontal.sort_values(["study_id", "view_rank", "dicom_id"])
        .drop_duplicates("study_id", keep="first")
        .sort_values(["split", "subject_id", "study_id"]).reset_index(drop=True)
    )
    patient_counts = (
        manifest.groupby("subject_id").size().rename("n_rows").reset_index()
        .sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    )
    train_patients, test_patients = [], []
    train_rows = test_rows = 0
    for row in patient_counts.itertuples(index=False):
        if train_rows + row.n_rows <= TRAIN_STUDIES_N:
            train_patients.append(row.subject_id)
            train_rows += row.n_rows
        elif test_rows + row.n_rows <= TEST_STUDIES_N:
            test_patients.append(row.subject_id)
            test_rows += row.n_rows
        if train_rows == TRAIN_STUDIES_N and test_rows == TEST_STUDIES_N:
            break
    manifest["probe_split"] = "unused"
    manifest.loc[manifest.subject_id.isin(train_patients), "probe_split"] = "train"
    manifest.loc[manifest.subject_id.isin(test_patients), "probe_split"] = "test"
    result = manifest[manifest.probe_split.isin(["train", "test"])].copy()
    columns = ["subject_id", "study_id", "dicom_id", "split", "probe_split", "ViewPosition", "view", "image_path"]
    return result[columns + TARGET_LABELS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=resolve_path, required=True, help="Fresh runs/<run_name> folder")
    args = parser.parse_args()
    with run_log(args.output_root):
        run(args)


def run(args):
    output = args.output_root / "processed_data/chexpertplus_frontal_5labels.csv"
    df = prepare_manifest()
    assert df.study_id.is_unique
    assert df.groupby("subject_id").probe_split.nunique().max() == 1
    assert df.probe_split.value_counts().to_dict() == {"train": TRAIN_STUDIES_N, "test": TEST_STUDIES_N}
    print(df.groupby("probe_split").agg(studies=("study_id", "size"), patients=("subject_id", "nunique")))
    print(df["view"].value_counts())
    print(df[TARGET_LABELS].eq(1).groupby(df.probe_split).mean())
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print("Saved", output)


if __name__ == "__main__":
    main()
