import os
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from config import get_config

def load_contact_matrix(csv_path: str) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError(f"Missing 'time' column in {csv_path}")
    # Feature columns: all except 'time' and any label columns
    feature_cols = [c for c in df.columns if c not in ("time", "state_label")]
    X = df[feature_cols].to_numpy(dtype=float)
    return X, feature_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PCA on contact matrices across runs")
    parser.add_argument(
        "-t",
        type=str,
        required=True,
        help=(
            "Type identifier (e.g., WT, MUT1). Uses analysis_PCA_new/config.py to resolve base_dir and resi_add_num."
        ),
    )
    parser.add_argument("-start", type=int, default=1)
    parser.add_argument("-end", type=int, default=140)
    parser.add_argument("-n_components", type=int, default=2)
    args = parser.parse_args()

    args.base = os.path.join(get_config(args.t)["base_dir"], "run-all")
    args.out_dir = os.path.join("/M-run-complex-FAST-10/analysis_contact_PCA", args.t)
    print(f"Base path: {args.base}, output directory: {args.out_dir}")
    
    X_list: list[np.ndarray] = []
    feature_cols_ref: list[str] | None = None

    for run_idx in range(args.start, args.end + 1):
        csv_path = os.path.join(
            args.base,
            f"run-{run_idx}",
            "process",
            "full_trajectory_contacts_by_state_name.csv",
        )
        if not os.path.exists(csv_path):
            continue
        X, feature_cols = load_contact_matrix(csv_path)
        if feature_cols_ref is None:
            feature_cols_ref = feature_cols
        else:
            if feature_cols != feature_cols_ref:
                raise ValueError(
                    f"Feature columns mismatch in {csv_path}. Ensure consistent residue columns across runs."
                )
        X_list.append(X)

    if not X_list:
        raise RuntimeError("No contact CSVs found to train PCA")

    X_all = np.vstack(X_list)
    pca = PCA(n_components=args.n_components)
    coords = pca.fit_transform(X_all)

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "contact_pca_model.npz")
    coords_path = os.path.join(args.out_dir, "contact_pca_all_coordinates.csv")
    np.savez(
        model_path,
        components=pca.components_,
        mean=pca.mean_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        n_components=pca.n_components_,
        n_features=pca.n_features_in_,
        feature_cols=np.array(feature_cols_ref, dtype=object),
    )

    pd.DataFrame(coords, columns=["PC1", "PC2"]).to_csv(coords_path, index=False)
    print(f"Saved PCA model to {model_path}")
    print(f"Saved concatenated coordinates to {coords_path}")


if __name__ == "__main__":
    main()


