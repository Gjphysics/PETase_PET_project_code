import os
import argparse
from typing import List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import importlib.util


STATE_MAP = {
    "Apo": 1,
    "ES": [2, 3],
    "DS": 4,
    "PS": 5,
}


def load_and_combine_trajectory_contacts(run_process_dir: str, state_names_to_load: List[str], resi_add_num: int) -> Optional[pd.DataFrame]:
    all_dfs_for_concat: List[pd.DataFrame] = []

    for state_name in state_names_to_load:
        file_path = os.path.join(run_process_dir, f"single_residue_contact_{state_name}.csv")
        if not os.path.exists(file_path):
            continue

        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()

            if not first_line or not first_line.startswith('#'):
                # Skip files without the expected header format
                continue

            headers = [h.strip() for h in first_line[1:].split(',')]
            df = pd.read_csv(file_path, names=headers, skiprows=1, comment='#')

            if df.empty:
                continue

            time_col_name = None
            renamed_cols = {}
            for col in df.columns:
                if isinstance(col, str):
                    if col.startswith('resid_'):
                        try:
                            res_num = int(col.split('_')[1])
                            renamed_cols[col] = res_num + resi_add_num
                        except ValueError:
                            renamed_cols[col] = col
                    elif 'time' in col.lower() or 'frame' in col.lower():
                        time_col_name = col
                        renamed_cols[col] = col
                    else:
                        renamed_cols[col] = col
                else:
                    renamed_cols[col] = col

            df = df.rename(columns=renamed_cols)

            if time_col_name is None:
                continue

            df[time_col_name] = pd.to_numeric(df[time_col_name], errors='coerce')
            df = df.dropna(subset=[time_col_name])
            df = df.set_index(time_col_name)

            df['state_label'] = state_name
            df = df.copy()
            all_dfs_for_concat.append(df)

        except pd.errors.EmptyDataError:
            continue
        except Exception:
            continue

    if not all_dfs_for_concat:
        return None

    try:
        combined_df = pd.concat(all_dfs_for_concat)
    except Exception:
        # Fallback: align on common numeric residue columns and 'state_label'
        common_cols_to_try = set()
        first_df_cols = None
        for df_item in all_dfs_for_concat:
            current_cols = set([c for c in df_item.columns if isinstance(c, (int, np.integer)) or c == 'state_label'])
            if first_df_cols is None:
                first_df_cols = current_cols
                common_cols_to_try.update(current_cols)
            else:
                common_cols_to_try.intersection_update(current_cols)

        if not common_cols_to_try or 'state_label' not in common_cols_to_try:
            return None

        aligned_dfs = []
        for df_item in all_dfs_for_concat:
            df_reindexed = df_item.reindex(columns=list(common_cols_to_try))
            aligned_dfs.append(df_reindexed)
        try:
            combined_df = pd.concat(aligned_dfs)
        except Exception:
            return None

    combined_df = combined_df.sort_index()
    return combined_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate missing combined contact CSVs (paths derived via -t). "
            "Merges single_residue_contact_{state}.csv for runs that have a process folder "
            "but lack full_trajectory_contacts_by_state_name.csv."
        )
    )
    parser.add_argument(
        "-t",
        type=str,
        required=True,
        help=(
            "Type identifier (e.g., WT, MUT1). Uses analysis_PCA_new/config.py to resolve base_dir and resi_add_num."
        ),
    )
    parser.add_argument("-start", type=int, default=1, help="Start run index (inclusive)")
    parser.add_argument("-end", type=int, default=140, help="End run index (inclusive)")
    parser.add_argument(
        "-states",
        type=str,
        default="Apo,ES,DS,PS",
        help="Comma-separated state names matching single_residue_contact_{state}.csv",
    )
    args = parser.parse_args()

    # Resolve base path and residue offset from config via type
    type_id = args.t
    config_path = "/M-run-complex-FAST-10/analysis_contact_PCA/config.py"
    spec = importlib.util.spec_from_file_location("analysis_pca_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    type_config = config_module.get_config(type_id)
    args.base = os.path.join(type_config["base_dir"], "run-all")
    args.resi_add_num = int(type_config["resi_add_num"]) 
    print(f"Base path: {args.base}, residue add num: {args.resi_add_num}")

    state_names = [s.strip() for s in args.states.split(',') if s.strip()]
    processed = 0
    skipped_exists = 0
    skipped_noprocess = 0
    failed = 0

    for run_idx in tqdm(range(args.start, args.end + 1), desc="Scanning runs"):
        run_dir = os.path.join(args.base, f"run-{run_idx}")
        process_dir = os.path.join(run_dir, "process")
        if not os.path.isdir(process_dir):
            skipped_noprocess += 1
            continue

        combined_path = os.path.join(process_dir, "full_trajectory_contacts_by_state_name.csv")
        if os.path.exists(combined_path):
            skipped_exists += 1
            continue

        combined_df = load_and_combine_trajectory_contacts(process_dir, state_names, args.resi_add_num)
        if combined_df is None or combined_df.empty:
            failed += 1
            continue

        try:
            combined_df.to_csv(combined_path, index=True)
            processed += 1
        except Exception:
            failed += 1

    print(
        f"Done. Created combined CSVs: {processed}, already existed: {skipped_exists}, "
        f"no process dir: {skipped_noprocess}, failed to create: {failed}."
    )


if __name__ == "__main__":
    main()


