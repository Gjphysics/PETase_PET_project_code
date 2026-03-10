import os
import sys
import argparse
import pandas as pd


def check_file_coverage(csv_path: str, min_required_ps: float, max_required_ps: float) -> dict:
    if not os.path.exists(csv_path):
        return {
            "exists": False,
            "min_time": None,
            "max_time": None,
            "covers_min": False,
            "covers_max": False,
            "num_rows": 0,
            "issue": "file_missing",
        }

    try:
        df = pd.read_csv(csv_path)
        if "time" not in df.columns:
            return {
                "exists": True,
                "min_time": None,
                "max_time": None,
                "covers_min": False,
                "covers_max": False,
                "num_rows": len(df),
                "issue": "missing_time_column",
            }

        times = pd.to_numeric(df["time"], errors="coerce").dropna()
        if times.empty:
            return {
                "exists": True,
                "min_time": None,
                "max_time": None,
                "covers_min": False,
                "covers_max": False,
                "num_rows": len(df),
                "issue": "no_valid_time_values",
            }

        min_time = float(times.min())
        max_time = float(times.max())
        covers_min = min_time <= min_required_ps + 1e-6
        covers_max = max_time >= max_required_ps - 1e-6

        return {
            "exists": True,
            "min_time": min_time,
            "max_time": max_time,
            "covers_min": covers_min,
            "covers_max": covers_max,
            "num_rows": len(df),
            "issue": None if (covers_min and covers_max) else "coverage_incomplete",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "exists": True,
            "min_time": None,
            "max_time": None,
            "covers_min": False,
            "covers_max": False,
            "num_rows": 0,
            "issue": f"exception:{type(exc).__name__}:{exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check coverage of contact CSVs across runs."
    )
    parser.add_argument(
        "-base",
        type=str,
        default="/M-run-complex-WT-10/run-all",
        help="Base directory containing run-* folders",
    )
    parser.add_argument(
        "-start",
        type=int,
        default=1,
        help="Start run index (inclusive)",
    )
    parser.add_argument(
        "-end",
        type=int,
        default=140,
        help="End run index (inclusive)",
    )
    parser.add_argument(
        "-min_ps",
        type=float,
        default=0.0,
        help="Required minimum time (ps)",
    )
    parser.add_argument(
        "-max_ps",
        type=float,
        default=999800.0,
        help="Required maximum time (ps)",
    )
    parser.add_argument(
        "-out",
        type=str,
        default="/M-run-complex-FAST-10/analysis_contact_PCA/coverage_report.csv",
        help="Output CSV report path",
    )
    args = parser.parse_args()

    results = []
    for run_idx in range(args.start, args.end + 1):
        run_dir = os.path.join(args.base, f"run-{run_idx}")
        csv_path = os.path.join(
            run_dir,
            "process",
            "full_trajectory_contacts_by_state_name.csv",
        )
        res = check_file_coverage(csv_path, args.min_ps, args.max_ps)
        res_row = {
            "run": run_idx,
            "csv_path": csv_path,
            **res,
        }
        results.append(res_row)

    df_report = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_report.to_csv(args.out, index=False)

    num_present = (df_report["exists"]).sum()
    num_missing = (~df_report["exists"]).sum()
    meets_both = (df_report["exists"]) & (df_report["covers_min"]) & (df_report["covers_max"])
    num_ok = meets_both.sum()
    num_bad = (df_report["exists"]) & (~meets_both)
    num_bad_count = num_bad.sum()

    print(f"Checked runs {args.start}-{args.end}")
    print(f"  With file present: {int(num_present)}")
    print(f"  Missing files: {int(num_missing)}")
    print(f"  Satisfy coverage (min<= {args.min_ps}, max>= {args.max_ps}): {int(num_ok)}")
    print(f"  Not satisfying coverage among present: {int(num_bad_count)}")
    
    if num_bad_count > 0:
        bad_df = df_report.loc[num_bad, ["run", "min_time", "max_time", "csv_path"]].sort_values("run")
        print("\nRuns not meeting coverage; showing min_time and max_time (ps):")
        for _, row in bad_df.iterrows():
            print(f"  run-{int(row['run'])}: min_time={row['min_time']}, max_time={row['max_time']} :: {row['csv_path']}")
        # Also save a separate failures CSV beside the main report
        fail_out = os.path.splitext(args.out)[0] + "_failures.csv"
        bad_df.to_csv(fail_out, index=False)
        print(f"Failures saved: {fail_out}")

    print(f"\nReport saved: {args.out}")


if __name__ == "__main__":
    main()


