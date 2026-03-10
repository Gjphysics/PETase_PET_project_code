import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import glob
import shutil
import importlib.util


def load_pca_model(model_path: str) -> PCA:
    """Load an existing PCA model stored with numpy.savez."""
    data = np.load(model_path, allow_pickle=True)
    pca = PCA(n_components=int(data["n_components"]))
    pca.components_ = data["components"]
    pca.mean_ = data["mean"]
    pca.explained_variance_ = data["explained_variance"]
    pca.explained_variance_ratio_ = data["explained_variance_ratio"]
    pca.n_components_ = int(data["n_components"])
    pca.n_features_in_ = int(data["n_features"])
    return pca


def calculate_free_energy_surface(coordinates, n_bins=100, temp=298, kBT_unit="kJ/mol"):
    """Compute a 2-D free-energy surface from pre-projected PCA coordinates."""
    # Boltzmann constant
    kB = 0.008314462618  # kJ/mol/K
    if kBT_unit.lower() == "kcal/mol":
        kB /= 4.184

    x, y = coordinates[:, 0], coordinates[:, 1]

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    # add 10% padding around the data so contour edges are visible
    dx = (xmax - xmin) * 0.1 if xmax > xmin else 1.0
    dy = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
    extent = [xmin - dx, xmax + dx, ymin - dy, ymax + dy]

    hist, xedges, yedges = np.histogram2d(x, y, bins=n_bins, range=[extent[:2], extent[2:]])
    hist = hist.T  # transpose to match x/y orientation

    # avoid log(0)
    hist += 1e-12
    prob = hist / np.sum(hist)

    free_energy = -kB * temp * np.log(prob)
    free_energy -= np.min(free_energy)  # normalise minimum to 0

    return free_energy, extent


def load_full_trajectory_data_for_run(base_dir: str, run_number: int) -> tuple:
    """Load complete trajectory contact data for a specific run.
    
    Returns:
        tuple: (contact_data, times, state_labels) or (None, None, None) if failed
    """
    run_dir = os.path.join(base_dir, f"run-{run_number}")
    csv_path = os.path.join(run_dir, "process", "full_trajectory_contacts_by_state_name.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None, None, None
    
    try:
        # Read the full trajectory file
        df = pd.read_csv(csv_path)
        
        if "time" not in df.columns or "state_label" not in df.columns:
            print(f"Warning: Missing required columns (time, state_label) in {csv_path}")
            return None, None, None
        
        # Get time and state information
        times = df["time"].to_numpy(dtype=float)
        state_labels = df["state_label"].to_numpy(dtype=str)
        
        # Get residue columns (should be numeric column names like '30', '31', ..., '292')
        numeric_cols = [col for col in df.columns if isinstance(col, str) and col.isdigit()]
        
        if not numeric_cols:
            print(f"Warning: No numeric residue columns found in {csv_path}")
            return None, None, None
        
        # Sort columns by numeric value
        numeric_cols.sort(key=int)  # Sort by integer value
        
        # Extract contact data
        contact_data = df[numeric_cols].to_numpy(dtype=float)
        
        print(f"Loaded run {run_number}: {len(contact_data)} frames, time range: {times[0]:.1f}-{times[-1]:.1f} ps")
        print(f"  States found: {sorted(set(state_labels))}")
        
        return contact_data, times, state_labels
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None, None, None


def plot_run_projection(run_number: int, contact_data: np.ndarray, times: np.ndarray, 
                       state_labels: np.ndarray, pca_model: PCA, 
                       background_coords: np.ndarray, output_path: str,
                       csv_dir: str = None, save_csv: bool = True,
                       n_bins: int = 100, temp: float = 298.0, unit: str = "kJ/mol",
                       show_endpoints: bool = True):
    """Plot projection of a single run's contact data onto PCA space and optionally save data."""
    
    # Calculate background free energy surface
    free_energy_surf, extent = calculate_free_energy_surface(background_coords, n_bins, temp, unit)
    
    # Set up the plot
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Plot background free energy surface in grayscale
    levels = np.linspace(0, 15, 16)
    plt.contourf(free_energy_surf,
                 extent=extent,
                 origin='lower',
                 levels=levels,
                 cmap='gray',
                 alpha=0.3)
    plt.contour(free_energy_surf,
                extent=extent,
                origin='lower',
                levels=levels,
                colors='gray',
                linewidths=0.5,
                alpha=0.5)
    
    # Define colors for different states
    state_colors = {
        'Apo': '#1f77b4',  # blue
        'ES': '#ff7f0e',   # orange
        'DS': '#2ca02c',   # green
        'PS': '#d62728'    # red
    }
    
    # Check feature dimension compatibility
    if contact_data.shape[1] != pca_model.n_features_in_:
        raise ValueError(
            f"Feature dimension mismatch: "
            f"{contact_data.shape[1]} vs PCA expects {pca_model.n_features_in_}")
    
    # Project all contact data to PCA space
    projected_coords = pca_model.transform(contact_data)
    
    # Get unique states for plotting in desired order
    all_states = set(state_labels)
    desired_order = ['Apo', 'ES', 'DS', 'PS']
    unique_states = [state for state in desired_order if state in all_states]
    # Add any remaining states not in the desired order
    unique_states.extend(sorted(all_states - set(unique_states)))
    
    # Plot each state's data with different colors
    legend_handles = []
    for state in unique_states:
        state_mask = state_labels == state
        state_coords = projected_coords[state_mask]
        
        if len(state_coords) == 0:
            continue
        
        # Plot scatter points
        color = state_colors.get(state, 'black')
        scatter = plt.scatter(state_coords[:, 0], state_coords[:, 1],
                            c=color, label=f'{state} ({len(state_coords)} points)',
                            s=8, alpha=0.7, edgecolors='none')
        legend_handles.append(scatter)
    
    # Mark the start and end points of the trajectory for this run
    if show_endpoints and len(projected_coords) > 0:
        start_xy = projected_coords[0, :2]
        end_xy = projected_coords[-1, :2]
        start_handle = plt.scatter(start_xy[0], start_xy[1],
                                   s=80, marker='o', facecolors='none', edgecolors='k',
                                   linewidths=1.5, label='Start')
        end_handle = plt.scatter(end_xy[0], end_xy[1],
                                 s=80, marker='X', c='k',
                                 linewidths=1.0, label='End')
        legend_handles.extend([start_handle, end_handle])
    
    # Add labels and legend
    plt.xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.title(f"Contact PCA Projection - Run {run_number}")
    
    if legend_handles:
        plt.legend(handles=legend_handles, loc='best', framealpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved projection plot for run {run_number}: {output_path}")
    
    # Save CSV data if requested
    if save_csv and csv_dir:
        # Create DataFrame with real time information
        df = pd.DataFrame({
            'PC1': projected_coords[:, 0],
            'PC2': projected_coords[:, 1],
            'time': times,
            'state': state_labels,
            'run_number': [run_number] * len(projected_coords)
        })
        
        # Save CSV
        csv_path = os.path.join(csv_dir, f"contact_projection_run_{run_number}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved projection data for run {run_number}: {csv_path}")
        
        # Also save NPZ format for compatibility with existing analysis
        npz_path = os.path.join(csv_dir, f"contact_projection_run_{run_number}.npz")
        
        # Normalize times to 0-1 for compatibility
        norm_times = (times - times.min()) / (times.max() - times.min() + 1e-12)
        
        np.savez(npz_path,
                 proj_coords=projected_coords.astype(np.float32),  # Match reference dtype
                 times=times.astype(np.float64),  # Real time data
                 norm_times=norm_times.astype(np.float64),  # Normalized time
                 state_labels=state_labels.astype('U10'),  # State labels
                 run_number=run_number,
                 run_type="contact_projection")
        print(f"Saved projection data (NPZ) for run {run_number}: {npz_path}")


def find_all_runs(base_dir: str) -> list:
    """Find all available run numbers in the base directory."""
    runs = []
    for run_dir in glob.glob(os.path.join(base_dir, "run-*")):
        if os.path.isdir(run_dir):
            run_name = os.path.basename(run_dir)
            try:
                run_number = int(run_name.split('-')[1])
                runs.append(run_number)
            except (ValueError, IndexError):
                continue
    return sorted(runs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project individual run contact data onto trained contact PCA (paths resolved by -t)")
    parser.add_argument("-start", type=int, default=1)
    parser.add_argument("-end", type=int, default=140)
    parser.add_argument(
        "-run",
        type=int,
        default=None,
        help="Process single run number (if not specified, process all runs in range)"
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Skip saving CSV projection data"
    )
    parser.add_argument(
        "--no_endpoints",
        action="store_true",
        help="Do not mark start/end points on the trajectory plot"
    )
    parser.add_argument(
        "-t",
        type=str,
        required=True,
        help=(
            "Type identifier (e.g., WT, MUT1). Automatically resolves: model, background coordinates, "
            "CSV directories, output directories, and base run-all path."
        )
    )
    
    args = parser.parse_args()

    # Auto-resolve key paths based on type identifier
    type_id = args.t
    type_root = f"/M-run-complex-FAST-10/analysis_contact_PCA/{type_id}"
    # Analysis paths
    args.out_dir = os.path.join(type_root, "each_run_projection")
    args.csv_dir = os.path.join(type_root, "each_run_projection_csv")
    args.model = os.path.join(type_root, "contact_pca_model.npz")
    args.background_csv = os.path.join(type_root, "contact_pca_coordinates_ALL_STATES.csv")
    # Resolve run base directory for this type using local analysis_contact_PCA/config.py
    config_path = "/M-run-complex-FAST-10/analysis_contact_PCA/config.py"
    spec = importlib.util.spec_from_file_location("analysis_pca_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    type_config = config_module.get_config(type_id)
    args.base = os.path.join(type_config["base_dir"], "run-all")
    print(f"Base path: {args.base}")

    # Load PCA model
    print("Loading PCA model...")
    pca_model = load_pca_model(args.model)
    
    # Load background coordinates for free energy surface
    print("Loading background coordinates...")
    if os.path.exists(args.background_csv):
        bg_df = pd.read_csv(args.background_csv)
        background_coords = bg_df[["PC1", "PC2"]].values
    else:
        print(f"Warning: Background CSV {args.background_csv} not found, using empty background")
        background_coords = np.array([[0, 0]])  # dummy data
    
    # Create output directories
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.no_csv:
        os.makedirs(args.csv_dir, exist_ok=True)
    
    # Determine which runs to process
    if args.run is not None:
        # Process single run
        run_numbers = [args.run]
    else:
        # Process all runs in range
        all_runs = find_all_runs(args.base)
        run_numbers = [r for r in all_runs if args.start <= r <= args.end]
    
    print(f"Processing {len(run_numbers)} runs: {run_numbers}")
    
    # Process each run
    failed_runs = []
    for run_number in tqdm(run_numbers, desc="Processing runs"):
        try:
            # Load full trajectory contact data for this run
            contact_data, times, state_labels = load_full_trajectory_data_for_run(args.base, run_number)
            
            if contact_data is None:
                print(f"No valid contact data found for run {run_number}, skipping")
                failed_runs.append(run_number)
                continue
            
            # Categorize by whether the trajectory reaches PS
            reaches_ps = np.any(state_labels == 'PS')
            category_dir_name = 'ps-only' if reaches_ps else 'no-ps'
            
            # Prepare categorized output directories
            run_out_dir = os.path.join(args.out_dir, category_dir_name)
            os.makedirs(run_out_dir, exist_ok=True)
            
            if not args.no_csv:
                run_csv_dir = os.path.join(args.csv_dir, category_dir_name)
                os.makedirs(run_csv_dir, exist_ok=True)
            else:
                run_csv_dir = None
            
            # Generate output path
            output_path = os.path.join(run_out_dir, f"run-{run_number}.png")
            
            # Create projection plot
            plot_run_projection(run_number, contact_data, times, state_labels, pca_model, 
                              background_coords, output_path,
                              csv_dir=run_csv_dir,
                              save_csv=not args.no_csv,
                              show_endpoints=not args.no_endpoints)
            
        except Exception as e:
            print(f"Error processing run {run_number}: {e}")
            failed_runs.append(run_number)
            continue
    
    # Summary
    successful_runs = len(run_numbers) - len(failed_runs)
    print(f"\nCompleted processing {successful_runs}/{len(run_numbers)} runs")
    if failed_runs:
        print(f"Failed runs: {failed_runs}")
    
    print(f"Results saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
