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
                       show_endpoints: bool = True, global_extent: list = None):
    """Plot projection of a single run's contact data onto PCA space and optionally save data."""
    
    # Calculate background free energy surface
    free_energy_surf, extent = calculate_free_energy_surface(background_coords, n_bins, temp, unit)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Plot background free energy surface in grayscale
    levels = np.linspace(0, 15, 16)
    ax.contourf(free_energy_surf,
                extent=extent,
                origin='lower',
                levels=levels,
                cmap='gray',
                alpha=0.3)
    ax.contour(free_energy_surf,
               extent=extent,
               origin='lower',
               levels=levels,
               colors='gray',
               linewidths=0.5,
               alpha=0.5)
    
    # Check feature dimension compatibility
    if contact_data.shape[1] != pca_model.n_features_in_:
        raise ValueError(
            f"Feature dimension mismatch: "
            f"{contact_data.shape[1]} vs PCA expects {pca_model.n_features_in_}")
    
    # Project all contact data to PCA space
    projected_coords = pca_model.transform(contact_data)
    
    # Extract PC1 and PC2 coordinates
    pc1 = projected_coords[:, 0]
    pc2 = projected_coords[:, 1]
    
    # Normalize time for colormap (0 to 1)
    time_normalized = (times - times.min()) / (times.max() - times.min() + 1e-12)
    
    # Create line segments for trajectory
    from matplotlib.collections import LineCollection
    
    # Create points array for line segments
    points = np.array([pc1, pc2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create LineCollection with time-based colors
    lc = LineCollection(segments, cmap='viridis', linewidth=2, alpha=0.8)
    lc.set_array(time_normalized[:-1])  # Use time values for coloring
    
    # Add line collection to plot
    line = ax.add_collection(lc)
    
    # Add colorbar for time
    cbar = fig.colorbar(line, ax=ax, label='Normalized Time')
    cbar.set_label('Time (ns)', fontsize=12)
    
    # Optionally update colorbar labels to show actual time in ns
    time_min_ns = times.min() / 1000  # Convert ps to ns
    time_max_ns = times.max() / 1000
    cbar.ax.set_ylabel(f'Time ({time_min_ns:.1f} - {time_max_ns:.1f} ns)', fontsize=12)
    
    # Mark the start and end points of the trajectory
    if show_endpoints and len(projected_coords) > 0:
        start_xy = projected_coords[0, :2]
        end_xy = projected_coords[-1, :2]
        ax.scatter(start_xy[0], start_xy[1],
                  s=150, marker='o', facecolors='white', edgecolors='black',
                  linewidths=2, label='Start', zorder=10)
        ax.scatter(end_xy[0], end_xy[1],
                  s=150, marker='*', c='red', edgecolors='black',
                  linewidths=1.5, label='End', zorder=10)
        ax.legend(loc='best', framealpha=0.9, fontsize=10)
    
    # Add labels
    ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% variance)", fontsize=12)
    ax.set_title(f"Contact PCA Trajectory Projection - Run {run_number}", fontsize=14)
    
    # Set fixed axis limits to ensure consistent scales across all plots
    # Use global_extent if provided (from pre-scan), otherwise use background extent
    if global_extent is not None:
        ax.set_xlim(global_extent[0], global_extent[1])
        ax.set_ylim(global_extent[2], global_extent[3])
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    
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
    args.out_dir = os.path.join(type_root, "each_run_projection_trj")
    args.csv_dir = os.path.join(type_root, "each_run_projection_csv_trj")
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
    
    # Pre-scan all runs to determine global axis limits if background is dummy/missing
    global_extent = None
    if background_coords.shape[0] == 1 and np.allclose(background_coords, [[0, 0]]):
        print("\nNo valid background data. Pre-scanning all runs to determine global axis limits...")
        all_projected_coords = []
        
        for run_number in tqdm(run_numbers, desc="Pre-scanning runs"):
            try:
                contact_data, _, _ = load_full_trajectory_data_for_run(args.base, run_number)
                if contact_data is None:
                    continue
                if contact_data.shape[1] != pca_model.n_features_in_:
                    continue
                projected = pca_model.transform(contact_data)
                all_projected_coords.append(projected[:, :2])  # PC1 and PC2 only
            except Exception as e:
                print(f"Error pre-scanning run {run_number}: {e}")
                continue
        
        if all_projected_coords:
            all_coords = np.vstack(all_projected_coords)
            x_all, y_all = all_coords[:, 0], all_coords[:, 1]
            xmin, xmax = np.min(x_all), np.max(x_all)
            ymin, ymax = np.min(y_all), np.max(y_all)
            dx = (xmax - xmin) * 0.1 if xmax > xmin else 1.0
            dy = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
            global_extent = [xmin - dx, xmax + dx, ymin - dy, ymax + dy]
            print(f"Global extent calculated: PC1=[{global_extent[0]:.2f}, {global_extent[1]:.2f}], "
                  f"PC2=[{global_extent[2]:.2f}, {global_extent[3]:.2f}]")
        else:
            print("Warning: No valid projection data found during pre-scan. Will use per-plot extents.")
    
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
                              show_endpoints=not args.no_endpoints,
                              global_extent=global_extent)
            
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
