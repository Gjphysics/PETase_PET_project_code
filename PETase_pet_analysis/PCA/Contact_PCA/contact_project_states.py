import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from config import get_config

def load_pca_model(model_path: str) -> PCA:
    data = np.load(model_path, allow_pickle=True)
    pca = PCA(n_components=int(data["n_components"]))
    pca.components_ = data["components"]
    pca.mean_ = data["mean"]
    pca.explained_variance_ = data["explained_variance"]
    pca.explained_variance_ratio_ = data["explained_variance_ratio"]
    pca.n_components_ = int(data["n_components"])
    pca.n_features_in_ = int(data["n_features"])
    return pca


def plot_free_energy(coords: np.ndarray, out_png: str, explained_variance_ratio: np.ndarray, n_bins: int = 100, temp: float = 298.0, unit: str = "kJ/mol") -> None:
    kB = 0.008314462618
    if unit.lower() == "kcal/mol":
        kB /= 4.184

    x = coords[:, 0]
    y = coords[:, 1]
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    dx = (xmax - xmin) * 0.1 if xmax > xmin else 1.0
    dy = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
    extent = [xmin - dx, xmax + dx, ymin - dy, ymax + dy]

    hist, xedges, yedges = np.histogram2d(x, y, bins=n_bins, range=[extent[:2], extent[2:]])
    hist = hist.T
    hist += 1e-12
    prob = hist / np.sum(hist)
    F = -kB * temp * np.log(prob)
    F -= np.min(F)

    plt.figure(figsize=(8, 6), dpi=300)
    levels = np.linspace(0, 15, 16)
    ctf = plt.contourf(F, extent=extent, origin="lower", levels=levels, cmap=plt.cm.Blues_r, alpha=0.85)
    ct = plt.contour(F, extent=extent, origin="lower", levels=levels, colors='k', linewidths=0.4)
    plt.clabel(ct, inline=True, fontsize=7, fmt='%.1f')
    cbar = plt.colorbar(ctf)
    cbar.set_label(f"Free Energy ({unit})")
    plt.xlabel(f"PC1 ({explained_variance_ratio[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({explained_variance_ratio[1]*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_free_energy_subplot(ax, coords: np.ndarray, title: str, explained_variance_ratio: np.ndarray, n_bins: int = 100, temp: float = 298.0, unit: str = "kJ/mol") -> None:
    kB = 0.008314462618
    if unit.lower() == "kcal/mol":
        kB /= 4.184

    x = coords[:, 0]
    y = coords[:, 1]
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    dx = (xmax - xmin) * 0.1 if xmax > xmin else 1.0
    dy = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
    extent = [xmin - dx, xmax + dx, ymin - dy, ymax + dy]

    hist, xedges, yedges = np.histogram2d(x, y, bins=n_bins, range=[extent[:2], extent[2:]])
    hist = hist.T
    hist += 1e-12
    prob = hist / np.sum(hist)
    F = -kB * temp * np.log(prob)
    F -= np.min(F)

    levels = np.linspace(0, 15, 16)
    ctf = ax.contourf(F, extent=extent, origin="lower", levels=levels, cmap=plt.cm.Blues_r, alpha=0.85)
    ct = ax.contour(F, extent=extent, origin="lower", levels=levels, colors='k', linewidths=0.4)
    ax.clabel(ct, inline=True, fontsize=6, fmt='%.1f')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]*100:.1f}%)", fontsize=10)
    
    return ctf


def plot_free_energy_subplot_unified(ax, coords: np.ndarray, title: str, explained_variance_ratio: np.ndarray, global_extent: list, n_bins: int = 100, temp: float = 298.0, unit: str = "kJ/mol") -> None:
    kB = 0.008314462618
    if unit.lower() == "kcal/mol":
        kB /= 4.184

    x = coords[:, 0]
    y = coords[:, 1]
    
    # Use global extent if provided, otherwise calculate from current data
    if global_extent:
        extent = global_extent
    else:
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        dx = (xmax - xmin) * 0.1 if xmax > xmin else 1.0
        dy = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
        extent = [xmin - dx, xmax + dx, ymin - dy, ymax + dy]

    hist, xedges, yedges = np.histogram2d(x, y, bins=n_bins, range=[extent[:2], extent[2:]])
    hist = hist.T
    hist += 1e-12
    prob = hist / np.sum(hist)
    F = -kB * temp * np.log(prob)
    F -= np.min(F)

    levels = np.linspace(0, 15, 16)
    ctf = ax.contourf(F, extent=extent, origin="lower", levels=levels, cmap=plt.cm.Blues_r, alpha=0.85)
    ct = ax.contour(F, extent=extent, origin="lower", levels=levels, colors='k', linewidths=0.4)
    ax.clabel(ct, inline=True, fontsize=6, fmt='%.1f')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]*100:.1f}%)", fontsize=10)
    
    # Set axis limits to match global extent
    if global_extent:
        ax.set_xlim(global_extent[0], global_extent[1])
        ax.set_ylim(global_extent[2], global_extent[3])
    
    return ctf


def plot_combined_subplots(state_coords_dict: dict, out_png: str, explained_variance_ratio: np.ndarray, n_bins: int = 100, temp: float = 298.0, unit: str = "kJ/mol") -> None:
    """Plot four subplots for Apo, ES, DS, PS states with unified axis ranges"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.suptitle('Free Energy Landscapes for Different States', fontsize=16, fontweight='bold')
    
    state_order = ['Apo', 'ES', 'DS', 'PS']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # First, calculate global axis ranges from all states
    all_x_coords = []
    all_y_coords = []
    for state in state_order:
        if state in state_coords_dict and len(state_coords_dict[state]) > 0:
            all_x_coords.extend(state_coords_dict[state][:, 0])
            all_y_coords.extend(state_coords_dict[state][:, 1])
    
    if all_x_coords and all_y_coords:
        x_min, x_max = np.min(all_x_coords), np.max(all_x_coords)
        y_min, y_max = np.min(all_y_coords), np.max(all_y_coords)
        # Add 10% padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        global_extent = [x_min, x_max, y_min, y_max]
    else:
        global_extent = None
    
    ctf_list = []
    for i, state in enumerate(state_order):
        row, col = positions[i]
        ax = axes[row, col]
        
        if state in state_coords_dict and len(state_coords_dict[state]) > 0:
            ctf = plot_free_energy_subplot_unified(ax, state_coords_dict[state], f'{state} State', explained_variance_ratio, global_extent, n_bins, temp, unit)
            ctf_list.append(ctf)
        else:
            ax.text(0.5, 0.5, f'No data for {state}', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{state} State', fontsize=12, fontweight='bold')
            if global_extent:
                ax.set_xlim(global_extent[0], global_extent[1])
                ax.set_ylim(global_extent[2], global_extent[3])
            ctf_list.append(None)
    
    # Add colorbar
    if any(ctf is not None for ctf in ctf_list):
        # Find the first valid contourf for colorbar
        valid_ctf = next(ctf for ctf in ctf_list if ctf is not None)
        # Create a separate axis for colorbar to avoid overlap
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(valid_ctf, cax=cbar_ax)
        cbar.set_label(f"Free Energy ({unit})", fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)  # Make room for colorbar
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Project per-state contacts onto trained contact PCA and plot free energy")
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
    parser.add_argument(
        "-states",
        type=str,
        default="Apo,ES,DS,PS",
        help="Comma-separated state names matching single_residue_contact_{state}.csv",
    )

    args = parser.parse_args()
    args.base = os.path.join(get_config(args.t)["base_dir"], "run-all")
    args.out_dir = os.path.join("/M-run-complex-FAST-10/analysis_contact_PCA", args.t)
    args.model = os.path.join(args.out_dir, "contact_pca_model.npz")
    args.resi_add_num = get_config(args.t)["resi_add_num"]

    pca = load_pca_model(args.model)
    explained = pca.explained_variance_ratio_

    state_names = [s.strip() for s in args.states.split(",") if s.strip()]
    
    # Store all coordinates for combined plot
    all_coords_list = []
    all_state_labels = []
    
    # Store coordinates for each state separately for subplot
    state_coords_dict = {}

    for state in state_names:
        coords_list: list[np.ndarray] = []
        for run_idx in range(args.start, args.end + 1):
            csv_path = os.path.join(
                args.base,
                f"run-{run_idx}",
                "process",
                f"single_residue_contact_{state}.csv",
            )
            if not os.path.exists(csv_path):
                continue
            
            try:
                # Read the file with proper header handling
                with open(csv_path, 'r') as f:
                    first_line = f.readline().strip()
                
                if not first_line.startswith('#'):
                    print(f"Warning: {csv_path} does not start with #, skipping")
                    continue
                
                # Parse headers: # time,resid_1,resid_2,...,resid_263
                headers = [h.strip() for h in first_line[1:].split(',')]
                df = pd.read_csv(csv_path, names=headers, skiprows=1, comment='#')
                
                # Convert resid_1, resid_2, etc. to 30, 31, etc.
                renamed_cols = {}
                for col in df.columns:
                    if isinstance(col, str) and col.startswith('resid_'):
                        try:
                            res_num = int(col.split('_')[1])
                            renamed_cols[col] = res_num + args.resi_add_num
                        except ValueError:
                            renamed_cols[col] = col
                    else:
                        renamed_cols[col] = col
                
                df = df.rename(columns=renamed_cols)
                
                # Remove time column and get only residue columns (should be 30, 31, ..., 292)
                residue_cols = [col for col in df.columns if isinstance(col, (int, np.integer))]
                if not residue_cols:
                    print(f"Warning: No residue columns found in {csv_path}")
                    continue
                
                X = df[residue_cols].to_numpy(dtype=float)
                
                # Ensure feature dimension matches PCA training
                if X.shape[1] != pca.n_features_in_:
                    raise ValueError(
                        f"Feature dimension mismatch for {csv_path}: {X.shape[1]} vs PCA expects {pca.n_features_in_}"
                    )
                
                coords = pca.transform(X)
                coords_list.append(coords)
                
                # Store for combined plot
                all_coords_list.append(coords)
                all_state_labels.extend([state] * len(coords))
                
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
                continue

        if not coords_list:
            print(f"No data found for state {state}; skipping")
            continue

        coords_all = np.vstack(coords_list)
        
        # Store coordinates for this state
        state_coords_dict[state] = coords_all
        
        out_csv = os.path.join(args.out_dir, f"contact_pca_coordinates_{state}.csv")
        out_png = os.path.join(args.out_dir, f"contact_free_energy_{state}.png")
        os.makedirs(args.out_dir, exist_ok=True)
        pd.DataFrame(coords_all, columns=["PC1", "PC2"]).to_csv(out_csv, index=False)
        plot_free_energy(coords_all, out_png, explained)
        print(f"Saved {state} PCA coords -> {out_csv}")
        print(f"Saved {state} free energy plot -> {out_png}")

    # Plot combined free energy surface with all states (excluding Apo)
    if all_coords_list:
        print("\nGenerating combined free energy surface (excluding Apo)...")
        all_coords_combined = np.vstack(all_coords_list)
        
        # Save combined coordinates
        combined_csv = os.path.join(args.out_dir, "contact_pca_coordinates_ALL_STATES.csv")
        combined_df = pd.DataFrame(all_coords_combined, columns=["PC1", "PC2"])
        combined_df["state_label"] = all_state_labels
        combined_df.to_csv(combined_csv, index=False)
        print(f"Saved combined coordinates -> {combined_csv}")
        
        # Plot combined free energy surface
        combined_png = os.path.join(args.out_dir, "contact_free_energy_ALL_STATES.png")
        plot_free_energy(all_coords_combined, combined_png, explained)
        print(f"Saved combined free energy plot -> {combined_png}")
        
        # Create combined plot excluding Apo
        non_apo_coords = []
        non_apo_labels = []
        for i, label in enumerate(all_state_labels):
            if label != "Apo":
                non_apo_coords.append(all_coords_combined[i])
                non_apo_labels.append(label)
        
        if len(non_apo_coords) > 0:
            non_apo_coords = np.array(non_apo_coords)
            non_apo_csv = os.path.join(args.out_dir, "contact_pca_coordinates_NON_APO_STATES.csv")
            non_apo_df = pd.DataFrame(non_apo_coords, columns=["PC1", "PC2"])
            non_apo_df["state_label"] = non_apo_labels
            non_apo_df.to_csv(non_apo_csv, index=False)
            print(f"Saved non-Apo coordinates -> {non_apo_csv}")
            
            non_apo_png = os.path.join(args.out_dir, "contact_free_energy_NON_APO_STATES.png")
            plot_free_energy(non_apo_coords, non_apo_png, explained)
            print(f"Saved non-Apo free energy plot -> {non_apo_png}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"Total data points: {len(all_coords_combined)}")
        for state in state_names:
            count = all_state_labels.count(state)
            if count > 0:
                print(f"  {state}: {count} points")
        
        if len(non_apo_coords) > 0:
            print(f"\nNon-Apo combined:")
            print(f"  Total non-Apo points: {len(non_apo_coords)}")
            for state in set(non_apo_labels):
                count = non_apo_labels.count(state)
                print(f"    {state}: {count} points")
    else:
        print("No data found for any state; cannot generate combined plot")
    
    # Generate four-subplot figure
    if state_coords_dict:
        print("\nGenerating four-subplot free energy landscape...")
        subplot_png = os.path.join(args.out_dir, "contact_free_energy_FOUR_SUBPLOTS.png")
        plot_combined_subplots(state_coords_dict, subplot_png, explained)
        print(f"Saved four-subplot free energy landscape -> {subplot_png}")
    else:
        print("No state data available for subplot generation")


if __name__ == "__main__":
    main()


