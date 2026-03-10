# PETase-PET Interaction: Coarse-Grained Molecular Dynamics Simulation & Analysis

This repository contains simulation setup files and analysis scripts for studying PETase enzyme–PET substrate interactions using Martini coarse-grained molecular dynamics simulations.

---

## Project Overview

PETase (PET-degrading enzyme) variants are simulated interacting with PET (polyethylene terephthalate) substrate slabs using the **Martini 3** coarse-grained force field in **GROMACS**. The analysis pipeline classifies trajectory frames into functional states (Apo → ES → DS → PS) and characterizes conformational transitions via PCA, clustering, and free energy landscape analysis.

### PETase Variants Studied

| Variant | Description |
|---|---|
| WT | Wild-type PETase |
| FAST | FAST-PETase mutant |
| HOT | HOT-PETase mutant |
| Thermo | Thermostable variant |
| WT119, WT141 | WT single-point mutants |
| Thermo119, Thermo146 | Thermo variant with additional mutations |

---

## Repository Structure

```
project_001_file_to_github/
├── PETase_pet_analysis/           # Analysis scripts
│   ├── contact-binding-pose.py    # Binding pose contact scores (tanh switching)
│   ├── contact-protein.py         # Whole protein-PET contact scores
│   ├── contact_differ.py          # Per-state contact comparisons
│   ├── radar.py                   # SER160-PET distance radar (KD-tree)
│   ├── state.py                   # 5-state classification with hysteresis
│   ├── state_first_arrival_newstate.py  # First arrival time analysis
│   ├── state_reach_newstate.py    # State reachability analysis
│   ├── rmsf_state.py              # Per-state RMSF aggregation
│   ├── rmsf_state_single.py       # Per-run RMSF calculation
│   ├── *_sbatch*.sh               # Slurm job submission templates
│   ├── PCA/
│   │   ├── Contact_PCA/           # PCA on contact matrices
│   │   │   ├── contact_train_pca.py
│   │   │   ├── contact_project_states.py
│   │   │   ├── contact_trajectory_projection.py
│   │   │   ├── contact_trajectory_projection_trj.py
│   │   │   ├── ps_spatial_clustering_with_representatives.py
│   │   │   ├── compare_rotation_straight_proportions.py
│   │   │   ├── contact_coverage_check.py
│   │   │   └── contact_generate_missing_combined.py
│   │   └── Coordinate_PCA/        # PCA on atom coordinates
│   │       ├── pca_getPC_local.py
│   │       ├── pca_state_hpc.py
│   │       ├── pca_clustering_states.py
│   │       ├── pca_clustering_state_trends.py
│   │       └── pca_free_energy.py
│
├── PETase_pet_simulation/         # Simulation setup
│   ├── sbatch-complex-all.sh      # 100-run simulation submission template
│   ├── mdp_files/
│   │   ├── ions.mdp               # Ion placement minimization
│   │   ├── martini_em.mdp         # Energy minimization
│   │   ├── martini_eq_p_large.mdp # NPT equilibration (1 μs)
│   │   └── water.gro              # CG water coordinates
│   └── plumed-files/
│       └── virtual-wall-go.dat    # COM virtual wall restraint
│
└── pet_slab_simulation/           # PET slab glass transition simulation
    ├── run-md-tg-eq.py            # Tg equilibration orchestrator
    ├── run-md-tg-md.py            # Production MD orchestrator
    ├── run-md-tg-t-eq-600.sh      # Initial 600K equilibration
    ├── run-md-tg-t-eq.sh          # Temperature step equilibration
    ├── tg-specific-volume-v3.py   # Glass transition temperature (Tg) calculation
    └── mdp-files/
        ├── martini_em.mdp         # EM parameters
        ├── martini_eq_p.mdp       # NPT equilibration template
        ├── martini_eq_t.mdp       # NVT equilibration template
        ├── martini_run.mdp        # Production MD parameters
        ├── box_CG_W_eq.gro        # Bulk water box coordinates
        ├── martini_v3.0.0.itp     # Martini 3.0.0 force field
        ├── martini_v3.0.0_solvents_v1.itp
        └── martini_v3.0_solvents.itp
```

---

## Software Dependencies

| Software | Version | Purpose |
|---|---|---|
| GROMACS | 2024.2 | Molecular dynamics engine |
| PLUMED | 2.9.2 | Enhanced sampling (virtual walls) |
| Python | 3.x | Analysis scripting |
| MDAnalysis | latest | Trajectory I/O and analysis |
| scikit-learn | latest | PCA, KMeans clustering |
| NumPy | latest | Numerical computing |
| SciPy | latest | KD-tree, Wasserstein distance, curve fitting |
| Matplotlib | latest | Visualization |
| seaborn | latest | Heatmap visualization |
| pandas | latest | Data I/O |

### Python Environment Setup

```bash
conda create -n petase_analysis python=3.10
conda activate petase_analysis
pip install MDAnalysis scikit-learn numpy scipy matplotlib seaborn pandas
```

---
