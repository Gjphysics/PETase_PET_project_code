import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from MDAnalysis.lib.nsgrid import FastNS
from argparse import ArgumentParser
from MDAnalysis.analysis import distances
from config import get_config

def analyze_atoms_near_residues_optimized(topology_file, trajectory_file, group_A, group_B, time_control):

    u = mda.Universe(topology_file, trajectory_file)

    # 获取目标残基的原子组
    residue_group = u.select_atoms(f"bynum {group_A[0]}:{group_A[1]}")
    #residue_group = u.select_atoms(' or '.join(f"resid {r}" for r in group_A))
    # 获取 PET 的原子组，排除原子名称为 W 的原子
    pet_atoms = u.select_atoms(f"bynum {group_B[0]}:{group_B[1]}")

    data = []

    for ts in tqdm(u.trajectory[time_control[0]:time_control[1]:time_control[2]]):  # 遍历帧
        dist_matrix = distances.distance_array(residue_group.positions, pet_atoms.positions)
        #print(dist_matrix)
        contact_scores = contact_function(dist_matrix)
        # 总体 contact 强度
        total_contact_score = np.sum(contact_scores)

        data.append((ts.time, total_contact_score))

    return data

# 激活函数定义,单位是Å
def contact_function(x):
    a=1
    d=8
    return 0.5 - 0.5 * np.tanh(a * (x - d))

def plot_contact_any(data, save_png):
    data = np.array(data)
    plt.figure(dpi=300)
    plt.plot(data[:, 0]/1000, data[:, 1], color = color_id)
    plt.xlabel("Time (ns)", fontsize=14)
    plt.ylabel("Number of Contact", fontsize=14)
    plt.xlim(0, 1000)
    #plt.ylim(0, 7.2)
    plt.savefig(save_png)

def write_data(data, save_xvg):
    with open(save_xvg, "w") as f:
        f.write("# time(ps) atom_counts\n")
        for t, d in data:
            f.write(f"{t:.2f} {d}\n")

if __name__ == "__main__":
    ArgumentParser(description="Radar plot for distance between residue and groupB")
    argument = ArgumentParser()
    argument.add_argument("-run", type=int, help="Run number", required=True)
    argument.add_argument("-t", 
                    type=str, 
                    choices=["WT","FAST", "WT119", "WT141","HOT","Thermo","FAST119","WT188","Thermo119","Thermo146"], 
                    required=True,
                    help="Run type: FAST, WT119, or WT141 (case-sensitive)")
    args = argument.parse_args()

    run_num = args.run
    run_type = args.t

    # 获取配置
    config = get_config(args.t)
    color_id = "#FFBE7A"
    file_path = os.path.join(config['base_dir'], "run-all", f"run-{run_num}")
    
    gro_file = f"{file_path}/2-eq.gro"
    xtc_file = f"{file_path}/2-eq.xtc"
    save_png = f"{file_path}/process/pbc_contact_protein.png"
    save_xvg = f"{file_path}/process/pbc_contact_protein.xvg"

    pet_range = config['group_b']
    resid = config["protein_range"]
    time_control = (1, -1, 5)

    contact_data = analyze_atoms_near_residues_optimized(gro_file, xtc_file, resid, pet_range, time_control)
    #print(contact_data)
    plot_contact_any(contact_data, save_png)
    write_data(contact_data, save_xvg)  #保存数据
