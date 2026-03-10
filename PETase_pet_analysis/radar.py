import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import os
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from config import get_config
import subprocess
#plt.rcParams['font.family'] = 'Arial'

def find_nearest(positions, point, cutoff, num_to_find):
    tree = cKDTree(positions)
    indices = tree.query_ball_point(point, cutoff)
    neighbors = positions[indices]
    distances = np.linalg.norm(point - neighbors, axis=1)
    if len(distances) > num_to_find:
        nearest_indices = np.argsort(distances)[:num_to_find]
        nearest_positions = neighbors[nearest_indices]
        #返回原本positions的索引
        global_indices = np.array(indices)[nearest_indices]

    else:
        nearest_indices = np.arange(len(neighbors))
        nearest_positions = neighbors
        global_indices = np.array(indices)[nearest_indices]
    return nearest_positions, global_indices

def radar(gro_file, xtc_file, group_b_select, cutoff, atom_id,num_to_find,time_control,save_xvg):

    u = mda.Universe(gro_file, xtc_file)
    data = []

    # 获取三个resid的选择
    #sel_res = u.select_atoms(" or ".join([f"resid {r}" for r in resids]))
    sel_res = u.select_atoms(f"bynum {atom_id}")
    # 获取GroupB的选择
    sel_b =  u.select_atoms(f"bynum {group_b_select[0]}:{group_b_select[1]}")
    
    for ts in tqdm(u.trajectory[time_control[0]:time_control[1]:time_control[2]]):
        # 计算GroupA的质心
        com_res = sel_res.center_of_mass()

        # 构建KD树,得到GroupB的list
        positions_b = sel_b.positions

        #确保groupB中原子坐标都小于70A，positions_b = positions_b[positions_b[:, 2] < 70]
        # 保存原始的全局索引
        original_indices = np.arange(len(positions_b))
        # 筛选满足条件的局部索引
        mask = positions_b[:, 2] < 70  # 生成布尔掩码
        filtered_positions_b = positions_b[mask]  # 筛选坐标
        filtered_indices = original_indices[mask]  # 对应的全局索引

        nearest_pos, nearest_idx = find_nearest(filtered_positions_b, com_res, cutoff * 10,num_to_find)

        #求出GroupB的质心
        com_b = nearest_pos.mean(axis=0)
        dist_res_b = np.linalg.norm(sel_res.center_of_mass() - com_b) / 10.0
        data.append((ts.time, dist_res_b))

    with open(save_xvg, "w") as f:
        f.write("# time(ps) distance(nm)\n")
        for t, d in data:
            f.write(f"{t:.2f} {d:.3f}\n")
    return data

def plot_radar(data, save_file):
    data = np.array(data)
    plt.figure(dpi=300)
    plt.plot(data[:, 0]/1000, data[:, 1], color = color_id)
    plt.xlabel("Time (ns)", fontsize=14)
    plt.ylabel("Distance (nm)", fontsize=14)
    plt.xlim(0, 1000)
    plt.ylim(0, 7.2)
    #plt.title("Distance between Residue and GroupB", fontsize=16)
    #plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(save_file)
    plt.show()

if __name__ == "__main__":
    ArgumentParser(description="Radar plot for distance between residue and groupB")
    argument = ArgumentParser()
    argument.add_argument("-run", type=int, help="Run number", required=True)
    argument.add_argument("-t", 
                    type=str, 
                    choices=["WT","FAST", "WT119", "WT141","WT188","HOT","Thermo","FAST119","Thermo119","Thermo146"], 
                    required=True,
                    help="Run type: FAST, WT119, or WT141 (case-sensitive)")
    args = argument.parse_args()

    run_num = args.run
    run_type = args.t
    
    # 获取配置
    config = get_config(args.t)
    color_id = "#FFBE7A"
    atom_id = config['sc_ser_160']  #SER 131 的侧链原子
    file_path = os.path.join(config['base_dir'], "run-all", f"run-{run_num}")
    group_b_select = config['group_b']

    gro_file = f"{file_path}/2-eq.gro"
    xtc_file = f"{file_path}/2-eq.xtc"
    save_file = f"{file_path}/process/pbc_rader.png"
    save_xvg = f"{file_path}/process/pbc_rader.xvg"

    if not os.path.exists(os.path.join(file_path, "process")):
        os.makedirs(os.path.join(file_path, "process"))

    time_control = (1, -1, 1) #时间控制,(开始帧，结束帧，步长)
    cutoff = 7.0  # 搜索范围（nm）
    top_num = 5
    
    #检查输入文件是否存在
    # if not os.path.exists(xtc_file):
    #     cmd = f'bash /data/hcz/M-run-complex/analysis/PBC_for_radar_{run_type}.sh {file_path}'
    #     subprocess.run(cmd, shell=True)
    
    radar_data = radar(gro_file, xtc_file, group_b_select, cutoff,atom_id,top_num,time_control,save_xvg)
    plot_radar(radar_data, save_file)
