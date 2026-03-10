import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import os
import hashlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pandas as pd
from sklearn.decomposition import PCA
from config import get_config


def load_pca_model(model_path):
    data = np.load(model_path)
    pca_model = PCA(n_components=data["n_components"])
    pca_model.components_ = data["components"]
    pca_model.mean_ = data["mean"]
    pca_model.explained_variance_ = data["explained_variance"]
    pca_model.explained_variance_ratio_ = data["explained_variance_ratio"]
    pca_model.n_components_ = data["n_components"]
    pca_model.n_features_in_ = data["n_features"]
    return pca_model

def get_time(time_file):
    """
    从文件中读取时间数据。
    """
    with open(time_file, "r") as f:
        lines = f.readlines()
    times = []
    for line in lines:
        if not line.startswith("#") and not line.startswith("@"):
            parts = line.strip().split()
            state, start, end = int(parts[0]), float(parts[1]), float(parts[2])
            times.append((start, end))
    return times if times else None

def get_state_time_files(run_path, state_name):
    """
    根据新的state名称获取对应的时间文件。
    state映射关系：
    - Apo -> state_1
    - ES -> state_2, state_3
    - DS -> state_4
    - PS -> state_5
    """
    state_map = {
        "Apo": ["state_1"],
        "ES": ["state_2", "state_3"],
        "DS": ["state_4"],
        "PS": ["state_5"]
    }
    
    all_times = []
    for state_file in state_map[state_name]:
        time_file = os.path.join(run_path, "process","state_old", f"{state_file}_absolute_time.xvg")
        if os.path.exists(time_file):
            times = get_time(time_file)
            if times:
                all_times.extend(times)
    
    return all_times if all_times else None

def generate_cache_filename(topology_file, trajectory_file, state_number, run_type):
    """
    根据拓扑、轨迹文件路径、状态编号和运行类型生成唯一的缓存文件名。
    """
    combined_path = f"{topology_file}{trajectory_file}state_{state_number}{run_type}"
    hash_value = hashlib.md5(combined_path.encode()).hexdigest()
    return f"aligned_trajectory_{hash_value}.xtc"

def generate_pca_cache_filename(topology_file, trajectory_file, run_type):
    """
    根据pca_getPC_local的缓存文件名格式生成缓存文件名。
    注意：pca_getPC_local不包含run_type参数
    """
    combined_path = f"{topology_file}{trajectory_file}"
    hash_value = hashlib.md5(combined_path.encode()).hexdigest()
    return f"aligned_trajectory_{hash_value}.xtc"

def load_aligned_trajectory(topology_file, trajectory_file, state_number, run_type, cache_dir):
    """
    加载对齐后的轨迹。如果缓存文件不存在，则返回 None。
    """
    cache_filename = generate_cache_filename(topology_file, trajectory_file, state_number, run_type)
    cache_path = os.path.join(cache_dir, cache_filename)
    if os.path.exists(cache_path):
        print(f"Loading aligned trajectory from cache: {cache_path}")
        u = mda.Universe(topology_file, cache_path)
        return u
    return None

# def save_aligned_trajectory(u, topology_file, trajectory_file, state_number, run_type, cache_dir):
#     """
#     保存对齐后的轨迹到缓存文件。
#     """
#     cache_filename = generate_cache_filename(topology_file, trajectory_file, state_number, run_type)
#     cache_path = os.path.join(cache_dir, cache_filename)
#     print(f"Saving aligned trajectory to cache: {cache_path}")
#     with mda.Writer(cache_path, n_atoms=u.atoms.n_atoms) as writer:
#         for ts in u.trajectory:
#             writer.write(u.atoms)

def load_and_align_trajectory(topology_file, trajectory_file, run_type, reference_file, protein_range, state_number, cache_dir):
    """
    加载轨迹。优先使用pca_getPC_local生成的缓存文件，避免重复对齐。
    
    优先级：
    1. 首先尝试加载pca_getPC_hpc生成的缓存文件
    2. 如果不存在，尝试加载state特定的缓存文件
    3. 如果都不存在，才进行对齐操作并保存缓存
    """
    # 优先尝试加载pca_getPC_hpc生成的缓存文件
    pca_cache_filename = generate_pca_cache_filename(topology_file, trajectory_file, run_type)
    pca_cache_path = os.path.join(cache_dir, pca_cache_filename)
    
    if os.path.exists(pca_cache_path):
        print(f"Loading PCA aligned trajectory from cache: {pca_cache_path}")
        u = mda.Universe(topology_file, pca_cache_path)
        return u
    
    # 如果PCA缓存不存在，尝试加载state特定的缓存
    u = load_aligned_trajectory(topology_file, trajectory_file, state_number, run_type, cache_dir)
    if u is not None:
        return u
    else:
        raise ValueError(f"Trajectory not found in cache: {trajectory_file}")
    # # 如果缓存文件都不存在，则加载原始轨迹并进行对齐
    # print(f"Aligning trajectory: {trajectory_file}")
    # u = mda.Universe(topology_file, trajectory_file)
    # ref = mda.Universe(reference_file)
    
    # resid_start, resid_end = protein_range
    # if select_part == "whole":
    #     selection_str = f"resid {resid_start}-{resid_end}"
    # elif select_part == "BB":
    #     selection_str = f"resid {resid_start}-{resid_end} and name BB"
    # else:
    #     raise ValueError(f"Invalid select_part: {select_part}")
    
    # align.AlignTraj(u, ref, select=selection_str, in_memory=True).run()

    # # 保存对齐后的轨迹到缓存
    # save_aligned_trajectory(u, topology_file, trajectory_file, state_number, run_type, cache_dir)
    # return u

def project_to_pca(model, data):
    """
    将新数据投影到已有的 PCA 模型上。
    """
    # 将三维坐标数据reshape为二维矩阵
    data_reshaped = data.reshape(len(data), -1)
    expected_features = model.n_features_in_
    actual_features = data_reshaped.shape[1]
    
    # 验证特征数量是否匹配
    if actual_features != expected_features:
        expected_atoms = expected_features // 3
        actual_atoms = actual_features // 3
        raise ValueError(
            f"Feature dimension mismatch!\n"
            f"PCA model expects {expected_features} features ({expected_atoms} atoms × 3 coordinates)\n"
            f"Current data has {actual_features} features ({actual_atoms} atoms × 3 coordinates)\n"
            f"Please ensure both pca_getPC_local.py and pca_state_hpc.py use the same protein_range (pca_resi_single)."
        )
    
    projected_coordinates = model.transform(data_reshaped)
    return projected_coordinates

def plot_free_energy(pca_coordinates, save_png_file, run_type, explained_variance_ratio, n_bins=100, temp=298, kBT_unit='kJ/mol', ax=None, extent=None):
    """
    根据PCA坐标绘制自由能面图
    """
    # 设置颜色和坐标范围
    color_map = {
        "WT": "#1f77b4",    # 蓝色
        "HOT": "#ff7f0e",   # 橙色
        "Thermo": "#2ca02c", # 绿色
        "FAST": "#d62728"    # 红色
    }
    cmap = plt.cm.Blues_r if run_type in ["WT", "HOT"] else plt.cm.Oranges_r

    # 设置物理常数
    kB = 0.008314462618  # 气体常数 (kJ/mol/K)
    if kBT_unit.lower() == 'kcal/mol':
        kB /= 4.184  # 转换为 kcal/mol/K

    # 计算二维直方图
    x = pca_coordinates[:, 0]
    y = pca_coordinates[:, 1]
    
    # 使用提供的extent或计算新的extent
    if extent is None:
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        dx = (xmax - xmin) * 0.1  # 扩展10%边界
        dy = (ymax - ymin) * 0.1
        extent = [xmin-dx, xmax+dx, ymin-dy, ymax+dy]

    # 计算概率密度
    hist, xedges, yedges = np.histogram2d(x, y, bins=n_bins, range=[extent[:2], extent[2:]])
    hist = hist.T  # 转置以匹配坐标方向
    
    # 添加小值避免log(0)
    hist += 1e-10
    prob = hist / np.sum(hist)
    
    # 计算自由能 (单位: kBT)
    free_energy = -kB * temp * np.log(prob)
    free_energy -= np.min(free_energy)  # 将最小值设为0

    # 创建或使用提供的轴
    is_standalone = ax is None
    if is_standalone:
        plt.figure(figsize=(10, 8), dpi=300)
        ax = plt.gca()

    # 绘制等高线
    levels = np.linspace(0, 15, 16)  # 0-15 kBT，间隔1 kBT
    ctf = ax.contourf(free_energy, 
                     extent=extent,
                     origin='lower',
                     levels=levels,
                     cmap=cmap,
                     alpha=0.8)

    # 绘制等高线标注
    ct = ax.contour(free_energy, 
                   extent=extent,
                   origin='lower',
                   levels=levels,
                   colors='k',
                   linewidths=0.5)
    ax.clabel(ct, inline=True, fontsize=8, fmt='%.1f')

    # 设置坐标轴
    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1] * 100:.1f}%)")
    
    # 如果是独立的图，添加色标和标题并保存
    if is_standalone and save_png_file:
        cbar = plt.colorbar(ctf)
        cbar.set_label(f'Free Energy ({kBT_unit})')
        plt.title(f'Free Energy Landscape ({run_type})')
        plt.savefig(save_png_file, bbox_inches='tight')
        plt.close()

    # 找到自由能最低点坐标
    min_idx = np.unravel_index(np.argmin(free_energy), free_energy.shape)
    min_pc1 = xedges[min_idx[0]]
    min_pc2 = yedges[min_idx[1]]
    
    return (min_pc1, min_pc2), ctf

def plot_states_for_run_type(run_type, state_coordinates_dict, save_png_file, explained_variance_ratio, n_bins=100, temp=298, kBT_unit='kJ/mol'):
    """
    为单个run type的所有状态绘制自由能面图
    """
    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=300)
    axes = axes.flatten()
    
    # 计算全局的extent
    all_coords = np.vstack([coords for coords in state_coordinates_dict.values()])
    xmin, xmax = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
    ymin, ymax = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
    dx = (xmax - xmin) * 0.1
    dy = (ymax - ymin) * 0.1
    global_extent = [xmin-dx, xmax+dx, ymin-dy, ymax+dy]
    
    # 为每个状态绘制自由能面
    for idx, (state_name, coords) in enumerate(state_coordinates_dict.items()):
        ax = axes[idx]
        min_pc, ctf = plot_free_energy(coords, None, run_type, explained_variance_ratio, 
                                     n_bins=n_bins, temp=temp, kBT_unit=kBT_unit,
                                     ax=ax, extent=global_extent)
        ax.set_title(f'{state_name} State')
    
    # 添加色标
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ctf, cax=cbar_ax)
    cbar.set_label(f'Free Energy ({kBT_unit})')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # 保存图像
    plt.savefig(save_png_file, bbox_inches='tight')
    plt.close()

def plot_all_free_energy(pca_coordinates_dict, save_png_file, explained_variance_ratio, n_bins=100, temp=298, kBT_unit='kJ/mol'):
    """
    为所有运行类型绘制自由能面图
    """
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=300)
    axes = axes.flatten()
    
    # 计算全局的extent
    all_coords = np.vstack([coords for coords in pca_coordinates_dict.values()])
    xmin, xmax = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
    ymin, ymax = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
    dx = (xmax - xmin) * 0.1
    dy = (ymax - ymin) * 0.1
    global_extent = [xmin-dx, xmax+dx, ymin-dy, ymax+dy]

    # 为每个运行类型绘制自由能面
    for idx, (run_type, coords) in enumerate(pca_coordinates_dict.items()):
        ax = axes[idx]
        min_pc, ctf = plot_free_energy(coords, None, run_type, explained_variance_ratio, 
                                     n_bins=n_bins, temp=temp, kBT_unit=kBT_unit,
                                     ax=ax, extent=global_extent)
        ax.set_title(f'Free Energy Landscape ({run_type})')

    # 添加色标
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ctf, cax=cbar_ax)
    cbar.set_label(f'Free Energy ({kBT_unit})')

    # 调整布局
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # 保存图像
    plt.savefig(save_png_file, bbox_inches='tight')
    plt.close()

def process_trajectory_batch(u, protein_range, state_time, step=1, selection_str=None):
    """
    使用生成器处理轨迹，减少内存使用
    """
    selected_atoms = u.select_atoms(selection_str)
    
    for ts in u.trajectory[::step]:
        time = u.trajectory.time
        for start, end in state_time:
            if start <= time <= end:
                coords = selected_atoms.positions
                yield coords.copy(), (u.filename, u.trajectory.filename, ts.frame)
                break

def process_run_batch(run_path, run_type, protein_range, state_time, reference_file, cache_dir, state_name, step=1, selection_str=None):
    """
    处理单个run的数据
    """
    topology_file = os.path.join(run_path, "2-eq-noPBC-PET.gro")
    trajectory_file = os.path.join(run_path, "2-eq-noPBC-PET.xtc")
    
    if not os.path.exists(topology_file) or not os.path.exists(trajectory_file):
        # 返回一个空的生成器
        return iter([])
    
    try:
        u = load_and_align_trajectory(topology_file, trajectory_file, run_type, reference_file, protein_range, state_name, cache_dir)
        return process_trajectory_batch(u, protein_range, state_time, step, selection_str)
    except Exception as e:
        print(f"Error loading or processing trajectory {trajectory_file}: {e}")
        return iter([])

def main():
    parser = ArgumentParser(description="PCA analysis for multiple trajectories.")
    parser.add_argument("-cache", type=str, help="cache file dir", default="/M-run-complex-FAST-10/analysis_PCA_new/pca_cache")
    parser.add_argument("-out", type=str, help="output file dir", default="/M-run-complex-FAST-10/analysis_PCA_new/pca_results")
    parser.add_argument("-t", type=str, help="Type of the trajectory", required=True)
    #parser.add_argument("-select", type=str, help="Select the part of protein to do PCA", default="whole")
    parser.add_argument("-align", type=str, help="align file", default="/M-run-complex-WT-10/run-all/run-3/2-eq-noPBC-PET.gro")
    parser.add_argument("-pca_model", type=str, help="pca model file", default="/M-run-complex-FAST-10/analysis_PCA_BB/pca_WT_BB/pca_model.npz")
    args = parser.parse_args()
    
    # 定义所有需要处理的运行类型和状态
    run_types = [args.t]
    states = ["Apo", "ES", "DS", "PS"]
    
    # 定义路径和参数
    pca_model_path = args.pca_model
    pca_dir = args.out
    cache_dir = args.cache
    reference_file = args.align
    
    # 创建缓存目录
    # 注意：优先使用pca_getPC_local生成的缓存文件，避免重复对齐
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(pca_dir, exist_ok=True)
    config = get_config(args.t)
    protein_range = config["pca_resi_single"]
    if args.t == "WT":
        select_string = (f"resid {protein_range[0]}:{protein_range[1]} "
                        + "and not index 559")
    elif args.t == "Thermo":
        select_string = (f"resid {protein_range[0]}:{protein_range[1]} "
                         + "and not (index 346 or index 347)")
    else:
        raise ValueError(f"Invalid type_id: {args.t}")
    # 加载 PCA 模型
    if os.path.exists(pca_model_path):
        pca_model = load_pca_model(pca_model_path)
        print(f"PCA model loaded from {pca_model_path}")
    else:
        raise FileNotFoundError("PCA model not found")
    
    explained_variance_ratio = pca_model.explained_variance_ratio_
    
    # 处理每个运行类型
    for run_type in run_types:
        print(f"\nProcessing run type: {run_type}")
        config = get_config(run_type)
        
        # 收集该run type下所有状态的数据
        state_coordinates = {}
        
        # 处理每个状态
        for state_name in states:
            print(f"\nProcessing state: {state_name}")
            
            # 定义输出路径
            out_put_dir = pca_dir
            save_pca_coord_path = os.path.join(out_put_dir, f"pca_{run_type}_coordinates_{state_name}.csv")
            save_free_energy_path = os.path.join(out_put_dir, f"free_energy_{run_type}_{state_name}.png")
            
            os.makedirs(out_put_dir, exist_ok=True)
            step = 10  # 采样步长
            
            if not os.path.exists(save_pca_coord_path):
                print(f"Generating new PCA coordinates for {run_type} {state_name}")
                
                all_distances_batch = []
                all_frame_info_batch = []
                
                # 标记是否是第一次写入文件
                first_write = True

                # 遍历所有可能的run目录
                for run in tqdm(range(1, 141), desc=f"Processing {run_type} {state_name} runs"):
                    run_path = os.path.join(config['base_dir'], "run-all", f"run-{run}")
                    
                    # 获取状态时间
                    state_time = get_state_time_files(run_path, state_name)
                    if state_time is None:
                        continue
                    
                    # 处理当前run的数据
                    # 会优先使用pca_getPC_local生成的缓存文件，避免重复对齐
                    protein_range = config['pca_resi_single']
                    for coords, frame_info in process_run_batch(run_path, run_type, protein_range, state_time, reference_file, cache_dir, state_name, step, select_string):
                        all_distances_batch.append(coords)
                        all_frame_info_batch.append(frame_info)
                    
                    # 每处理10个run或到最后一个run时，进行PCA投影和保存
                    if len(all_distances_batch) > 0 and (run % 10 == 0 or run == 119):
                        state_data = np.array(all_distances_batch)
                        projected_coordinates = project_to_pca(pca_model, state_data)
                        
                        # 创建包含元数据的DataFrame
                        df_batch = pd.DataFrame(projected_coordinates, columns=['PC1', 'PC2'])
                        df_batch['topology'] = [info[0] for info in all_frame_info_batch]
                        df_batch['trajectory'] = [info[1] for info in all_frame_info_batch]
                        df_batch['frame'] = [info[2] for info in all_frame_info_batch]
                        
                        # 写入文件
                        if first_write:
                            df_batch.to_csv(save_pca_coord_path, index=False)
                            first_write = False
                        else:
                            df_batch.to_csv(save_pca_coord_path, mode='a', header=False, index=False)
                        
                        # 清空列表，释放内存
                        all_distances_batch = []
                        all_frame_info_batch = []
                
                # 绘制最终的自由能面 (从完整文件中加载数据)
                if os.path.exists(save_pca_coord_path):
                    df = pd.read_csv(save_pca_coord_path)
                    if not df.empty:
                         # 存储坐标用于后续组合绘图
                         state_coordinates[state_name] = df[['PC1', 'PC2']].values
                         
                         # 绘制单个状态图
                         min_pc, ctf = plot_free_energy(df[['PC1', 'PC2']].values, save_free_energy_path, run_type, explained_variance_ratio)
                         print(f"Completed analysis for {run_type} {state_name}")
                    else:
                         print(f"No data found in {save_pca_coord_path} for {run_type} {state_name}")
                else:
                     print(f"PCA coordinate file not found after processing for {run_type} {state_name}")

            else:
                print(f"Loading existing PCA coordinates for {run_type} {state_name}")
                # 加载 PCA 投影坐标
                df = pd.read_csv(save_pca_coord_path)
                
                if not df.empty:
                    # 存储坐标用于后续组合绘图
                    state_coordinates[state_name] = df[['PC1', 'PC2']].values
                    
                    # 绘制单个状态图
                    min_pc, ctf = plot_free_energy(df[['PC1', 'PC2']].values, save_free_energy_path, run_type, explained_variance_ratio)
                    print(f"Completed analysis for {run_type} {state_name}")
                else:
                    print(f"PCA coordinate file {save_pca_coord_path} is empty for {run_type} {state_name}")
        
        # 为当前run type绘制所有状态的组合图
        if state_coordinates:
            save_combined_path = os.path.join(out_put_dir, f"free_energy_{run_type}_all_states.png")
            plot_states_for_run_type(run_type, state_coordinates, save_combined_path, explained_variance_ratio)
            print(f"Generated combined plot for all states of {run_type}")
    
    # 由于只处理WT类型，跳过多run type的组合图生成
    print(f"\nCompleted PCA analysis for {run_types[0]} across all states.")

if __name__ == "__main__":
    main() 