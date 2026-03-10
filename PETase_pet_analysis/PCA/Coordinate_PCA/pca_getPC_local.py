import MDAnalysis as mda
from MDAnalysis.analysis import align, distances
from sklearn.decomposition import PCA
import numpy as np
import os
import hashlib
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from config import get_config


def generate_cache_filename(topology_file, trajectory_file):
    """
    根据拓扑和轨迹文件路径生成唯一的缓存文件名。
    """
    combined_path = f"{topology_file}{trajectory_file}"
    hash_value = hashlib.md5(combined_path.encode()).hexdigest()
    return f"aligned_trajectory_{hash_value}.xtc"


def load_aligned_trajectory(topology_file, trajectory_file, cache_dir):
    """
    加载对齐后的轨迹。如果缓存文件不存在，则返回 None。
    """
    cache_filename = generate_cache_filename(topology_file, trajectory_file)
    cache_path = os.path.join(cache_dir, cache_filename)
    if os.path.exists(cache_path):
        print(f"Loading aligned trajectory from cache: {cache_path}")
        u = mda.Universe(topology_file, cache_path)
        return u
    return None


def save_aligned_trajectory(u, topology_file, trajectory_file, cache_dir):
    """
    保存对齐后的轨迹到缓存文件。
    """
    cache_filename = generate_cache_filename(topology_file, trajectory_file)
    cache_path = os.path.join(cache_dir, cache_filename)
    print(f"Saving aligned trajectory to cache: {cache_path}")
    with mda.Writer(cache_path, n_atoms=u.atoms.n_atoms) as writer:
        for ts in u.trajectory:
            writer.write(u.atoms)


def load_and_align_trajectory(topology_file, trajectory_file, protein_range, reference_file, cache_dir, select_string):
    """
    加载轨迹并对齐。如果缓存文件存在，则直接加载；否则对齐并保存。
    """
    # 尝试从缓存加载对齐后的轨迹
    u = load_aligned_trajectory(topology_file, trajectory_file, cache_dir)
    if u is not None:
        return u
    else:
        print(f"Trajectory not found in cache: {trajectory_file}")
    # 如果缓存文件不存在，则加载原始轨迹并进行对齐
    print(f"Aligning trajectory: {trajectory_file}")
    u = mda.Universe(topology_file, trajectory_file)
    ref = mda.Universe(reference_file)
    align.AlignTraj(u, ref, select=f"resid {protein_range[0]}:{protein_range[1]}", in_memory=True).run() #这里有align的范围

    # 保存对齐后的轨迹到缓存
    save_aligned_trajectory(u, topology_file, trajectory_file, cache_dir)
    return u


def sample_trajectory(u, atom_selection, step=100):
    """
    对轨迹每隔一定帧数采样，返回采样的坐标。
    """
    sampled_coordinates = []
    for ts in tqdm(u.trajectory[::step], desc="Sampling frames", unit="frame"):
        sampled_coordinates.append(atom_selection.positions.copy())
    return np.array(sampled_coordinates)


def merge_trajectories(trajectories, protein_range, reference_file, cache_dir, step=100, select_string=None):
    """
    合并多条轨迹的采样数据，并返回数据和对应的颜色列表
    """
    merged_data = []
    colors = []
    for top_file, traj_file, traj_type in tqdm(trajectories, desc="Processing trajectories"):
        u = load_and_align_trajectory(top_file, traj_file, protein_range, reference_file, cache_dir, select_string)
        atom_selection = u.select_atoms(select_string)        # 这里选择load的范围
        sampled_data = sample_trajectory(u, atom_selection, step=step)
        
        merged_data.extend(sampled_data)
        # 根据轨迹类型添加颜色
        color = "#BEB8DC" if traj_type == "elastic" else "#FFBE7A"
        colors.extend([color] * len(sampled_data))
    
    return np.array(merged_data), colors


def compute_pca(data, n_components=2):
    """
    对合并的数据执行 PCA 分析。
    """
    pca_model = PCA(n_components=n_components)
    pca_coordinates = pca_model.fit_transform(data.reshape(len(data), -1))
    print("Explained variance ratio:", pca_model.explained_variance_ratio_)
    return pca_model, pca_coordinates


def project_to_pca(model, data):
    """
    将新数据投影到已有的 PCA 模型上。
    """
    projected_coordinates = model.transform(data.reshape(len(data), -1))
    return projected_coordinates


def plot_pca(pca_coordinates, colors, ref_pca, save_png_file):
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(pca_coordinates[:, 0], pca_coordinates[:, 1], 
                c=colors, alpha=0.6, edgecolors='w', label="Trajectories")
    # 用红色五角星标记参考结构
    plt.scatter(ref_pca[:, 0], ref_pca[:, 1], 
                c="red", marker="*", s=300, edgecolor="black", label="Reference")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_png_file)
    plt.close()


def main():
    parser = ArgumentParser(description="PCA analysis for multiple trajectories.")
    parser.add_argument("-cache", type=str, help="cache file dir", default="/M-run-complex-FAST-10/analysis_PCA_Thermo/pca_cache")
    parser.add_argument("-out", type=str, help="output file dir", default="/M-run-complex-FAST-10/analysis_PCA_Thermo/pca_results")
    parser.add_argument("-t", type=str, help="Type of the trajectory", required=True)
    parser.add_argument("-align", type=str, help="align file", default="/M-run-complex-WT-10/run-all/run-3/2-eq-noPBC-PET.gro")
    args = parser.parse_args()
    # 定义颜色
    # elastic_color = "#BEB8DC"
    go_color = "#FFBE7A"
    align_file = args.align
    out_put_dir = args.out
    cache_dir = args.cache
    os.makedirs(out_put_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    config = get_config(args.t)
    protein_range = config["pca_resi_single"]
    step = 100

    if args.t == "WT":
        select_string = (f"resid {protein_range[0]}:{protein_range[1]} "
                        + "and not index 559")
    elif args.t == "Thermo":
        select_string = (f"resid {protein_range[0]}:{protein_range[1]} "
                         + "and not (index 346 or index 347)")
    else:
        raise ValueError(f"Invalid type_id: {args.t}")
    # 保存 PCA 结果
    save_pca_path = os.path.join(out_put_dir, "pca_coordinates.csv")
    save_pca_model_path = os.path.join(out_put_dir, "pca_model.npz")

    if not os.path.exists(save_pca_model_path):
        # 动态生成轨迹路径
        all_trajectories = []
        for run in range(1, 120):  
            base_dir = config['base_dir']
            md_trj_path = f"{base_dir}/run-all/run-{run}"
            if os.path.exists(os.path.join(md_trj_path, "process")):
                topology_file = os.path.join(md_trj_path, "2-eq-noPBC-PET.gro")
                trajectory_file = os.path.join(md_trj_path, "2-eq-noPBC-PET.xtc")

                all_trajectories.append((topology_file, trajectory_file,"go"))

        # 合并所有轨迹的采样数据
        start_time = time.perf_counter()
        print("Merging trajectories...")
        merged_data, colors = merge_trajectories(all_trajectories, protein_range, align_file, cache_dir, step, select_string)
        end_time = time.perf_counter()
        print(f"Time used for merging: {end_time - start_time:.2f} s")

        # 检查是否有数据
        if len(merged_data) == 0:
            print("No data available for PCA analysis. Exiting.")
            return

        # 执行 PCA 分析
        print("Running PCA...")
        start_time = time.perf_counter()
        pca_model, pca_coordinates = compute_pca(merged_data)

        end_time = time.perf_counter()
        print(f"Time used for PCA: {end_time - start_time:.2f} s")

        # 保存 PCA 模型和投影坐标
        # 更完整的保存方式
        np.savez(
            save_pca_model_path,
            components=pca_model.components_,
            mean=pca_model.mean_,
            explained_variance=pca_model.explained_variance_,
            explained_variance_ratio=pca_model.explained_variance_ratio_,
            n_components=pca_model.n_components_,
            n_features=pca_model.n_features_in_
        )
        np.savetxt(save_pca_path, pca_coordinates, delimiter=",", header="PC1,PC2")
        print(f"PCA coordinates saved to {save_pca_path}")
        print(f"PCA model saved to {save_pca_model_path}")
    else:
        # 加载 PCA 模型和投影坐标
        pca_coordinates = np.loadtxt(save_pca_path, delimiter=",", skiprows=1)
        data = np.load(save_pca_model_path)  # 添加 allow_pickle=True
        pca_model = PCA()
        pca_model = PCA(n_components=data["n_components"])
        pca_model.components_ = data["components"]
        pca_model.mean_ = data["mean"]
        pca_model.explained_variance_ = data["explained_variance"]
        pca_model.explained_variance_ratio_ = data["explained_variance_ratio"]
        pca_model.n_components_ = data["n_components"]
        pca_model.n_features_in_ = data["n_features"]
        print(f"PCA coordinates loaded from {save_pca_path}")
        print(f"PCA model loaded from {save_pca_model_path}")

    # 添加参考结构投影
    #随意添加一个结构的投影到PCA坐标系中
    any_pdb = align_file
    #和参考结构align
    any_universe = mda.Universe(any_pdb)
    ref_universe = mda.Universe(align_file)
    align.AlignTraj(any_universe, ref_universe, select=select_string, in_memory=True).run()

    any_atoms = any_universe.select_atoms(select_string)
    any_coords = any_atoms.positions.flatten().reshape(1, -1)
    any_pca = pca_model.transform(any_coords)

    # 绘制 PCA 图像
    save_png_path = os.path.join(out_put_dir, "pca_projection.png")
    #colors = [elastic_color] * 200100 + [go_color] * 200100
    plot_pca(pca_coordinates, colors, any_pca, save_png_path)
    print(f"PCA plot saved to {save_png_path}")


if __name__ == "__main__":
    main()