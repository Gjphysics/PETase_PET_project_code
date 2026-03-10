#!/usr/bin/env python3
"""
PS轨迹空间分布聚类分析 - 基于坐标点分布，忽略路径顺序
=========================================================

不考虑轨迹的时间顺序，只基于投影坐标点的空间分布进行聚类分析。
使用代表性轨迹作为聚类中心，计算其他轨迹与代表性轨迹的空间分布相似度。

特点：
1. 忽略路径的时间顺序
2. 只考虑坐标点在PC1-PC2空间中的分布
3. 使用空间分布特征进行聚类

Requirements:
    pip install scikit-learn matplotlib seaborn tqdm
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
import pickle
from tqdm import tqdm
from config import get_config

# ===================================================================
# 手动指定的代表性轨迹字典
# ===================================================================
REPRESENTATIVE_RUNS = {
    "WT": {
        0: {'run': 9, 'label': 'Rotate_gradual'},
        1: {'run': 26, 'label': 'Rotate_hopping'},
        2: {'run': 5, 'label': 'Straight_right'},
        3: {'run': 58, 'label': 'Straight_wrong'},
    },
    "WT-EL": {
        0: {'run': 9, 'label': 'Rotate_gradual'},
        1: {'run': 26, 'label': 'Rotate_hopping'},
        2: {'run': 5, 'label': 'Straight_right'},
        3: {'run': 58, 'label': 'Straight_wrong'},
    },
    "Thermo": {
        0: {'run': 100, 'label': 'Rotate_gradual'},
        1: {'run': 71, 'label': 'Rotate_hopping'},
        2: {'run': 46, 'label': 'Straight_right'},
        3: {'run': 75, 'label': 'Straight_wrong'},
    },
    "FAST": {
        0: {'run': 40, 'label': 'Rotate_gradual'},
        1: {'run': 44, 'label': 'Rotate_hopping'},
        2: {'run': 46, 'label': 'Straight_right'},
        3: {'run': -1, 'label': 'Straight_wrong'},
    },
    "HOT": {
        0: {'run': 11, 'label': 'Rotate_gradual'},
        1: {'run': -1, 'label': 'Rotate_hopping'},
        2: {'run': 10, 'label': 'Straight_right'},
        3: {'run': 47, 'label': 'Straight_wrong'},
    }
}


 


def load_ps_trajectory_data(run_numbers: list, data_dir: str) -> dict:
    """加载PS到达轨迹的投影数据"""
    trajectory_data = {}
    
    print(f"Loading {len(run_numbers)} PS-reaching trajectories...")
    for run_number in tqdm(run_numbers, desc="Loading trajectory data"):
        npz_path = os.path.join(data_dir, f"contact_projection_run_{run_number}.npz")
        
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            trajectory_data[run_number] = {
                'proj_coords': data['proj_coords'],
                'times': data['times'],
                'state_labels': data['state_labels'],
                'run_number': run_number
            }
        else:
            print(f"Warning: File not found for run {run_number}: {npz_path}")
    
    print(f"Successfully loaded {len(trajectory_data)} trajectories")
    return trajectory_data


def compute_spatial_distribution_features(coords, n_bins=20):
    """
    计算坐标点的空间分布特征，忽略时间顺序
    """
    # 基本统计特征
    features = []
    
    # 1. 基本统计量
    features.extend([
        np.mean(coords[:, 0]),  # PC1 均值
        np.mean(coords[:, 1]),  # PC2 均值
        np.std(coords[:, 0]),   # PC1 标准差
        np.std(coords[:, 1]),   # PC2 标准差
        np.min(coords[:, 0]),   # PC1 最小值
        np.max(coords[:, 0]),   # PC1 最大值
        np.min(coords[:, 1]),   # PC2 最小值
        np.max(coords[:, 1]),   # PC2 最大值
    ])
    
    # 2. 分布范围
    pc1_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
    pc2_range = np.max(coords[:, 1]) - np.min(coords[:, 1])
    features.extend([pc1_range, pc2_range])
    
    # 3. 分位数
    pc1_percentiles = np.percentile(coords[:, 0], [25, 50, 75])
    pc2_percentiles = np.percentile(coords[:, 1], [25, 50, 75])
    features.extend(pc1_percentiles)
    features.extend(pc2_percentiles)
    
    # 4. 空间分布的2D直方图特征
    try:
        H, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=n_bins)
        # 归一化直方图
        H = H / np.sum(H) if np.sum(H) > 0 else H
        # 展平为特征向量
        hist_features = H.flatten()
        features.extend(hist_features)
    except:
        # 如果计算失败，添加零向量
        features.extend([0.0] * (n_bins * n_bins))
    
    # 5. 相关性系数
    correlation = np.corrcoef(coords[:, 0], coords[:, 1])[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    features.append(correlation)
    
    return np.array(features)


def compute_representative_spatial_features(trajectory_data, representative_runs, n_bins=20):
    """计算代表性轨迹的空间分布特征"""
    print("Computing representative trajectory spatial features...")
    
    representative_features = {}
    representative_coords = {}
    
    for cluster_id, run_info in representative_runs.items():
        run_num = run_info['run']
        
        # 如果run为-1，表示这个cluster类型不存在
        if run_num == -1:
            print(f"  Cluster {cluster_id} ({run_info['label']}): No representative trajectory (marked as -1)")
            continue
            
        if run_num not in trajectory_data:
            print(f"Warning: Representative run {run_num} not found in PS trajectory data!")
            continue
            
        coords = trajectory_data[run_num]['proj_coords']
        representative_coords[cluster_id] = coords
        
        # 计算空间分布特征（忽略时间顺序）
        spatial_features = compute_spatial_distribution_features(coords, n_bins=n_bins)
        representative_features[cluster_id] = spatial_features
        
        print(f"  Cluster {cluster_id} (Run {run_num}, {run_info['label']}): {len(coords)} points -> {len(spatial_features)} features")
    
    return representative_features, representative_coords


def compute_trajectory_similarity(coords1, coords2, method='wasserstein'):
    """
    计算两个轨迹的空间分布相似度
    """
    if method == 'wasserstein':
        # 使用Wasserstein距离计算PC1和PC2的分布差异
        try:
            dist_pc1 = wasserstein_distance(coords1[:, 0], coords2[:, 0])
            dist_pc2 = wasserstein_distance(coords1[:, 1], coords2[:, 1])
            return (dist_pc1 + dist_pc2) / 2
        except:
            # 如果计算失败，使用简单的均值差
            mean_diff = np.linalg.norm(np.mean(coords1, axis=0) - np.mean(coords2, axis=0))
            return mean_diff
    
    elif method == 'histogram':
        # 使用2D直方图比较
        try:
            # 计算联合范围
            all_coords = np.vstack([coords1, coords2])
            x_range = [all_coords[:, 0].min(), all_coords[:, 0].max()]
            y_range = [all_coords[:, 1].min(), all_coords[:, 1].max()]
            
            # 计算直方图
            H1, _, _ = np.histogram2d(coords1[:, 0], coords1[:, 1], bins=20, range=[x_range, y_range])
            H2, _, _ = np.histogram2d(coords2[:, 0], coords2[:, 1], bins=20, range=[x_range, y_range])
            
            # 归一化
            H1 = H1 / np.sum(H1) if np.sum(H1) > 0 else H1
            H2 = H2 / np.sum(H2) if np.sum(H2) > 0 else H2
            
            # 计算直方图差异
            return np.sum(np.abs(H1 - H2))
        except:
            return np.inf


def assign_trajectories_to_clusters_spatial(trajectory_data, representative_features, representative_coords,
                                           representative_runs, n_bins=20, similarity_method='wasserstein'):
    """基于空间分布相似度将轨迹分配到聚类中"""
    print("Assigning PS trajectories based on spatial distribution similarity...")
    
    # 检查是否有有效的代表性轨迹
    if len(representative_features) == 0:
        print("Warning: No valid representative trajectories found! All clusters marked as -1.")
        return {}, {}, {}
    
    # 标准化代表性特征（保持cluster_id与特征向量顺序一致）
    cluster_ids_order = list(representative_features.keys())
    repr_feature_matrix = np.array([representative_features[cid] for cid in cluster_ids_order])
    scaler = StandardScaler()
    repr_features_scaled = scaler.fit_transform(repr_feature_matrix)
    
    assignments = {}
    assignment_distances = {}
    assignment_similarities = {}
    
    pbar = tqdm(trajectory_data.items(), desc="Processing PS trajectories")
    for run_number, traj_data in pbar:
        coords = traj_data['proj_coords']
        
        # 方法1：使用空间分布特征进行分配
        spatial_features = compute_spatial_distribution_features(coords, n_bins=n_bins)
        feature_scaled = scaler.transform(spatial_features.reshape(1, -1))[0]
        
        # 计算到各个代表性轨迹的特征距离（与cluster_id对齐）
        feature_distances = {}
        for idx, repr_feature in enumerate(repr_features_scaled):
            cid = cluster_ids_order[idx]
            dist = np.linalg.norm(feature_scaled - repr_feature)
            feature_distances[cid] = dist
        
        # 方法2：使用直接的空间分布相似度
        spatial_similarities = {}
        for cluster_id, repr_coords in representative_coords.items():
            similarity = compute_trajectory_similarity(coords, repr_coords, method=similarity_method)
            spatial_similarities[cluster_id] = similarity
        
        # 综合两种方法进行分配（优先使用空间相似度）
        assigned_cluster = min(spatial_similarities.keys(), key=lambda k: spatial_similarities[k])
        
        assignments[run_number] = assigned_cluster
        assignment_distances[run_number] = feature_distances[assigned_cluster]
        assignment_similarities[run_number] = spatial_similarities[assigned_cluster]
        
        pbar.set_postfix({
            'run': run_number,
            'cluster': assigned_cluster,
            'similarity': f'{spatial_similarities[assigned_cluster]:.3f}'
        })
    
    return assignments, assignment_distances, assignment_similarities


def analyze_spatial_cluster_assignments(assignments, representative_runs, assignment_similarities):
    """分析空间聚类分配结果"""
    cluster_info = {}
    
    print("\nPS Trajectory Spatial Clustering Assignment Results:")
    print("=" * 70)
    
    total_trajectories = len(assignments) if assignments else 1  # 避免除零错误
    
    # 统计每个聚类的轨迹
    for cluster_id in representative_runs.keys():
        run_info = representative_runs[cluster_id]
        
        # 如果代表性轨迹为-1，表示该类型不存在
        if run_info['run'] == -1:
            cluster_info[cluster_id] = {
                'representative_run': -1,
                'representative_label': run_info['label'],
                'assigned_runs': [],
                'count': 0,
                'percentage': 0.0,
                'avg_similarity': 0.0
            }
            print(f"Cluster {cluster_id} (Representative: N/A, {run_info['label']}):")
            print(f"  Assigned trajectories: 0 (0.0%) - No representative trajectory")
            print(f"  Average spatial similarity: N/A")
            print(f"  Run numbers: []")
            print()
            continue
        
        cluster_runs = [run for run, cluster in assignments.items() if cluster == cluster_id]
        
        # 计算该聚类内的平均相似度
        cluster_similarities = [assignment_similarities[run] for run in cluster_runs]
        avg_similarity = np.mean(cluster_similarities) if cluster_similarities else 0
        
        cluster_info[cluster_id] = {
            'representative_run': run_info['run'],
            'representative_label': run_info['label'],
            'assigned_runs': sorted(cluster_runs),
            'count': len(cluster_runs),
            'percentage': len(cluster_runs) / total_trajectories * 100,
            'avg_similarity': avg_similarity
        }
        
        print(f"Cluster {cluster_id} (Representative: Run {run_info['run']}, {run_info['label']}):")
        print(f"  Assigned trajectories: {len(cluster_runs)} ({cluster_info[cluster_id]['percentage']:.1f}%)")
        print(f"  Average spatial similarity: {avg_similarity:.3f}")
        print(f"  Run numbers: {sorted(cluster_runs)}")
        print()
    
    return cluster_info


def calculate_free_energy_surface(coords, n_bins=100, temp=298.0):
    """计算2D自由能面"""
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range  
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    H, xedges, yedges = np.histogram2d(coords[:, 0], coords[:, 1], 
                                      bins=n_bins, 
                                      range=[[x_min, x_max], [y_min, y_max]])
    
    H = H + 1e-10
    kT = 8.314 * temp / 1000  # kJ/mol
    free_energy = -kT * np.log(H / np.max(H))
    
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return free_energy.T, extent


def create_background_coords_from_trajectories(trajectory_data):
    """从轨迹数据中创建背景坐标"""
    all_coords = []
    for run_number, traj_data in trajectory_data.items():
        coords = traj_data['proj_coords']
        all_coords.append(coords)
    
    background_coords = np.vstack(all_coords)
    print(f"Created background coordinates from {len(trajectory_data)} trajectories")
    return background_coords


def plot_spatial_cluster_assignments(trajectory_data, assignments, representative_runs, 
                                    background_coords, output_dir):
    """绘制空间聚类分配结果"""
    free_energy_surf, extent = calculate_free_energy_surface(background_coords)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    colors = plt.cm.Set1(np.linspace(0, 1, len(representative_runs)))
    
    for idx, (cluster_id, run_info) in enumerate(representative_runs.items()):
        ax = axes[idx]
        repr_run = run_info['run']
        repr_label = run_info['label']
        
        # 绘制自由能面背景
        levels = np.linspace(0, 15, 16)
        ax.contourf(free_energy_surf, extent=extent, origin='lower', 
                   levels=levels, cmap=plt.cm.Blues_r, alpha=0.6)
        ax.contour(free_energy_surf, extent=extent, origin='lower',
                  levels=levels, colors='k', linewidths=0.5, alpha=0.4)
        
        # 如果代表性轨迹为-1，显示空图
        if repr_run == -1:
            ax.set_title(f'Cluster {cluster_id} ({repr_label}) - No Data\n'
                        f'Representative: N/A (No representative trajectory)')
            ax.text(0.5, 0.5, 'No Representative\nTrajectory Available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        else:
            # 获取该聚类的所有轨迹
            cluster_runs = [run for run, cluster in assignments.items() if cluster == cluster_id]
            
            # 绘制分配到该聚类的所有轨迹点
            for run_num in cluster_runs:
                if run_num in trajectory_data:
                    coords = trajectory_data[run_num]['proj_coords']
                    alpha = 0.8 if run_num == repr_run else 0.3
                    s = 20 if run_num == repr_run else 5
                    ax.scatter(coords[:, 0], coords[:, 1], color=colors[idx], 
                              alpha=alpha, s=s)
            
            # 高亮代表轨迹
            if repr_run in trajectory_data:
                repr_coords = trajectory_data[repr_run]['proj_coords']
                ax.scatter(repr_coords[:, 0], repr_coords[:, 1], color=colors[idx], 
                          s=30, alpha=1.0, edgecolors='black', linewidths=1,
                          label=f'Representative (run {repr_run})')
            
            ax.set_title(f'Cluster {cluster_id} ({repr_label}) - {len(cluster_runs)} trajectories\n'
                        f'Representative: Run {repr_run} (Spatial Distribution)')
            ax.legend(fontsize=10)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    
    fig.suptitle('PS Trajectory Spatial Distribution Clustering\n(Path Order Ignored)', fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ps_spatial_cluster_assignments.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved spatial cluster assignment plot")


def plot_unified_spatial_cluster_view(trajectory_data, assignments, representative_runs,
                                     background_coords, output_dir):
    """绘制分离的空间聚类视图 - 每个cluster一个子图"""
    free_energy_surf, extent = calculate_free_energy_surface(background_coords)
    
    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(representative_runs)))
    
    # 为每个cluster创建单独的子图
    for idx, (cluster_id, run_info) in enumerate(representative_runs.items()):
        ax = axes[idx]
        repr_run = run_info['run']
        repr_label = run_info['label']
        
        # 绘制自由能面背景
        levels = np.linspace(0, 15, 31)
        im = ax.contourf(free_energy_surf, extent=extent, origin='lower', 
                        levels=levels, cmap='Blues_r', alpha=0.7)
        
        contours = ax.contour(free_energy_surf, extent=extent, origin='lower',
                             levels=levels[::2], colors='k', linewidths=0.5, alpha=0.5)
        
        # 如果代表性轨迹为-1，显示空图
        if repr_run == -1:
            ax.set_title(f'Cluster {cluster_id} ({repr_label})\nNo representative trajectory', 
                        fontsize=12, pad=10)
            ax.text(0.5, 0.5, 'No Representative\nTrajectory Available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        else:
            # 获取该聚类的所有轨迹
            cluster_runs = [run for run, cluster in assignments.items() if cluster == cluster_id]
            
            # 绘制该聚类的所有轨迹点
            for run_num in cluster_runs:
                if run_num in trajectory_data:
                    coords = trajectory_data[run_num]['proj_coords']
                    alpha = 0.8 if run_num == repr_run else 0.3
                    s = 20 if run_num == repr_run else 5
                    ax.scatter(coords[:, 0], coords[:, 1], 
                              color=colors[cluster_id], alpha=alpha, s=s)
            
            # 高亮代表轨迹
            if repr_run in trajectory_data:
                coords = trajectory_data[repr_run]['proj_coords']
                ax.scatter(coords[:, 0], coords[:, 1], 
                          color=colors[cluster_id], s=30, alpha=1.0,
                          edgecolors='black', linewidths=2, 
                          label=f'Representative (Run {repr_run})')
            
            # 设置子图标题和标签
            ax.set_title(f'Cluster {cluster_id} ({repr_label})\n{len(cluster_runs)} trajectories, Rep: Run {repr_run}', 
                        fontsize=12, pad=10)
            ax.legend(fontsize=10)
        
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        
        # 添加颜色条（只在第一个子图）
        if idx == 0:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Free Energy (kJ/mol)', fontsize=10)
    
    # 添加总标题
    fig.suptitle('PS Trajectory Spatial Distribution Clustering - Separate Views\n(Path Order Ignored)', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    output_path = os.path.join(output_dir, 'ps_spatial_clustering_unified.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved separated spatial cluster views: {output_path}")


def plot_cluster_percentage_bar_chart(cluster_info, representative_runs, output_dir):
    """绘制四种类型分类的百分比柱状图"""
    # 准备数据
    cluster_ids = []
    percentages = []
    labels = []
    counts = []
    
    for cluster_id in sorted(cluster_info.keys()):
        info = cluster_info[cluster_id]
        cluster_ids.append(cluster_id)
        percentages.append(info['percentage'])
        labels.append(info['representative_label'])
        counts.append(info['count'])
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 使用不同颜色
    colors = plt.cm.Set1(np.linspace(0, 1, len(cluster_ids)))
    
    # 绘制柱状图
    bars = ax.bar(range(len(cluster_ids)), percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 在柱子上添加百分比和轨迹数量标签
    for i, (bar, percentage, count) in enumerate(zip(bars, percentages, counts)):
        height = bar.get_height()
        if percentage == 0:
            label_text = '0.0%\n(No data)'
        else:
            label_text = f'{percentage:.1f}%\n({count} trajs)'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                label_text, 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 设置x轴标签
    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels([f'Cluster {cid}\n{label}' for cid, label in zip(cluster_ids, labels)], 
                       fontsize=12)
    
    # 设置y轴
    ax.set_ylabel('Percentage of Trajectories (%)', fontsize=14)
    ax.set_ylim(0, max(percentages) * 1.15)
    
    # 添加网格
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # 设置标题
    ax.set_title('PS Trajectory Distribution by Cluster Type\n(Spatial Distribution Clustering)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 美化图形
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'ps_cluster_percentage_bar_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved cluster percentage bar chart: {output_path}")
    
    # 也创建一个水平柱状图版本
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制水平柱状图
    bars = ax.barh(range(len(cluster_ids)), percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 在柱子上添加百分比和轨迹数量标签
    for i, (bar, percentage, count) in enumerate(zip(bars, percentages, counts)):
        width = bar.get_width()
        if percentage == 0:
            label_text = '0.0% (No data)'
        else:
            label_text = f'{percentage:.1f}% ({count} trajs)'
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                label_text, 
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    # 设置y轴标签
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels([f'Cluster {cid}: {label}' for cid, label in zip(cluster_ids, labels)], 
                       fontsize=12)
    
    # 设置x轴
    ax.set_xlabel('Percentage of Trajectories (%)', fontsize=14)
    ax.set_xlim(0, max(percentages) * 1.2)
    
    # 添加网格
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # 设置标题
    ax.set_title('PS Trajectory Distribution by Cluster Type\n(Spatial Distribution Clustering)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 美化图形
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # 保存水平版本
    output_path_h = os.path.join(output_dir, 'ps_cluster_percentage_horizontal_bar_chart.png')
    plt.savefig(output_path_h, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved horizontal cluster percentage bar chart: {output_path_h}")


def load_and_validate_correction_file(correction_file, ps_only_dir, representative_runs):
    """加载并验证手动修正的分类文件"""
    print(f"Loading correction file: {correction_file}")
    
    if not os.path.exists(correction_file):
        raise FileNotFoundError(f"Correction file not found: {correction_file}")
    
    # 读取修正文件
    correction_df = pd.read_csv(correction_file)
    print(f"Loaded correction file with {len(correction_df)} clusters")
    
    # 获取ps-only目录中的所有轨迹号
    npz_files = glob.glob(os.path.join(ps_only_dir, "contact_projection_run_*.npz"))
    ps_run_numbers = set()
    for f in npz_files:
        base = os.path.basename(f)
        try:
            num = int(base.replace('contact_projection_run_', '').replace('.npz', ''))
            ps_run_numbers.add(num)
        except ValueError:
            continue
    
    print(f"Found {len(ps_run_numbers)} PS trajectories in {ps_only_dir}")
    
    # 解析修正文件中的assigned_runs
    corrected_assignments = {}
    all_assigned_runs = set()
    
    print("\nParsing assigned_runs for each cluster:")
    for _, row in correction_df.iterrows():
        cluster_id = row['cluster_id']
        assigned_runs_str = str(row['assigned_runs']).strip()
        
        print(f"  Cluster {cluster_id}: assigned_runs_str = '{assigned_runs_str}'")
        
        if assigned_runs_str and assigned_runs_str != 'nan' and assigned_runs_str != '' and assigned_runs_str != 'nan':
            # 去除可能的引号
            assigned_runs_str = assigned_runs_str.strip('"').strip("'")
            
            # 解析轨迹号列表
            try:
                if ',' in assigned_runs_str:
                    run_numbers = [int(x.strip()) for x in assigned_runs_str.split(',') if x.strip() and x.strip() != '']
                else:
                    # 单个数字
                    if assigned_runs_str.strip():
                        run_numbers = [int(assigned_runs_str.strip())]
                    else:
                        run_numbers = []
                
                print(f"    Parsed run numbers: {run_numbers}")
                
                for run_num in run_numbers:
                    if run_num in corrected_assignments:
                        print(f"    ERROR: Run {run_num} already assigned to cluster {corrected_assignments[run_num]}")
                        raise ValueError(f"Run {run_num} assigned to multiple clusters (already in cluster {corrected_assignments[run_num]}, now trying to assign to cluster {cluster_id})!")
                    corrected_assignments[run_num] = cluster_id
                    all_assigned_runs.add(run_num)
                
                print(f"    Successfully assigned {len(run_numbers)} trajectories to cluster {cluster_id}")
                
            except (ValueError, AttributeError) as e:
                print(f"    Warning: Error parsing assigned_runs for cluster {cluster_id}: {e}")
                print(f"    assigned_runs_str: '{assigned_runs_str}'")
    
    # 验证一致性
    print("\nValidating correction assignments...")
    
    # 检查是否有遗漏的轨迹
    missing_runs = ps_run_numbers - all_assigned_runs
    if missing_runs:
        raise ValueError(f"Missing trajectories in correction: {sorted(missing_runs)}")
    
    # 检查是否有多余的轨迹
    extra_runs = all_assigned_runs - ps_run_numbers
    if extra_runs:
        raise ValueError(f"Extra trajectories in correction (not in ps-only): {sorted(extra_runs)}")
    
    print(f"✓ Validation passed: {len(all_assigned_runs)} trajectories correctly assigned")
    print(f"✓ All PS trajectories accounted for")
    
    return correction_df, corrected_assignments


def recalculate_cluster_statistics(corrected_assignments, trajectory_data, representative_runs, assignment_similarities):
    """基于修正的分配重新计算聚类统计数据"""
    print("Recalculating cluster statistics based on corrected assignments...")
    
    cluster_info = {}
    total_trajectories = len(corrected_assignments)
    
    # 为每个聚类重新计算统计数据
    for cluster_id in representative_runs.keys():
        run_info = representative_runs[cluster_id]
        
        # 获取分配到该聚类的轨迹
        cluster_runs = [run for run, cluster in corrected_assignments.items() if cluster == cluster_id]
        
        if run_info['run'] == -1:
            # 处理没有代表性轨迹的聚类
            cluster_info[cluster_id] = {
                'representative_run': -1,
                'representative_label': run_info['label'],
                'assigned_runs': sorted(cluster_runs),
                'count': len(cluster_runs),
                'percentage': len(cluster_runs) / total_trajectories * 100 if total_trajectories > 0 else 0.0,
                'avg_similarity': 0.0
            }
        else:
            # 计算该聚类内的平均相似度（如果可用）
            cluster_similarities = []
            for run in cluster_runs:
                if run in assignment_similarities:
                    cluster_similarities.append(assignment_similarities[run])
            
            avg_similarity = np.mean(cluster_similarities) if cluster_similarities else 0.0
            
            cluster_info[cluster_id] = {
                'representative_run': run_info['run'],
                'representative_label': run_info['label'],
                'assigned_runs': sorted(cluster_runs),
                'count': len(cluster_runs),
                'percentage': len(cluster_runs) / total_trajectories * 100 if total_trajectories > 0 else 0.0,
                'avg_similarity': avg_similarity
            }
        
        print(f"  Cluster {cluster_id} ({run_info['label']}): {len(cluster_runs)} trajectories ({cluster_info[cluster_id]['percentage']:.1f}%)")
    
    return cluster_info


def save_corrected_results(cluster_info, corrected_assignments, output_dir, type_id):
    """保存修正后的结果"""
    print("Saving corrected clustering results...")
    
    # 保存聚类摘要
    summary_data = []
    for cluster_id, info in cluster_info.items():
        summary_data.append({
            'cluster_id': cluster_id,
            'representative_run': info['representative_run'],
            'representative_label': info['representative_label'],
            'count': info['count'],
            'percentage': info['percentage'],
            'avg_spatial_similarity': info['avg_similarity'],
            'assigned_runs': ','.join(map(str, info['assigned_runs']))
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'ps_spatial_cluster_summary_corrected.csv'), index=False)
    
    # 保存详细的轨迹分配
    detail_data = []
    for run_num, cluster_id in corrected_assignments.items():
        detail_data.append({
            'run_number': run_num,
            'assigned_cluster': cluster_id,
            'representative_run': cluster_info[cluster_id]['representative_run'],
            'representative_label': cluster_info[cluster_id]['representative_label'],
            'spatial_similarity': cluster_info[cluster_id]['avg_similarity']  # 使用平均相似度
        })
    
    detail_df = pd.DataFrame(detail_data).sort_values('run_number')
    detail_df.to_csv(os.path.join(output_dir, 'ps_spatial_trajectory_assignments_corrected.csv'), index=False)
    
    # 创建文本摘要
    summary_text = f"""PS Trajectory Spatial Distribution Clustering Results (CORRECTED)
================================================================

Clustering Method: Manual Correction of Spatial Distribution Clustering
Type: {type_id}
Number of clusters: {len(cluster_info)}

Corrected Representative Trajectory Configuration:
"""
    
    for cluster_id, info in cluster_info.items():
        summary_text += f"""
Cluster {cluster_id} (Representative: Run {info['representative_run']}, {info['representative_label']}):
  - Assigned trajectories: {info['count']} ({info['percentage']:.1f}%)
  - Average spatial similarity: {info['avg_similarity']:.3f}
  - Run numbers: {info['assigned_runs']}
"""
    
    with open(os.path.join(output_dir, 'PS_SPATIAL_CLUSTERING_CORRECTED_SUMMARY.txt'), 'w') as f:
        f.write(summary_text)
    
    print("Corrected clustering results saved:")
    print(f"  - Cluster summary: ps_spatial_cluster_summary_corrected.csv")
    print(f"  - Trajectory assignments: ps_spatial_trajectory_assignments_corrected.csv") 
    print(f"  - Text summary: PS_SPATIAL_CLUSTERING_CORRECTED_SUMMARY.txt")


def save_spatial_clustering_results(cluster_info, assignments, assignment_similarities, 
                                   output_dir, n_bins, similarity_method):
    """保存空间聚类分析结果"""
    # 保存聚类摘要
    summary_data = []
    for cluster_id, info in cluster_info.items():
        summary_data.append({
            'cluster_id': cluster_id,
            'representative_run': info['representative_run'],
            'representative_label': info['representative_label'],
            'count': info['count'],
            'percentage': info['percentage'],
            'avg_spatial_similarity': info['avg_similarity'],
            'assigned_runs': ','.join(map(str, info['assigned_runs']))
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'ps_spatial_cluster_summary.csv'), index=False)
    
    # 保存详细的轨迹分配
    detail_data = []
    for run_num, cluster_id in assignments.items():
        detail_data.append({
            'run_number': run_num,
            'assigned_cluster': cluster_id,
            'representative_run': cluster_info[cluster_id]['representative_run'],
            'representative_label': cluster_info[cluster_id]['representative_label'],
            'spatial_similarity': assignment_similarities[run_num]
        })
    
    detail_df = pd.DataFrame(detail_data).sort_values('run_number')
    detail_df.to_csv(os.path.join(output_dir, 'ps_spatial_trajectory_assignments.csv'), index=False)
    
    # 保存详细数据
    cluster_assignment_data = {}
    for cluster_id, info in cluster_info.items():
        cluster_assignment_data[f'cluster_{cluster_id}'] = {
            'representative_run': info['representative_run'],
            'representative_label': info['representative_label'],
            'assigned_runs': info['assigned_runs'],
            'count': info['count'],
            'avg_similarity': info['avg_similarity']
        }
    
    with open(os.path.join(output_dir, 'ps_spatial_cluster_detailed.pkl'), 'wb') as f:
        pickle.dump(cluster_assignment_data, f)
    
    # 创建文本摘要
    summary_text = f"""PS Trajectory Spatial Distribution Clustering Results
====================================================

Clustering Method: Spatial Distribution (Path Order Ignored)
Parameters:
- Histogram bins: {n_bins}
- Similarity method: {similarity_method}
- Number of representative clusters: {len(cluster_info)}

Representative Trajectory Configuration:
"""
    
    for cluster_id, info in cluster_info.items():
        summary_text += f"""
Cluster {cluster_id} (Representative: Run {info['representative_run']}, {info['representative_label']}):
  - Assigned trajectories: {info['count']} ({info['percentage']:.1f}%)
  - Average spatial similarity: {info['avg_similarity']:.3f}
  - Run numbers: {info['assigned_runs']}
"""
    
    with open(os.path.join(output_dir, 'PS_SPATIAL_CLUSTERING_SUMMARY.txt'), 'w') as f:
        f.write(summary_text)
    
    print("Spatial clustering results saved:")
    print(f"  - Cluster summary: ps_spatial_cluster_summary.csv")
    print(f"  - Trajectory assignments: ps_spatial_trajectory_assignments.csv") 
    print(f"  - Detailed data: ps_spatial_cluster_detailed.pkl")
    print(f"  - Text summary: PS_SPATIAL_CLUSTERING_SUMMARY.txt")
    print(f"  - Cluster percentage bar chart: ps_cluster_percentage_bar_chart.png")
    print(f"  - Horizontal bar chart: ps_cluster_percentage_horizontal_bar_chart.png")


def main():
    parser = argparse.ArgumentParser(description="PS trajectory spatial distribution clustering")
    parser.add_argument('-t', '--type', type=str, default='WT', help='Run type (WT, FAST, Thermo, …)')
    parser.add_argument('--n-bins', type=int, default=200,
                       help='Number of bins for histogram features')
    
    parser.add_argument('--similarity-method', default='wasserstein',
                       choices=['wasserstein', 'histogram'],
                       help='Method for computing trajectory similarity')
    
    parser.add_argument('--correct', action='store_true',
                       help='Use manually corrected clustering results from ps_spatial_cluster_summary_correction.csv')
    
    args = parser.parse_args()

    type_id = args.type
    # 直接从 ps-only 目录读取已分类为到达 PS 的轨迹数据
    analysis_root = "/M-run-complex-FAST-10/analysis_contact_PCA"
    data_dir = os.path.join(analysis_root, type_id, 'each_run_projection_csv', 'ps-only')
    
    # 根据是否使用修正模式设置输出目录
    if args.correct:
        base_output_dir = os.path.join(analysis_root, type_id, 'ps_spatial_clustering_representatives')
        output_dir = os.path.join(base_output_dir, 'correction')
        correction_file = os.path.join(base_output_dir, 'ps_spatial_cluster_summary_correction.csv')
    else:
        output_dir = os.path.join(analysis_root, type_id, 'ps_spatial_clustering_representatives')

    # 选择当前类型的代表性轨迹配置
    if type_id not in REPRESENTATIVE_RUNS:
        raise ValueError(f"Type {type_id} not found in REPRESENTATIVE_RUNS: available {list(REPRESENTATIVE_RUNS.keys())}")
    reps_for_type = REPRESENTATIVE_RUNS[type_id]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    if args.correct:
        print("PS Trajectory Spatial Distribution Clustering (CORRECTION MODE)")
        print("=" * 70)
        print("Method: Manual correction of spatial distribution clustering")
        print(f"Correction file: {correction_file}")
    else:
        print("PS Trajectory Spatial Distribution Clustering")
        print("=" * 60)
        print("Method: Spatial distribution clustering (path order ignored)")
        print(f"Histogram bins: {args.n_bins}")
        print(f"Similarity method: {args.similarity_method}")
    
    print(f"Type: {type_id}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print("Representative clusters:")
    for cid, info in reps_for_type.items():
        print(f"  Cluster {cid}: Run {info['run']} ({info['label']})")
    print()
    
    # 从 ps-only 数据目录列出所有轨迹（按已分类好的 NPZ 文件）
    print("Listing PS-reaching trajectories from ps-only directory...")
    npz_files = glob.glob(os.path.join(data_dir, "contact_projection_run_*.npz"))
    run_numbers = []
    for f in npz_files:
        base = os.path.basename(f)
        try:
            num = int(base.replace('contact_projection_run_', '').replace('.npz', ''))
            run_numbers.append(num)
        except ValueError:
            continue
    run_numbers = sorted(set(run_numbers))
    print(f"Found {len(run_numbers)} PS-reaching trajectories")

    # 加载轨迹数据
    trajectory_data = load_ps_trajectory_data(run_numbers, data_dir)
    
    if len(trajectory_data) == 0:
        print("No PS trajectory data found!")
        return
    
    # 检查代表性轨迹是否存在（忽略-1的情况）
    missing_runs = []
    for cluster_id, run_info in reps_for_type.items():
        run_num = run_info['run']
        if run_num != -1 and run_num not in trajectory_data:
            missing_runs.append(run_num)
    
    if missing_runs:
        print(f"Error: Representative runs not found in PS trajectory data: {missing_runs}")
        print(f"Available PS runs: {sorted(trajectory_data.keys())}")
        return
    
    # 从轨迹数据创建背景坐标
    print("Creating background coordinates...")
    background_coords = create_background_coords_from_trajectories(trajectory_data)
    
    # 根据模式选择不同的处理流程
    if args.correct:
        # === 修正模式 ===
        print("\n" + "="*50)
        print("CORRECTION MODE: Using manually corrected assignments")
        print("="*50)
        
        # 加载并验证修正文件
        correction_df, corrected_assignments = load_and_validate_correction_file(
            correction_file, data_dir, reps_for_type)
        
        # 为了计算相似度，我们仍需要计算代表性轨迹特征（仅用于相似度计算）
        representative_features, representative_coords = compute_representative_spatial_features(
            trajectory_data, reps_for_type, n_bins=args.n_bins)
        
        # 如果有有效的代表性轨迹，计算原始相似度用于参考
        assignment_similarities = {}
        if representative_features and representative_coords:
            print("Computing similarity scores for reference...")
            for run_number, traj_data in trajectory_data.items():
                coords = traj_data['proj_coords']
                cluster_id = corrected_assignments.get(run_number, -1)
                
                if cluster_id in representative_coords:
                    repr_coords = representative_coords[cluster_id]
                    similarity = compute_trajectory_similarity(coords, repr_coords, method=args.similarity_method)
                    assignment_similarities[run_number] = similarity
        
        # 基于修正的分配重新计算聚类统计数据
        cluster_info = recalculate_cluster_statistics(
            corrected_assignments, trajectory_data, reps_for_type, assignment_similarities)
        
        # 将修正的分配转换为与原始格式兼容的格式
        assignments = corrected_assignments
        
    else:
        # === 正常模式 ===
        # 计算代表性轨迹的空间分布特征
        representative_features, representative_coords = compute_representative_spatial_features(
            trajectory_data, reps_for_type, n_bins=args.n_bins)
        
        # 将所有轨迹分配到代表性聚类中
        assignments, assignment_distances, assignment_similarities = assign_trajectories_to_clusters_spatial(
            trajectory_data, representative_features, representative_coords, reps_for_type,
            n_bins=args.n_bins, similarity_method=args.similarity_method)
        
        # 分析聚类分配结果
        cluster_info = analyze_spatial_cluster_assignments(assignments, reps_for_type, assignment_similarities)
    
    # 绘制聚类分配结果
    print("Plotting spatial cluster assignments...")
    plot_spatial_cluster_assignments(trajectory_data, assignments, reps_for_type, 
                                   background_coords, output_dir)
    
    # 绘制统一的聚类视图
    print("Plotting unified spatial cluster view...")
    plot_unified_spatial_cluster_view(trajectory_data, assignments, reps_for_type,
                                     background_coords, output_dir)
    
    # 绘制四种类型分类的百分比柱状图
    print("Plotting cluster percentage bar charts...")
    plot_cluster_percentage_bar_chart(cluster_info, reps_for_type, output_dir)
    
    # 保存结果
    print("Saving results...")
    if args.correct:
        save_corrected_results(cluster_info, assignments, output_dir, type_id)
    else:
        save_spatial_clustering_results(
            cluster_info, assignments, assignment_similarities, output_dir, 
            args.n_bins, args.similarity_method)
    
    print("\nPS spatial distribution clustering completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
