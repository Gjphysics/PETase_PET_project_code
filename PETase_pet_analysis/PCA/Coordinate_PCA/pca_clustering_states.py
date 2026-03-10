"""
多状态PCA数据聚类分析脚本

使用固定的极小值点坐标对不同状态进行聚类分析
调整图片比例与自由能面图一致
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import time

# Apo态的四个极小值点坐标（固定使用）
FIXED_MINIMA_CENTERS = [
    (-18.407, 1.143),   # 极小值点 1
    (5.536, -20.413),   # 极小值点 2
    (2.271, 16.712),    # 极小值点 3
    (22.405, 4.736)     # 极小值点 5
]

def fast_clustering_with_fixed_centers(pca_csv_path, output_dir, state_name, use_all_data=False):
    """使用固定极小值点坐标执行快速聚类分析"""
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"使用固定的4个极小值点坐标进行聚类：")
    for i, (pc1, pc2) in enumerate(FIXED_MINIMA_CENTERS):
        print(f"  极小值点 {i+1}: PC1={pc1:.3f}, PC2={pc2:.3f}")
    
    if use_all_data:
        print("启用全数据模式：将使用所有数据点进行轮廓系数计算和绘图")
    
    # 加载数据
    print(f"\n正在加载{state_name}状态的PCA数据: {pca_csv_path}")
    start_time = time.time()
    pca_df = pd.read_csv(pca_csv_path)
    load_time = time.time() - start_time
    print(f"成功加载数据：{len(pca_df):,} 个数据点 (耗时: {load_time:.2f} 秒)")
    
    # 执行聚类
    print(f"\n使用4个固定极小值点对{state_name}状态进行聚类...")
    coordinates = pca_df[['PC1', 'PC2']].values
    
    # 初始化K-means
    init_centers = np.array(FIXED_MINIMA_CENTERS)
    kmeans = KMeans(
        n_clusters=4,
        init=init_centers,
        n_init=1,
        max_iter=300,
        random_state=42,
        verbose=1
    )
    
    # 执行聚类
    clustering_start = time.time()
    cluster_labels = kmeans.fit_predict(coordinates)
    clustering_time = time.time() - clustering_start
    print(f"聚类计算完成，耗时: {clustering_time:.2f} 秒")
    
    # 计算评估指标
    print("正在计算评估指标...")
    eval_start = time.time()
    
    if use_all_data or len(coordinates) <= 50000:
        # 使用全部数据计算轮廓系数
        silhouette_avg = silhouette_score(coordinates, cluster_labels)
        if use_all_data and len(coordinates) > 50000:
            print(f"轮廓系数 (使用全部 {len(coordinates):,} 个数据点): {silhouette_avg:.4f}")
        else:
            print(f"轮廓系数: {silhouette_avg:.4f}")
    else:
        # 对于大数据集，采样计算轮廓系数
        sample_size = 50000
        sample_indices = np.random.choice(len(coordinates), sample_size, replace=False)
        sample_coords = coordinates[sample_indices]
        sample_labels = cluster_labels[sample_indices]
        silhouette_avg = silhouette_score(sample_coords, sample_labels)
        print(f"轮廓系数 (基于 {sample_size:,} 个采样点): {silhouette_avg:.4f}")
    
    eval_time = time.time() - eval_start
    print(f"评估指标计算耗时: {eval_time:.2f} 秒")
    
    # 计算聚类统计
    print("正在计算聚类统计信息...")
    cluster_centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    
    pca_df['cluster'] = cluster_labels
    total_points = len(pca_df)
    
    cluster_stats = []
    print(f"\n{state_name}状态聚类结果：")
    print(f"簇内平方和 (Inertia): {inertia:.2f}")
    print(f"迭代次数: {kmeans.n_iter_}")
    print("\n聚类统计信息：")
    print("-" * 60)
    
    for i in range(4):
        cluster_data = pca_df[pca_df['cluster'] == i]
        count = len(cluster_data)
        percentage = (count / total_points) * 100
        
        final_center = cluster_centers[i]
        pc1_mean = cluster_data['PC1'].mean()
        pc1_std = cluster_data['PC1'].std()
        pc2_mean = cluster_data['PC2'].mean()
        pc2_std = cluster_data['PC2'].std()
        
        stats = {
            'cluster_id': i + 1,
            'count': count,
            'percentage': percentage,
            'final_center_pc1': final_center[0],
            'final_center_pc2': final_center[1],
            'pc1_mean': pc1_mean,
            'pc1_std': pc1_std,
            'pc2_mean': pc2_mean,
            'pc2_std': pc2_std
        }
        cluster_stats.append(stats)
        
        print(f"聚类 {i+1}:")
        print(f"  数据点数量: {count:,} ({percentage:.2f}%)")
        print(f"  最终聚类中心: PC1={final_center[0]:.3f}, PC2={final_center[1]:.3f}")
        print(f"  PC1 统计: {pc1_mean:.3f} ± {pc1_std:.3f}")
        print(f"  PC2 统计: {pc2_mean:.3f} ± {pc2_std:.3f}")
        print()
    
    # 保存结果
    print("正在保存结果...")
    
    # 1. 保存带聚类标签的数据
    output_csv = output_dir / f"pca_data_with_clusters_{state_name}.csv"
    pca_df.to_csv(output_csv, index=False)
    print(f"已保存带聚类标签的数据：{output_csv}")
    
    # 2. 保存聚类中心
    centers_file = output_dir / f"cluster_centers_{state_name}.csv"
    centers_df = pd.DataFrame(cluster_centers, columns=['PC1', 'PC2'])
    centers_df.index.name = 'cluster_id'
    centers_df.to_csv(centers_file)
    print(f"已保存聚类中心坐标：{centers_file}")
    
    # 3. 保存统计信息
    stats_file = output_dir / f"clustering_statistics_{state_name}.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"PCA数据聚类分析结果 ({state_name}状态)\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("分析参数:\n")
        f.write(f"- 状态: {state_name}\n")
        f.write(f"- 数据文件: {pca_csv_path}\n")
        f.write(f"- 数据点总数: {len(pca_df):,}\n")
        f.write(f"- 聚类数量: 4\n")
        f.write(f"- 使用全数据模式: {'是' if use_all_data else '否'}\n")
        f.write(f"- 轮廓系数: {silhouette_avg:.4f}\n")
        f.write(f"- 簇内平方和: {inertia:.2f}\n")
        f.write(f"- 迭代次数: {kmeans.n_iter_}\n\n")
        
        f.write("使用的固定极小值点坐标 (来自Apo状态):\n")
        f.write("-" * 30 + "\n")
        for i, (pc1, pc2) in enumerate(FIXED_MINIMA_CENTERS):
            f.write(f"极小值点 {i+1}: PC1={pc1:.3f}, PC2={pc2:.3f}\n")
        f.write("\n")
        
        f.write("最终聚类中心:\n")
        f.write("-" * 30 + "\n")
        for i, (pc1, pc2) in enumerate(cluster_centers):
            f.write(f"聚类中心 {i+1}: PC1={pc1:.3f}, PC2={pc2:.3f}\n")
        f.write("\n")
        
        f.write("聚类统计信息:\n")
        f.write("-" * 30 + "\n")
        for stats in cluster_stats:
            f.write(f"聚类 {stats['cluster_id']}:\n")
            f.write(f"  数据点数量: {stats['count']:,} ({stats['percentage']:.2f}%)\n")
            f.write(f"  PC1 统计: {stats['pc1_mean']:.3f} ± {stats['pc1_std']:.3f}\n")
            f.write(f"  PC2 统计: {stats['pc2_mean']:.3f} ± {stats['pc2_std']:.3f}\n")
            f.write("\n")
        
        # 计算时间统计
        total_time = clustering_time + eval_time + load_time
        f.write("性能统计:\n")
        f.write("-" * 30 + "\n")
        f.write(f"数据加载时间: {load_time:.2f} 秒\n")
        f.write(f"聚类计算时间: {clustering_time:.2f} 秒\n")
        f.write(f"评估指标时间: {eval_time:.2f} 秒\n")
        f.write(f"总计算时间: {total_time:.2f} 秒\n")
    
    print(f"已保存统计信息：{stats_file}")
    
    # 4. 生成调整比例的可视化图
    print("正在生成可视化图...")
    create_adjusted_plot(pca_df, cluster_centers, FIXED_MINIMA_CENTERS, state_name, output_dir, use_all_data)
    
    return cluster_stats, silhouette_avg, inertia

def create_adjusted_plot(pca_df, cluster_centers, fixed_centers, state_name, output_dir, use_all_data=False):
    """创建与自由能面图比例一致的可视化图"""
    
    # 根据use_all_data参数决定是否采样
    if use_all_data:
        plot_data = pca_df
        plot_sample_size = len(pca_df)
        print(f"绘图模式：使用全部 {len(pca_df):,} 个数据点")
    else:
        # 为了提高绘图速度，采样50k个点
        plot_sample_size = min(50000, len(pca_df))
        if len(pca_df) > plot_sample_size:
            plot_data = pca_df.sample(n=plot_sample_size, random_state=42)
            print(f"绘图模式：采样 {plot_sample_size:,} 个数据点用于绘图")
        else:
            plot_data = pca_df
            print(f"绘图模式：使用全部 {len(pca_df):,} 个数据点")
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#9B59B6']
    
    # 使用和自由能面图一致的图形比例 (10x8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    
    # 1. 聚类散点图 - 调整为和自由能面图一致的比例
    for i in range(4):
        cluster_data = plot_data[plot_data['cluster'] == i]
        count = len(pca_df[pca_df['cluster'] == i])
        percentage = (count / len(pca_df)) * 100
        
        ax1.scatter(cluster_data['PC1'], cluster_data['PC2'], 
                   c=colors[i], alpha=0.6, s=12, 
                   label=f'Cluster {i+1} ({percentage:.1f}%)')
    
    # 标注最终聚类中心
    for i, (center_x, center_y) in enumerate(cluster_centers):
        ax1.scatter(center_x, center_y, c='black', marker='x', s=200, linewidth=3, zorder=5)
        ax1.annotate(f'C{i+1}', (center_x, center_y), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'))
    
    # 标注固定的初始极小值点
    for i, (init_x, init_y) in enumerate(fixed_centers):
        ax1.scatter(init_x, init_y, c='red', marker='o', s=150, 
                   facecolors='none', edgecolors='red', linewidth=3, zorder=5)
        ax1.annotate(f'M{i+1}', (init_x, init_y), 
                    xytext=(-8, -20), textcoords='offset points',
                    fontsize=10, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='red'))
    
    # 设置坐标轴和标题
    ax1.set_xlabel('PC1', fontsize=14)
    ax1.set_ylabel('PC2', fontsize=14)
    
    # 根据是否显示全部数据调整标题
    if use_all_data or len(pca_df) <= plot_sample_size:
        title = f'PCA Clustering Results - {state_name} State\n({len(pca_df):,} total points, all shown)'
    else:
        title = f'PCA Clustering Results - {state_name} State\n({len(pca_df):,} total points, {plot_sample_size:,} shown)'
    
    ax1.set_title(title, fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 设置与自由能面图一致的坐标轴范围
    pc1_range = pca_df['PC1'].max() - pca_df['PC1'].min()
    pc2_range = pca_df['PC2'].max() - pca_df['PC2'].min()
    pc1_margin = pc1_range * 0.05
    pc2_margin = pc2_range * 0.05
    
    ax1.set_xlim(pca_df['PC1'].min() - pc1_margin, pca_df['PC1'].max() + pc1_margin)
    ax1.set_ylim(pca_df['PC2'].min() - pc2_margin, pca_df['PC2'].max() + pc2_margin)
    
    # 设置相同的纵横比
    ax1.set_aspect('equal', adjustable='box')
    
    # 2. 聚类占比饼图
    cluster_counts = [len(pca_df[pca_df['cluster'] == i]) for i in range(4)]
    cluster_percentages = [(count / len(pca_df)) * 100 for count in cluster_counts]
    labels = [f"Cluster {i+1}\n({perc:.1f}%)\n{count:,} points" for i, (perc, count) in enumerate(zip(cluster_percentages, cluster_counts))]
    
    wedges, texts, autotexts = ax2.pie(cluster_percentages, labels=labels, colors=colors[:4], 
                                      autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    ax2.set_title(f'Cluster Distribution - {state_name} State', fontsize=14)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = output_dir / f"clustering_result_{state_name}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"已保存可视化图：{output_path}")
    
    # 额外生成单独的高质量散点图
    create_high_quality_scatter(pca_df, cluster_centers, fixed_centers, state_name, output_dir, use_all_data)

def create_high_quality_scatter(pca_df, cluster_centers, fixed_centers, state_name, output_dir, use_all_data=False):
    """创建高质量的散点图，与自由能面图比例完全一致"""
    
    # 根据use_all_data参数决定是否采样
    if use_all_data:
        plot_data = pca_df
        plot_sample_size = len(pca_df)
    else:
        # 采样数据用于绘图
        plot_sample_size = min(50000, len(pca_df))
        if len(pca_df) > plot_sample_size:
            plot_data = pca_df.sample(n=plot_sample_size, random_state=42)
        else:
            plot_data = pca_df
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # 使用与自由能面图完全一致的尺寸和比例
    plt.figure(figsize=(10, 8), dpi=150)
    
    # 为每个聚类绘制散点
    for i in range(4):
        cluster_data = plot_data[plot_data['cluster'] == i]
        count = len(pca_df[pca_df['cluster'] == i])
        percentage = (count / len(pca_df)) * 100
        
        plt.scatter(cluster_data['PC1'], cluster_data['PC2'], 
                   c=colors[i], alpha=0.6, s=15, 
                   label=f'Cluster {i+1} ({percentage:.1f}%)')
    
    # 标注最终聚类中心
    for i, (center_x, center_y) in enumerate(cluster_centers):
        plt.scatter(center_x, center_y, c='black', marker='x', s=300, linewidth=4, zorder=5)
        plt.annotate(f'Center {i+1}', (center_x, center_y), 
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'))
    
    # 标注固定的初始极小值点
    for i, (init_x, init_y) in enumerate(fixed_centers):
        plt.scatter(init_x, init_y, c='red', marker='o', s=150, 
                   facecolors='none', edgecolors='red', linewidth=3, zorder=5)
        plt.annotate(f'Minima {i+1}', (init_x, init_y), 
                    xytext=(-8, -20), textcoords='offset points',
                    fontsize=10, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='red'))
    
    # 设置坐标轴和标题
    plt.xlabel('PC1', fontsize=14)
    plt.ylabel('PC2', fontsize=14)
    
    # 根据是否显示全部数据调整标题
    if use_all_data or len(pca_df) <= plot_sample_size:
        title = f'PCA Clustering Results - {state_name} State\n({len(pca_df):,} points, all shown)'
    else:
        title = f'PCA Clustering Results - {state_name} State\n({plot_sample_size:,} sampled from {len(pca_df):,} points)'
    
    plt.title(title, fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 设置与自由能面图一致的坐标轴范围和比例
    pc1_range = pca_df['PC1'].max() - pca_df['PC1'].min()
    pc2_range = pca_df['PC2'].max() - pca_df['PC2'].min()
    pc1_margin = pc1_range * 0.05
    pc2_margin = pc2_range * 0.05
    
    plt.xlim(pca_df['PC1'].min() - pc1_margin, pca_df['PC1'].max() + pc1_margin)
    plt.ylim(pca_df['PC2'].min() - pc2_margin, pca_df['PC2'].max() + pc2_margin)
    
    # 设置相同的纵横比
    plt.gca().set_aspect('equal', adjustable='box')
    
    # 保存高质量散点图
    output_path = output_dir / f"clustering_detailed_{state_name}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"已保存详细散点图：{output_path}")

def analyze_single_state(type_id, state, threshold=3.0, use_all_data=False):
    """分析单个状态的聚类"""
    
    # 构建文件路径
    pca_csv_path = f"/M-run-complex-FAST-10/analysis_PCA_new/pca_results/pca_{type_id}_coordinates_{state}.csv"
    
    # 输出目录
    output_dir = f"/M-run-complex-FAST-10/analysis_PCA_new/clustering_results_all_states/{type_id}_{state}"
    
    print(f"=" * 80)
    print(f"开始分析 {type_id}_{state} 状态的聚类")
    if use_all_data:
        print(f"模式：使用全部数据点进行聚类和绘图")
    else:
        print(f"模式：大数据集将采样计算轮廓系数和绘图")
    print(f"=" * 80)
    
    try:
        cluster_stats, silhouette_score, inertia = fast_clustering_with_fixed_centers(
            pca_csv_path, output_dir, state, use_all_data
        )
        
        print(f"\n{state}状态聚类分析完成！")
        return cluster_stats, silhouette_score, inertia
        
    except Exception as e:
        print(f"{state}状态分析过程中发生错误: {str(e)}")
        return None, None, None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多状态PCA数据聚类分析')
    parser.add_argument('-t', '--type', type=str, default='WT', help='分析类型ID')
    parser.add_argument('-states', '--states', nargs='+', default=['Apo', 'ES', 'DS', 'PS'],
                        help='要分析的状态列表（默认: Apo ES DS PS）')
    parser.add_argument('-thre', '--threshold', type=float, default=3.0, help='能量阈值（用于目录命名）')
    parser.add_argument('-all', '--all', action='store_true', help='使用所有数据点进行聚类和绘图')
    
    args = parser.parse_args()
    
    print("多状态PCA聚类分析")
    print("=" * 80)
    print(f"分析类型: {args.type}")
    print(f"分析状态: {', '.join(args.states)}")
    print(f"全数据模式: {'启用' if args.all else '禁用（大数据集将采样）'}")
    print(f"使用固定极小值点坐标（来自Apo状态）:")
    for i, (pc1, pc2) in enumerate(FIXED_MINIMA_CENTERS):
        print(f"  极小值点 {i+1}: PC1={pc1:.3f}, PC2={pc2:.3f}")
    print("=" * 80)
    
    # 记录总体分析结果
    all_results = {}
    total_start_time = time.time()
    
    # 对每个状态进行分析
    for state in args.states:
        state_start_time = time.time()
        
        cluster_stats, silhouette_score, inertia = analyze_single_state(args.type, state, args.threshold, args.all)
        
        if cluster_stats is not None:
            all_results[state] = {
                'cluster_stats': cluster_stats,
                'silhouette_score': silhouette_score,
                'inertia': inertia,
                'analysis_time': time.time() - state_start_time
            }
        
        print(f"\n{state}状态分析耗时: {time.time() - state_start_time:.2f} 秒")
        print("-" * 80)
    
    total_time = time.time() - total_start_time
    
    # 生成总体结果汇总
    create_summary_report(all_results, args.type, args.states, total_time, args.all)
    
    print(f"\n所有状态聚类分析完成！")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"结果保存在: /M-run-complex-FAST-10/analysis_PCA_new/clustering_results_all_states/")

def create_summary_report(all_results, type_id, states, total_time, use_all_data=False):
    """创建总体结果汇总报告"""
    
    summary_dir = f"/M-run-complex-FAST-10/analysis_PCA_new/clustering_results_all_states"
    Path(summary_dir).mkdir(parents=True, exist_ok=True)
    
    summary_file = f"{summary_dir}/{type_id}_all_states_clustering_summary.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"多状态PCA数据聚类分析总结报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"分析类型: {type_id}\n")
        f.write(f"分析状态: {', '.join(states)}\n")
        f.write(f"全数据模式: {'启用' if use_all_data else '禁用（大数据集采样）'}\n")
        f.write(f"总分析时间: {total_time:.2f} 秒\n\n")
        
        f.write("使用的固定极小值点坐标 (来自Apo状态):\n")
        f.write("-" * 40 + "\n")
        for i, (pc1, pc2) in enumerate(FIXED_MINIMA_CENTERS):
            f.write(f"极小值点 {i+1}: PC1={pc1:.3f}, PC2={pc2:.3f}\n")
        f.write("\n")
        
        # 为每个状态生成结果汇总
        for state in states:
            if state in all_results:
                result = all_results[state]
                f.write(f"{state}状态聚类结果:\n")
                f.write("-" * 30 + "\n")
                f.write(f"轮廓系数: {result['silhouette_score']:.4f}\n")
                f.write(f"簇内平方和: {result['inertia']:.2f}\n")
                f.write(f"分析时间: {result['analysis_time']:.2f} 秒\n")
                
                f.write("聚类分布:\n")
                total_points = sum(stats['count'] for stats in result['cluster_stats'])
                for stats in result['cluster_stats']:
                    f.write(f"  聚类 {stats['cluster_id']}: {stats['count']:,} 点 ({stats['percentage']:.2f}%)\n")
                f.write(f"  总计: {total_points:,} 点\n\n")
            else:
                f.write(f"{state}状态: 分析失败\n\n")
        
        # 生成状态间比较
        f.write("状态间聚类质量比较:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'状态':<8} {'轮廓系数':<12} {'簇内平方和':<15} {'分析时间(s)':<12}\n")
        f.write("-" * 50 + "\n")
        
        for state in states:
            if state in all_results:
                result = all_results[state]
                f.write(f"{state:<8} {result['silhouette_score']:<12.4f} {result['inertia']:<15.2f} {result['analysis_time']:<12.2f}\n")
            else:
                f.write(f"{state:<8} {'失败':<12} {'失败':<15} {'失败':<12}\n")
    
    print(f"已保存总结报告：{summary_file}")

if __name__ == "__main__":
    main() 