#!/usr/bin/env python3
"""
比较不同类型轨迹中Rotation和Straight的比例分布
========================================================

统计WT、Thermo、FAST、HOT四种轨迹类型中：
1. 所有Rotation类型的比例（Rotate_gradual + Rotate_hopping）
2. 所有Straight类型的比例（Straight_right + Straight_wrong）

并绘制折线图进行比较。

Requirements:
    pip install pandas matplotlib seaborn numpy
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# 定义轨迹类型映射
TRAJECTORY_TYPES = ['WT-EL','WT', 'Thermo', 'FAST', 'HOT']

# 定义聚类标签映射
CLUSTER_MAPPING = {
    0: 'Rotate_gradual',
    1: 'Rotate_hopping', 
    2: 'Straight_right',
    3: 'Straight_wrong'
}

# 定义分组
ROTATION_CLUSTERS = [0, 1]  # Rotate_gradual + Rotate_hopping
STRAIGHT_CLUSTERS = [2, 3]  # Straight_right + Straight_wrong


def validate_trajectory_assignments(analysis_root, trajectory_types):
    """验证手动修正的轨迹分配是否完整且无重复
    
    检查每个轨迹类型的correction文件中的assigned_runs是否：
    1. 包含所有实际存在的轨迹（不能少）
    2. 不包含不存在的轨迹（不能多）
    3. 没有重复分配的轨迹
    
    如果验证失败，报错退出程序。
    """
    print("\n" + "="*70)
    print("VALIDATION: Checking trajectory assignment integrity...")
    print("="*70)
    
    validation_passed = True
    all_issues = []
    
    for traj_type in trajectory_types:
        print(f"\nValidating {traj_type}...")
        
        # 读取correction文件
        correction_file = os.path.join(
            analysis_root, traj_type, 'ps_spatial_clustering_representatives',
            'ps_spatial_cluster_summary_correction.csv'
        )
        
        # 获取实际存在的轨迹文件
        projection_dir = os.path.join(
            analysis_root, traj_type, 'each_run_projection_trj', 'ps-only'
        )
        
        if not os.path.exists(correction_file):
            issue = f"  ❌ ERROR: Correction file not found: {correction_file}"
            print(issue)
            all_issues.append(f"{traj_type}: {issue}")
            validation_passed = False
            continue
            
        if not os.path.exists(projection_dir):
            issue = f"  ❌ ERROR: Projection directory not found: {projection_dir}"
            print(issue)
            all_issues.append(f"{traj_type}: {issue}")
            validation_passed = False
            continue
        
        # 读取correction文件中的assigned_runs
        correction_df = pd.read_csv(correction_file)
        assigned_runs = set()
        
        for _, row in correction_df.iterrows():
            runs_str = str(row['assigned_runs']).strip()
            if runs_str and runs_str != 'nan' and runs_str != '':
                runs = [int(r.strip()) for r in runs_str.split(',') if r.strip()]
                assigned_runs.update(runs)
        
        # 获取实际存在的轨迹编号
        actual_runs = set()
        for filename in os.listdir(projection_dir):
            if filename.startswith('run-') and filename.endswith('.png'):
                run_num = int(filename.replace('run-', '').replace('.png', ''))
                actual_runs.add(run_num)
        
        # 检查重复
        all_assigned = []
        for _, row in correction_df.iterrows():
            runs_str = str(row['assigned_runs']).strip()
            if runs_str and runs_str != 'nan' and runs_str != '':
                runs = [int(r.strip()) for r in runs_str.split(',') if r.strip()]
                all_assigned.extend(runs)
        
        duplicates = [run for run in set(all_assigned) if all_assigned.count(run) > 1]
        
        # 检查缺失和多余
        missing_runs = actual_runs - assigned_runs
        extra_runs = assigned_runs - actual_runs
        
        # 报告结果
        has_issues = False
        
        if duplicates:
            issue = f"  ❌ DUPLICATE ASSIGNMENTS: {sorted(duplicates)}"
            print(issue)
            all_issues.append(f"{traj_type}: Duplicate runs {sorted(duplicates)}")
            validation_passed = False
            has_issues = True
        
        if missing_runs:
            issue = f"  ❌ MISSING RUNS: {sorted(missing_runs)}"
            print(issue)
            all_issues.append(f"{traj_type}: Missing runs {sorted(missing_runs)}")
            validation_passed = False
            has_issues = True
        
        if extra_runs:
            issue = f"  ❌ EXTRA RUNS (not in actual data): {sorted(extra_runs)}"
            print(issue)
            all_issues.append(f"{traj_type}: Extra runs {sorted(extra_runs)}")
            validation_passed = False
            has_issues = True
        
        if not has_issues:
            print(f"  ✓ PASSED: {len(assigned_runs)} runs correctly assigned")
            print(f"    - No duplicates")
            print(f"    - No missing runs")
            print(f"    - No extra runs")
        else:
            print(f"  Summary: {len(actual_runs)} actual runs, {len(assigned_runs)} assigned runs")
    
    print("\n" + "="*70)
    if validation_passed:
        print("✓ VALIDATION PASSED: All trajectory assignments are valid!")
        print("="*70 + "\n")
        return True
    else:
        print("❌ VALIDATION FAILED: Issues found in trajectory assignments!")
        print("="*70)
        print("\nIssues summary:")
        for issue in all_issues:
            print(f"  - {issue}")
        print("\n" + "="*70)
        print("Please fix the issues in the correction files before proceeding.")
        print("="*70 + "\n")
        return False


def load_clustering_results(analysis_root, trajectory_types, use_corrected=False):
    """加载所有轨迹类型的聚类结果，从轨迹分配文件重新计算百分比
    
    重要：不使用summary文件中的percentage字段，因为手动修改轨迹分配后该值会过时。
    而是从trajectory_assignments文件读取每条轨迹的实际分配，重新计算真实百分比。
    """
    results = {}
    
    for traj_type in trajectory_types:
        if use_corrected:
            # 使用修正后的轨迹分配结果
            assignment_file = os.path.join(
                analysis_root, traj_type, 'ps_spatial_clustering_representatives',
                 'ps_spatial_cluster_summary_correction.csv'
            )
            data_source = "corrected"
        else:
            # 使用原始轨迹分配结果
            assignment_file = os.path.join(
                analysis_root, traj_type, 'ps_spatial_clustering_representatives',
                'ps_spatial_trajectory_assignments.csv'
            )
            data_source = "original"
        
        if os.path.exists(assignment_file):
            if use_corrected:
                # correction文件格式：cluster_id, representative_label, assigned_runs (逗号分隔)
                correction_df = pd.read_csv(assignment_file)
                
                # 解析assigned_runs并统计每个cluster的轨迹数
                cluster_data = []
                total_trajectories = 0
                
                for _, row in correction_df.iterrows():
                    runs_str = str(row['assigned_runs']).strip()
                    if runs_str and runs_str != 'nan' and runs_str != '':
                        runs = [int(r.strip()) for r in runs_str.split(',') if r.strip()]
                        count = len(runs)
                    else:
                        count = 0
                    
                    total_trajectories += count
                    cluster_data.append({
                        'cluster_id': int(row['cluster_id']),
                        'representative_label': row['representative_label'],
                        'count': count
                    })
                
                df = pd.DataFrame(cluster_data)
                
                # 计算真实百分比
                if total_trajectories > 0:
                    df['percentage'] = (df['count'] / total_trajectories * 100)
                else:
                    df['percentage'] = 0.0
                    
            else:
                # 原始文件格式：每行一个轨迹
                assignments_df = pd.read_csv(assignment_file)
                
                # 统计每个cluster的轨迹数量
                cluster_counts = assignments_df.groupby('assigned_cluster').size().reset_index(name='count')
                
                # 计算总轨迹数
                total_trajectories = len(assignments_df)
                
                # 计算真实百分比
                cluster_counts['percentage'] = (cluster_counts['count'] / total_trajectories * 100)
                
                # 获取代表性标签
                cluster_labels = assignments_df.groupby('assigned_cluster')['representative_label'].first().reset_index()
                
                # 合并数据
                df = pd.merge(cluster_counts, cluster_labels, on='assigned_cluster')
                df = df.rename(columns={'assigned_cluster': 'cluster_id'})
            
            # 确保所有4个cluster都存在（即使某些cluster没有轨迹）
            all_clusters = pd.DataFrame({
                'cluster_id': [0, 1, 2, 3],
                'representative_label': ['Rotate_gradual', 'Rotate_hopping', 'Straight_right', 'Straight_wrong']
            })
            df = pd.merge(all_clusters, df, on=['cluster_id', 'representative_label'], how='left')
            df['count'] = df['count'].fillna(0).astype(int)
            df['percentage'] = df['percentage'].fillna(0.0)
            
            results[traj_type] = df
            print(f"Loaded {traj_type} ({data_source}): {total_trajectories} trajectories, recalculated percentages from assignments")
        else:
            print(f"Warning: {data_source.capitalize()} assignment file not found for {traj_type}: {assignment_file}")
            # 创建空的DataFrame以保持一致性
            results[traj_type] = pd.DataFrame({
                'cluster_id': [0, 1, 2, 3],
                'representative_label': ['Rotate_gradual', 'Rotate_hopping', 'Straight_right', 'Straight_wrong'],
                'percentage': [0.0, 0.0, 0.0, 0.0],
                'count': [0, 0, 0, 0]
            })
    
    return results


def calculate_proportions(results):
    """计算每种轨迹类型中Rotation和Straight的比例"""
    proportions = {
        'trajectory_type': [],
        'rotation_percentage': [],
        'straight_percentage': [],
        'rotation_count': [],
        'straight_count': [],
        'total_count': []
    }
    
    for traj_type, df in results.items():
        proportions['trajectory_type'].append(traj_type)
        
        # 计算Rotation比例（cluster 0 + cluster 1）
        rotation_percentage = df[df['cluster_id'].isin(ROTATION_CLUSTERS)]['percentage'].sum()
        rotation_count = df[df['cluster_id'].isin(ROTATION_CLUSTERS)]['count'].sum()
        
        # 计算Straight比例（cluster 2 + cluster 3）
        straight_percentage = df[df['cluster_id'].isin(STRAIGHT_CLUSTERS)]['percentage'].sum()
        straight_count = df[df['cluster_id'].isin(STRAIGHT_CLUSTERS)]['count'].sum()
        
        # 总轨迹数
        total_count = df['count'].sum()
        
        proportions['rotation_percentage'].append(rotation_percentage)
        proportions['straight_percentage'].append(straight_percentage)
        proportions['rotation_count'].append(rotation_count)
        proportions['straight_count'].append(straight_count)
        proportions['total_count'].append(total_count)
        
        print(f"{traj_type}:")
        print(f"  Rotation: {rotation_percentage:.1f}% ({rotation_count} trajectories)")
        print(f"  Straight: {straight_percentage:.1f}% ({straight_count} trajectories)")
        print(f"  Total: {total_count} trajectories")
        print()
    
    return pd.DataFrame(proportions)


def plot_rotation_straight_comparison(proportions_df, output_dir, is_corrected=False):
    """绘制Rotation和Straight比例的折线图比较"""
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_positions = np.arange(len(proportions_df))
    trajectory_types = proportions_df['trajectory_type'].tolist()
    
    # 绘制折线图
    line1 = ax.plot(x_positions, proportions_df['rotation_percentage'], 
                    marker='o', linewidth=3, markersize=8, 
                    color='#2E86AB', label='Rotation (Gradual + Hopping)', 
                    markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2E86AB')
    
    line2 = ax.plot(x_positions, proportions_df['straight_percentage'], 
                    marker='s', linewidth=3, markersize=8, 
                    color='#A23B72', label='Straight (Right + Wrong)', 
                    markerfacecolor='white', markeredgewidth=2, markeredgecolor='#A23B72')
    
    # 在数据点上添加百分比标签
    for i, (rot_pct, str_pct) in enumerate(zip(proportions_df['rotation_percentage'], 
                                               proportions_df['straight_percentage'])):
        # Rotation标签
        ax.annotate(f'{rot_pct:.1f}%', 
                   xy=(i, rot_pct), xytext=(0, 15), 
                   textcoords='offset points', ha='center', va='bottom',
                   fontsize=11, fontweight='bold', color='#2E86AB',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='#2E86AB', alpha=0.8))
        
        # Straight标签  
        ax.annotate(f'{str_pct:.1f}%', 
                   xy=(i, str_pct), xytext=(0, -20), 
                   textcoords='offset points', ha='center', va='top',
                   fontsize=11, fontweight='bold', color='#A23B72',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='#A23B72', alpha=0.8))
    
    # 设置坐标轴
    ax.set_xticks(x_positions)
    ax.set_xticklabels(trajectory_types, fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of Trajectories (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trajectory Type', fontsize=14, fontweight='bold')
    
    # 设置y轴范围
    max_percentage = max(proportions_df['rotation_percentage'].max(), 
                        proportions_df['straight_percentage'].max())
    ax.set_ylim(0, max_percentage * 1.15)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置标题
    title_suffix = " (Manually Corrected)" if is_corrected else ""
    ax.set_title(f'Comparison of Rotation vs Straight Trajectory Proportions\nAcross Different Trajectory Types{title_suffix}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=12, frameon=True, 
             fancybox=True, shadow=True, framealpha=0.9)
    
    # 美化图形
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    # 保存图片
    filename = 'rotation_straight_proportion_comparison_corrected.png' if is_corrected else 'rotation_straight_proportion_comparison.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot: {output_path}")
    
    return output_path


def plot_rotation_breakdown(results, output_dir, is_corrected=False):
    """绘制Rotate_gradual和Rotate_hopping在Rotation总数中的占比折线图"""
    
    # 准备数据
    rotation_breakdown = {
        'trajectory_type': [],
        'gradual_percentage': [],
        'hopping_percentage': [],
        'gradual_count': [],
        'hopping_count': [],
        'total_rotation_count': []
    }
    
    for traj_type, df in results.items():
        rotation_breakdown['trajectory_type'].append(traj_type)
        
        # 获取Rotate_gradual和Rotate_hopping的数据
        gradual_count = df[df['cluster_id'] == 0]['count'].iloc[0] if len(df[df['cluster_id'] == 0]) > 0 else 0
        hopping_count = df[df['cluster_id'] == 1]['count'].iloc[0] if len(df[df['cluster_id'] == 1]) > 0 else 0
        
        total_rotation = gradual_count + hopping_count
        
        # 计算占比
        gradual_pct = (gradual_count / total_rotation * 100) if total_rotation > 0 else 0
        hopping_pct = (hopping_count / total_rotation * 100) if total_rotation > 0 else 0
        
        rotation_breakdown['gradual_count'].append(gradual_count)
        rotation_breakdown['hopping_count'].append(hopping_count)
        rotation_breakdown['total_rotation_count'].append(total_rotation)
        rotation_breakdown['gradual_percentage'].append(gradual_pct)
        rotation_breakdown['hopping_percentage'].append(hopping_pct)
        
        print(f"{traj_type} Rotation breakdown:")
        print(f"  Rotate_gradual: {gradual_pct:.1f}% ({gradual_count}/{total_rotation})")
        print(f"  Rotate_hopping: {hopping_pct:.1f}% ({hopping_count}/{total_rotation})")
        print()
    
    breakdown_df = pd.DataFrame(rotation_breakdown)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_positions = np.arange(len(breakdown_df))
    trajectory_types = breakdown_df['trajectory_type'].tolist()
    
    # 绘制折线图
    line1 = ax.plot(x_positions, breakdown_df['gradual_percentage'], 
                    marker='o', linewidth=3, markersize=10, 
                    color='#1E88E5', label='Rotate_gradual', 
                    markerfacecolor='white', markeredgewidth=2.5, markeredgecolor='#1E88E5')
    
    line2 = ax.plot(x_positions, breakdown_df['hopping_percentage'], 
                    marker='D', linewidth=3, markersize=10, 
                    color='#FFA726', label='Rotate_hopping', 
                    markerfacecolor='white', markeredgewidth=2.5, markeredgecolor='#FFA726')
    
    # 在数据点上添加百分比标签和计数
    for i, (grad_pct, hop_pct, grad_cnt, hop_cnt, total) in enumerate(
            zip(breakdown_df['gradual_percentage'], 
                breakdown_df['hopping_percentage'],
                breakdown_df['gradual_count'],
                breakdown_df['hopping_count'],
                breakdown_df['total_rotation_count'])):
        
        # Rotate_gradual标签
        ax.annotate(f'{grad_pct:.1f}%\n({grad_cnt}/{total})', 
                   xy=(i, grad_pct), xytext=(0, 15), 
                   textcoords='offset points', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='#1E88E5',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor='#1E88E5', alpha=0.9, linewidth=1.5))
        
        # Rotate_hopping标签  
        ax.annotate(f'{hop_pct:.1f}%\n({hop_cnt}/{total})', 
                   xy=(i, hop_pct), xytext=(0, -25), 
                   textcoords='offset points', ha='center', va='top',
                   fontsize=10, fontweight='bold', color='#FFA726',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor='#FFA726', alpha=0.9, linewidth=1.5))
    
    # 设置坐标轴
    ax.set_xticks(x_positions)
    ax.set_xticklabels(trajectory_types, fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage within Rotation Trajectories (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trajectory Type', fontsize=14, fontweight='bold')
    
    # 设置y轴范围为0-100%
    ax.set_ylim(0, 105)
    
    # 添加水平参考线
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% Reference')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置标题
    title_suffix = " (Manually Corrected)" if is_corrected else ""
    ax.set_title(f'Breakdown of Rotation Trajectory Types\n(Rotate_gradual vs Rotate_hopping){title_suffix}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 添加图例
    ax.legend(loc='best', fontsize=12, frameon=True, 
             fancybox=True, shadow=True, framealpha=0.9)
    
    # 美化图形
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    # 保存图片
    filename = 'rotation_breakdown_corrected.png' if is_corrected else 'rotation_breakdown.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved rotation breakdown plot: {output_path}")
    
    # 保存数据
    csv_filename = 'rotation_breakdown_corrected.csv' if is_corrected else 'rotation_breakdown.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    breakdown_df.to_csv(csv_path, index=False)
    print(f"Saved rotation breakdown data: {csv_path}")
    
    return output_path, breakdown_df


def plot_detailed_breakdown(proportions_df, results, output_dir, is_corrected=False):
    """绘制详细的分类比例堆叠柱状图"""
    
    # 准备详细数据
    detailed_data = {
        'trajectory_type': [],
        'Rotate_gradual': [],
        'Rotate_hopping': [],
        'Straight_right': [],
        'Straight_wrong': []
    }
    
    for traj_type in proportions_df['trajectory_type']:
        df = results[traj_type]
        detailed_data['trajectory_type'].append(traj_type)
        
        for cluster_id, label in CLUSTER_MAPPING.items():
            cluster_row = df[df['cluster_id'] == cluster_id]
            percentage = cluster_row['percentage'].iloc[0] if len(cluster_row) > 0 else 0.0
            detailed_data[label].append(percentage)
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # 创建堆叠柱状图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_positions = np.arange(len(detailed_df))
    
    # 定义颜色
    colors = {
        'Rotate_gradual': '#1E88E5',    # 蓝色
        'Rotate_hopping': '#42A5F5',   # 浅蓝色
        'Straight_right': '#E53935',   # 红色
        'Straight_wrong': '#FF5722'    # 橙红色
    }
    
    # 绘制堆叠柱状图
    bottom = np.zeros(len(detailed_df))
    
    for category in ['Rotate_gradual', 'Rotate_hopping', 'Straight_right', 'Straight_wrong']:
        values = detailed_df[category].values
        bars = ax.bar(x_positions, values, bottom=bottom, 
                     color=colors[category], label=category, 
                     alpha=0.8, edgecolor='white', linewidth=1)
        
        # 在非零部分添加百分比标签
        for i, (val, bot) in enumerate(zip(values, bottom)):
            if val > 2:  # 只在大于2%的部分显示标签
                ax.text(i, bot + val/2, f'{val:.1f}%', 
                       ha='center', va='center', fontweight='bold', 
                       color='white', fontsize=10)
        
        bottom += values
    
    # 设置坐标轴
    ax.set_xticks(x_positions)
    ax.set_xticklabels(detailed_df['trajectory_type'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage of Trajectories (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trajectory Type', fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 设置标题
    title_suffix = " (Manually Corrected)" if is_corrected else ""
    ax.set_title(f'Detailed Breakdown of Trajectory Types\n(Stacked Bar Chart){title_suffix}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=11, frameon=True, 
             fancybox=True, shadow=True, framealpha=0.9)
    
    # 美化图形
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    filename = 'detailed_trajectory_breakdown_corrected.png' if is_corrected else 'detailed_trajectory_breakdown.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved detailed breakdown plot: {output_path}")
    
    return output_path


def save_summary_results(proportions_df, results, output_dir, is_corrected=False):
    """保存汇总结果到文件"""
    
    # 保存比例汇总
    filename_suffix = "_corrected" if is_corrected else ""
    summary_file = os.path.join(output_dir, f'rotation_straight_summary{filename_suffix}.csv')
    proportions_df.to_csv(summary_file, index=False)
    
    # 保存详细分类结果
    detailed_results = []
    for traj_type in proportions_df['trajectory_type']:
        df = results[traj_type]
        for _, row in df.iterrows():
            detailed_results.append({
                'trajectory_type': traj_type,
                'cluster_id': row['cluster_id'],
                'cluster_label': row['representative_label'],
                'percentage': row['percentage'],
                'count': row['count'],
                'category': 'Rotation' if row['cluster_id'] in ROTATION_CLUSTERS else 'Straight'
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_file = os.path.join(output_dir, f'detailed_cluster_results{filename_suffix}.csv')
    detailed_df.to_csv(detailed_file, index=False)
    
    # 创建文本摘要
    title_suffix = " (CORRECTED)" if is_corrected else ""
    data_source = "Manually corrected clustering" if is_corrected else "Original clustering"
    
    summary_text = f"""Rotation vs Straight Trajectory Proportion Analysis{title_suffix}
==================================================

Data Source: {data_source}

Summary Across All Trajectory Types:
"""
    
    for _, row in proportions_df.iterrows():
        traj_type = row['trajectory_type']
        summary_text += f"""
{traj_type}:
  Rotation (Gradual + Hopping): {row['rotation_percentage']:.1f}% ({row['rotation_count']} trajectories)
  Straight (Right + Wrong): {row['straight_percentage']:.1f}% ({row['straight_count']} trajectories)
  Total trajectories: {row['total_count']}
"""

    # 计算总体统计
    total_rotation = proportions_df['rotation_count'].sum()
    total_straight = proportions_df['straight_count'].sum()
    grand_total = proportions_df['total_count'].sum()
    
    summary_text += f"""
Overall Statistics:
  Total Rotation trajectories: {total_rotation} ({total_rotation/grand_total*100:.1f}%)
  Total Straight trajectories: {total_straight} ({total_straight/grand_total*100:.1f}%)
  Grand total trajectories: {grand_total}

Cluster Definitions:
  Rotation = Rotate_gradual (Cluster 0) + Rotate_hopping (Cluster 1)
  Straight = Straight_right (Cluster 2) + Straight_wrong (Cluster 3)
"""
    
    text_file = os.path.join(output_dir, f'ROTATION_STRAIGHT_ANALYSIS_SUMMARY{filename_suffix.upper()}.txt')
    with open(text_file, 'w') as f:
        f.write(summary_text)
    
    print("Summary results saved:")
    print(f"  - Proportion summary: {summary_file}")
    print(f"  - Detailed results: {detailed_file}")
    print(f"  - Text summary: {text_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare Rotation vs Straight proportions across trajectory types")
    parser.add_argument('--analysis-root', type=str, 
                       default='/M-run-complex-FAST-10/analysis_contact_PCA',
                       help='Root directory of the analysis results')
    parser.add_argument('--output-dir', type=str, 
                       default=None,
                       help='/M-run-complex-FAST-10/analysis_contact_PCA/rotation_straight_comparison)')
    parser.add_argument('--trajectory-types', nargs='+', 
                       default=['WT-EL','WT', 'Thermo', 'FAST', 'HOT'],
                       help='List of trajectory types to compare')
    parser.add_argument('--correct', action='store_true',default=True,
                       help='Use manually corrected clustering results from correction directories')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        if args.correct:
            output_dir = os.path.join(args.analysis_root, 'rotation_straight_comparison_corrected')
        else:
            output_dir = os.path.join(args.analysis_root, 'rotation_straight_comparison')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    if args.correct:
        print("Rotation vs Straight Proportion Comparison Analysis (CORRECTED)")
        print("=" * 70)
        print("Using manually corrected clustering results")
    else:
        print("Rotation vs Straight Proportion Comparison Analysis")
        print("=" * 60)
        print("Using original clustering results")
    
    print(f"Analysis root: {args.analysis_root}")
    print(f"Output directory: {output_dir}")
    print(f"Trajectory types: {args.trajectory_types}")
    print()
    
    # 验证轨迹分配的完整性（仅在使用corrected数据时）
    if args.correct:
        print("Running validation checks on corrected assignments...")
        if not validate_trajectory_assignments(args.analysis_root, args.trajectory_types):
            print("\n" + "!"*70)
            print("CRITICAL ERROR: Validation failed!")
            print("Program terminated to prevent using invalid data.")
            print("!"*70)
            return
    
    # 加载聚类结果
    if args.correct:
        print("Loading corrected clustering results...")
    else:
        print("Loading original clustering results...")
    results = load_clustering_results(args.analysis_root, args.trajectory_types, use_corrected=args.correct)
    
    if not results:
        print("No clustering results found!")
        return
    
    # 计算比例
    print("Calculating proportions...")
    proportions_df = calculate_proportions(results)
    
    # 绘制比较图
    print("Creating comparison plots...")
    plot_rotation_straight_comparison(proportions_df, output_dir, is_corrected=args.correct)
    plot_detailed_breakdown(proportions_df, results, output_dir, is_corrected=args.correct)
    
    # 绘制Rotation类型的细分占比
    print("Creating rotation breakdown plot...")
    plot_rotation_breakdown(results, output_dir, is_corrected=args.correct)
    
    # 保存结果
    print("Saving summary results...")
    save_summary_results(proportions_df, results, output_dir, is_corrected=args.correct)
    
    print("\nAnalysis completed!")
    print(f"Results saved to: {output_dir}")
    
    # 显示最终汇总
    print("\nFinal Summary:")
    print("-" * 40)
    for _, row in proportions_df.iterrows():
        print(f"{row['trajectory_type']:8} | Rotation: {row['rotation_percentage']:5.1f}% | Straight: {row['straight_percentage']:5.1f}%")


if __name__ == "__main__":
    main()
