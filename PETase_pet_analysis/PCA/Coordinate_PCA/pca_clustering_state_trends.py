#!/usr/bin/env python3
"""
WT四个状态cluster比例变化趋势分析脚本

读取四个状态（Apo, ES, DS, PS）的聚类结果，绘制cluster比例随状态变化的趋势图

功能包括：
1. 单个cluster比例变化趋势折线图
2. 堆叠柱状图显示每个状态的cluster分布
3. 分组趋势分析：
   - 分组类型1: Cluster 1+2 vs Cluster 3+4
   - 分组类型2: Cluster 1+3 vs Cluster 2+4
4. 生成详细的数据表格和分析报告
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse


def extract_cluster_percentages(stats_file_path):
    """从统计文件中提取cluster比例信息"""
    
    cluster_percentages = {}
    
    try:
        with open(stats_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式提取cluster比例信息
        # 匹配格式：聚类 1:\n  数据点数量: 139,435 (26.08%)
        pattern = r'聚类 (\d+):\s*\n\s*数据点数量: [\d,]+ \((\d+\.?\d*)%\)'
        matches = re.findall(pattern, content)
        
        for cluster_id, percentage in matches:
            cluster_percentages[int(cluster_id)] = float(percentage)
        
        print(f"成功提取 {stats_file_path.name} 的cluster比例:")
        for cluster_id in sorted(cluster_percentages.keys()):
            print(f"  Cluster {cluster_id}: {cluster_percentages[cluster_id]:.2f}%")
        
        return cluster_percentages
        
    except Exception as e:
        print(f"读取文件 {stats_file_path} 时出错: {str(e)}")
        return {}


def collect_all_state_data(base_dir, type_id='WT', states=['Apo', 'ES', 'DS', 'PS']):
    """收集所有状态的cluster比例数据"""
    
    all_data = {}
    base_path = Path(base_dir)
    
    for state in states:
        state_dir = base_path / f"{type_id}_{state}"
        stats_file = state_dir / f"clustering_statistics_{state}.txt"
        
        if stats_file.exists():
            cluster_percentages = extract_cluster_percentages(stats_file)
            if cluster_percentages:
                all_data[state] = cluster_percentages
            else:
                print(f"警告: 无法从 {state} 状态提取cluster比例数据")
        else:
            print(f"警告: 找不到 {state} 状态的统计文件: {stats_file}")
    
    return all_data


def create_trend_plot(all_data, output_dir, type_id='WT'):
    """创建cluster比例变化趋势图"""
    
    states = list(all_data.keys())
    if not states:
        print("错误: 没有可用的数据")
        return
    
    # 将Apo替换为US
    states_display = ['US' if state == 'Apo' else state for state in states]
    
    # 获取cluster数量（假设所有状态都有相同数量的cluster）
    first_state = states[0]
    n_clusters = len(all_data[first_state])
    
    # 准备数据
    cluster_data = {}
    for cluster_id in range(1, n_clusters + 1):
        cluster_data[cluster_id] = []
        for state in states:
            if cluster_id in all_data[state]:
                cluster_data[cluster_id].append(all_data[state][cluster_id])
            else:
                cluster_data[cluster_id].append(0)  # 如果缺失数据，设为0
    
    # 创建图形 - 总图8*6英寸，左侧绘图区6*6，右侧图例区2*6
    fig = plt.figure(figsize=(8, 6), dpi=150)
    
    # 设置字体为Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 26
    
    # 创建主绘图区域(左侧6*6的区域，相当于总宽度的75%)
    ax = fig.add_axes([0.12, 0.15, 0.60, 0.75])  # [left, bottom, width, height]
    
    # 定义Cluster 1-4的颜色
    colors = ['#CFEAF1', '#A1A9D0', '#9E9E9E', '#C4A5DE']
    markers = ['o', 's', '^', 'D']
    
    # 为每个cluster绘制折线
    for cluster_id in range(1, n_clusters + 1):
        percentages = cluster_data[cluster_id]
        ax.plot(states_display, percentages, 
                marker=markers[cluster_id-1], 
                color=colors[cluster_id-1],
                linewidth=2.5, 
                markersize=12,
                label=str(cluster_id),  # 使用普通数字1-4
                markerfacecolor=colors[cluster_id-1],
                markeredgewidth=2.5,
                markeredgecolor=colors[cluster_id-1])
    
    # 设置图形属性 - 去掉标题，去掉grid
    ax.set_xlabel('States', fontsize=26, fontfamily='Arial')
    ax.set_ylabel('Population (%)', fontsize=26, fontfamily='Arial')
    
    # 只保留左侧和下面的坐标轴，去掉右侧和上边
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # 设置坐标轴线条宽度，使其与图例边框一致
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # 设置y轴范围
    all_percentages = [p for cluster_percentages in cluster_data.values() for p in cluster_percentages]
    y_min = max(0, min(all_percentages) - 5)
    y_max = min(100, max(all_percentages) + 5)
    ax.set_ylim(y_min, y_max)
    
    # 设置刻度标签字体
    ax.tick_params(axis='both', labelsize=26, width=1.0)  # 设置刻度线宽度
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')
    
    # 在右侧添加图例(占据右侧2*6的区域)
    legend = ax.legend(fontsize=26, loc='center left', bbox_to_anchor=(1.05, 0.5), prop={'family': 'Arial', 'size': 26}, frameon=False)

    # 保存图形
    output_path = Path(output_dir) / f"{type_id}_cluster_percentage_trends.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"已保存趋势图: {output_path}")
    
    return output_path


def create_stacked_bar_chart(all_data, output_dir, type_id='WT'):
    """创建堆叠柱状图显示每个状态的cluster分布"""
    
    states = list(all_data.keys())
    if not states:
        return
    
    # 获取cluster数量
    first_state = states[0]
    n_clusters = len(all_data[first_state])
    
    # 准备数据
    cluster_data = {}
    for cluster_id in range(1, n_clusters + 1):
        cluster_data[f'Cluster {cluster_id}'] = []
        for state in states:
            if cluster_id in all_data[state]:
                cluster_data[f'Cluster {cluster_id}'].append(all_data[state][cluster_id])
            else:
                cluster_data[f'Cluster {cluster_id}'].append(0)
    
    # 创建DataFrame
    df = pd.DataFrame(cluster_data, index=states)
    
    # 创建堆叠柱状图
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.6)
    
    # 设置图形属性
    ax.set_xlabel('States', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'{type_id} - Cluster Distribution Across States', fontsize=16, fontweight='bold')
    ax.legend(title='Clusters', fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(states, rotation=0, fontsize=12)
    ax.set_ylim(0, 100)
    
    # 在每个柱子上添加百分比标签
    for i, state in enumerate(states):
        y_offset = 0
        for cluster_id in range(1, n_clusters + 1):
            percentage = all_data[state].get(cluster_id, 0)
            if percentage > 3:  # 只显示大于3%的标签
                ax.text(i, y_offset + percentage/2, f'{percentage:.1f}%', 
                       ha='center', va='center', fontsize=9, fontweight='bold')
            y_offset += percentage
    
    plt.tight_layout()
    
    # 保存图形
    output_path = Path(output_dir) / f"{type_id}_cluster_stacked_distribution.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"已保存堆叠柱状图: {output_path}")
    
    return output_path


def create_data_table(all_data, output_dir, type_id='WT'):
    """创建数据表格并保存为CSV文件"""
    
    states = list(all_data.keys())
    if not states:
        return
    
    # 获取cluster数量
    first_state = states[0]
    n_clusters = len(all_data[first_state])
    
    # 创建DataFrame
    data_dict = {}
    for state in states:
        data_dict[state] = []
        for cluster_id in range(1, n_clusters + 1):
            data_dict[state].append(all_data[state].get(cluster_id, 0))
    
    df = pd.DataFrame(data_dict, index=[f'Cluster {i}' for i in range(1, n_clusters + 1)])
    
    # 保存CSV文件
    csv_path = Path(output_dir) / f"{type_id}_cluster_percentages.csv"
    df.to_csv(csv_path)
    
    print(f"已保存数据表格: {csv_path}")
    print("\nCluster比例数据表:")
    print(df.round(2))
    
    return csv_path


def create_grouped_trend_plot_type1(all_data, output_dir, type_id='WT'):
    """创建分组趋势图: Group A (Cluster 1+2) vs Group B (Cluster 3+4)"""
    
    states = list(all_data.keys())
    if not states:
        print("错误: 没有可用的数据")
        return
    
    # 计算分组数据
    group_data = {
        'Group A (C1+C2)': [],
        'Group B (C3+C4)': []
    }
    
    for state in states:
        # Group A: Cluster 1 + Cluster 2
        group_a_percentage = all_data[state].get(1, 0) + all_data[state].get(2, 0)
        group_data['Group A (C1+C2)'].append(group_a_percentage)
        
        # Group B: Cluster 3 + Cluster 4
        group_b_percentage = all_data[state].get(3, 0) + all_data[state].get(4, 0)
        group_data['Group B (C3+C4)'].append(group_b_percentage)
    
    # 创建图形
    plt.figure(figsize=(12, 8), dpi=150)
    
    # 设置颜色
    colors = ['#FF4444', '#4444FF']  # 红色和蓝色
    markers = ['o', 's']
    
    # 绘制分组折线
    for i, (group_name, percentages) in enumerate(group_data.items()):
        plt.plot(states, percentages, 
                marker=markers[i], 
                color=colors[i],
                linewidth=3, 
                markersize=10,
                label=group_name,
                markerfacecolor='white',
                markeredgewidth=3,
                markeredgecolor=colors[i])
        
        # 在每个点上添加数值标签
        for j, (state, percentage) in enumerate(zip(states, percentages)):
            plt.annotate(f'{percentage:.1f}%', 
                        (j, percentage), 
                        textcoords="offset points", 
                        xytext=(0, 15), 
                        ha='center',
                        fontsize=11,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', 
                                alpha=0.9,
                                edgecolor=colors[i]))
    
    # 设置图形属性
    plt.xlabel('States', fontsize=14, fontweight='bold')
    plt.ylabel('Group Percentage (%)', fontsize=14, fontweight='bold')
    plt.title(f'{type_id} - Grouped Cluster Trends: (C1+C2) vs (C3+C4)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=13, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置y轴范围
    all_percentages = [p for percentages in group_data.values() for p in percentages]
    y_min = max(0, min(all_percentages) - 5)
    y_max = min(100, max(all_percentages) + 5)
    plt.ylim(y_min, y_max)
    
    # 美化图形
    plt.tight_layout()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 保存图形
    output_path = Path(output_dir) / f"{type_id}_grouped_trends_type1_C12_vs_C34.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"已保存分组趋势图(类型1): {output_path}")
    
    # 打印分组数据
    print("\n分组趋势数据 (C1+C2 vs C3+C4):")
    for state in states:
        group_a = group_data['Group A (C1+C2)'][states.index(state)]
        group_b = group_data['Group B (C3+C4)'][states.index(state)]
        print(f"  {state}: Group A (C1+C2) = {group_a:.2f}%, Group B (C3+C4) = {group_b:.2f}%")
    
    return output_path, group_data


def create_grouped_trend_plot_type2(all_data, output_dir, type_id='WT'):
    """创建分组趋势图: Group X (Cluster 1+3) vs Group Y (Cluster 2+4)"""
    
    states = list(all_data.keys())
    if not states:
        print("错误: 没有可用的数据")
        return
    
    # 计算分组数据
    group_data = {
        'Group X (C1+C3)': [],
        'Group Y (C2+C4)': []
    }
    
    for state in states:
        # Group X: Cluster 1 + Cluster 3
        group_x_percentage = all_data[state].get(1, 0) + all_data[state].get(3, 0)
        group_data['Group X (C1+C3)'].append(group_x_percentage)
        
        # Group Y: Cluster 2 + Cluster 4
        group_y_percentage = all_data[state].get(2, 0) + all_data[state].get(4, 0)
        group_data['Group Y (C2+C4)'].append(group_y_percentage)
    
    # 创建图形
    plt.figure(figsize=(12, 8), dpi=150)
    
    # 设置颜色
    colors = ['#FF8C00', '#32CD32']  # 橙色和绿色
    markers = ['^', 'D']
    
    # 绘制分组折线
    for i, (group_name, percentages) in enumerate(group_data.items()):
        plt.plot(states, percentages, 
                marker=markers[i], 
                color=colors[i],
                linewidth=3, 
                markersize=10,
                label=group_name,
                markerfacecolor='white',
                markeredgewidth=3,
                markeredgecolor=colors[i])
        
        # 在每个点上添加数值标签
        for j, (state, percentage) in enumerate(zip(states, percentages)):
            plt.annotate(f'{percentage:.1f}%', 
                        (j, percentage), 
                        textcoords="offset points", 
                        xytext=(0, 15), 
                        ha='center',
                        fontsize=11,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', 
                                alpha=0.9,
                                edgecolor=colors[i]))
    
    # 设置图形属性
    plt.xlabel('States', fontsize=14, fontweight='bold')
    plt.ylabel('Group Percentage (%)', fontsize=14, fontweight='bold')
    plt.title(f'{type_id} - Grouped Cluster Trends: (C1+C3) vs (C2+C4)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=13, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置y轴范围
    all_percentages = [p for percentages in group_data.values() for p in percentages]
    y_min = max(0, min(all_percentages) - 5)
    y_max = min(100, max(all_percentages) + 5)
    plt.ylim(y_min, y_max)
    
    # 美化图形
    plt.tight_layout()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 保存图形
    output_path = Path(output_dir) / f"{type_id}_grouped_trends_type2_C13_vs_C24.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"已保存分组趋势图(类型2): {output_path}")
    
    # 打印分组数据
    print("\n分组趋势数据 (C1+C3 vs C2+C4):")
    for state in states:
        group_x = group_data['Group X (C1+C3)'][states.index(state)]
        group_y = group_data['Group Y (C2+C4)'][states.index(state)]
        print(f"  {state}: Group X (C1+C3) = {group_x:.2f}%, Group Y (C2+C4) = {group_y:.2f}%")
    
    return output_path, group_data


def save_grouped_data_tables(group_data_type1, group_data_type2, all_data, output_dir, type_id='WT'):
    """保存分组数据表格为CSV文件"""
    
    states = list(all_data.keys())
    if not states:
        return
    
    # 创建分组数据表格1: C1+C2 vs C3+C4
    df1_data = {}
    for state in states:
        df1_data[state] = [
            round(group_data_type1['Group A (C1+C2)'][states.index(state)], 2),
            round(group_data_type1['Group B (C3+C4)'][states.index(state)], 2)
        ]
    
    df1 = pd.DataFrame(df1_data, index=['Group A (C1+C2)', 'Group B (C3+C4)'])
    csv_path1 = Path(output_dir) / f"{type_id}_grouped_percentages_type1_C12_vs_C34.csv"
    df1.to_csv(csv_path1)
    print(f"已保存分组数据表格1: {csv_path1}")
    
    # 创建分组数据表格2: C1+C3 vs C2+C4
    df2_data = {}
    for state in states:
        df2_data[state] = [
            round(group_data_type2['Group X (C1+C3)'][states.index(state)], 2),
            round(group_data_type2['Group Y (C2+C4)'][states.index(state)], 2)
        ]
    
    df2 = pd.DataFrame(df2_data, index=['Group X (C1+C3)', 'Group Y (C2+C4)'])
    csv_path2 = Path(output_dir) / f"{type_id}_grouped_percentages_type2_C13_vs_C24.csv"
    df2.to_csv(csv_path2)
    print(f"已保存分组数据表格2: {csv_path2}")
    
    return csv_path1, csv_path2


def generate_summary_report(all_data, output_dir, type_id='WT'):
    """生成分析总结报告"""
    
    states = list(all_data.keys())
    if not states:
        return
    
    report_path = Path(output_dir) / f"{type_id}_cluster_trends_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"{type_id} Cluster比例变化趋势分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"分析状态: {', '.join(states)}\n")
        f.write(f"聚类数量: {len(all_data[states[0]])}\n\n")
        
        # 每个cluster的变化趋势
        for cluster_id in range(1, len(all_data[states[0]]) + 1):
            f.write(f"Cluster {cluster_id} 变化趋势:\n")
            f.write("-" * 30 + "\n")
            
            percentages = []
            for state in states:
                percentage = all_data[state].get(cluster_id, 0)
                percentages.append(percentage)
                f.write(f"  {state}: {percentage:.2f}%\n")
            
            # 计算变化量
            max_perc = max(percentages)
            min_perc = min(percentages)
            range_perc = max_perc - min_perc
            
            f.write(f"  最大变化幅度: {range_perc:.2f}% (从 {min_perc:.2f}% 到 {max_perc:.2f}%)\n")
            
            # 确定主要趋势
            if percentages[-1] > percentages[0]:
                trend = "上升"
            elif percentages[-1] < percentages[0]:
                trend = "下降"
            else:
                trend = "基本稳定"
            
            f.write(f"  总体趋势: {trend} (从Apo到PS: {percentages[0]:.2f}% → {percentages[-1]:.2f}%)\n\n")
        
        # 主导cluster分析
        f.write("各状态主导cluster分析:\n")
        f.write("-" * 30 + "\n")
        
        for state in states:
            dominant_cluster = max(all_data[state], key=all_data[state].get)
            dominant_percentage = all_data[state][dominant_cluster]
            f.write(f"{state}状态: Cluster {dominant_cluster} ({dominant_percentage:.2f}%)\n")
        
        # 分组分析
        f.write("\n分组分析结果:\n")
        f.write("=" * 40 + "\n\n")
        
        # 分组1: C1+C2 vs C3+C4
        f.write("分组类型1 - Cluster 1+2 vs Cluster 3+4:\n")
        f.write("-" * 40 + "\n")
        for state in states:
            group_a = all_data[state].get(1, 0) + all_data[state].get(2, 0)
            group_b = all_data[state].get(3, 0) + all_data[state].get(4, 0)
            f.write(f"  {state}: Group A(C1+C2) = {group_a:.2f}%, Group B(C3+C4) = {group_b:.2f}%\n")
        
        # 计算分组1的变化趋势
        group_a_values = [all_data[state].get(1, 0) + all_data[state].get(2, 0) for state in states]
        group_b_values = [all_data[state].get(3, 0) + all_data[state].get(4, 0) for state in states]
        
        group_a_trend = "上升" if group_a_values[-1] > group_a_values[0] else "下降" if group_a_values[-1] < group_a_values[0] else "稳定"
        group_b_trend = "上升" if group_b_values[-1] > group_b_values[0] else "下降" if group_b_values[-1] < group_b_values[0] else "稳定"
        
        f.write(f"\n  Group A(C1+C2) 总体趋势: {group_a_trend} ({group_a_values[0]:.2f}% → {group_a_values[-1]:.2f}%)\n")
        f.write(f"  Group B(C3+C4) 总体趋势: {group_b_trend} ({group_b_values[0]:.2f}% → {group_b_values[-1]:.2f}%)\n\n")
        
        # 分组2: C1+C3 vs C2+C4
        f.write("分组类型2 - Cluster 1+3 vs Cluster 2+4:\n")
        f.write("-" * 40 + "\n")
        for state in states:
            group_x = all_data[state].get(1, 0) + all_data[state].get(3, 0)
            group_y = all_data[state].get(2, 0) + all_data[state].get(4, 0)
            f.write(f"  {state}: Group X(C1+C3) = {group_x:.2f}%, Group Y(C2+C4) = {group_y:.2f}%\n")
        
        # 计算分组2的变化趋势
        group_x_values = [all_data[state].get(1, 0) + all_data[state].get(3, 0) for state in states]
        group_y_values = [all_data[state].get(2, 0) + all_data[state].get(4, 0) for state in states]
        
        group_x_trend = "上升" if group_x_values[-1] > group_x_values[0] else "下降" if group_x_values[-1] < group_x_values[0] else "稳定"
        group_y_trend = "上升" if group_y_values[-1] > group_y_values[0] else "下降" if group_y_values[-1] < group_y_values[0] else "稳定"
        
        f.write(f"\n  Group X(C1+C3) 总体趋势: {group_x_trend} ({group_x_values[0]:.2f}% → {group_x_values[-1]:.2f}%)\n")
        f.write(f"  Group Y(C2+C4) 总体趋势: {group_y_trend} ({group_y_values[0]:.2f}% → {group_y_values[-1]:.2f}%)\n")
    
    print(f"已保存分析报告: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='WT四个状态cluster比例变化趋势分析')
    parser.add_argument('-t', '--type', type=str, default='WT', help='分析类型ID')
    parser.add_argument('-states', '--states', nargs='+', default=['Apo', 'ES', 'DS', 'PS'],
                        help='要分析的状态列表')
    parser.add_argument('-base', '--base_dir', type=str, 
                        default='/M-run-complex-FAST-10/analysis_PCA_new/clustering_results_all_states',
                        help='聚类结果基础目录')
    parser.add_argument('-output', '--output_dir', type=str,
                        default='/M-run-complex-FAST-10/analysis_PCA_new/cluster_trends_analysis',
                        help='输出目录')
    
    args = parser.parse_args()
    
    print(f"WT Cluster比例变化趋势分析")
    print("=" * 60)
    print(f"分析类型: {args.type}")
    print(f"分析状态: {', '.join(args.states)}")
    print(f"输入目录: {args.base_dir}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有状态的数据
    print("\n正在收集cluster比例数据...")
    all_data = collect_all_state_data(args.base_dir, args.type, args.states)
    
    if not all_data:
        print("错误: 没有找到任何可用的cluster数据")
        return
    
    print(f"\n成功收集 {len(all_data)} 个状态的数据")
    
    # 生成可视化图表
    print("\n正在生成趋势折线图...")
    create_trend_plot(all_data, output_dir, args.type)
    
    print("正在生成堆叠柱状图...")
    create_stacked_bar_chart(all_data, output_dir, args.type)
    
    # 保存数据表格
    print("正在生成数据表格...")
    create_data_table(all_data, output_dir, args.type)
    
    # 生成分组趋势分析图
    print("\n正在生成分组趋势分析图...")
    print("分组类型1: Cluster 1+2 vs Cluster 3+4")
    _, group_data_type1 = create_grouped_trend_plot_type1(all_data, output_dir, args.type)
    
    print("\n分组类型2: Cluster 1+3 vs Cluster 2+4")
    _, group_data_type2 = create_grouped_trend_plot_type2(all_data, output_dir, args.type)
    
    # 保存分组数据表格
    print("\n正在保存分组数据表格...")
    save_grouped_data_tables(group_data_type1, group_data_type2, all_data, output_dir, args.type)
    
    # 生成分析报告
    print("\n正在生成分析报告...")
    generate_summary_report(all_data, output_dir, args.type)
    
    print(f"\nCluster比例变化趋势分析完成！")
    print(f"所有结果保存在: {output_dir}")


if __name__ == "__main__":
    main() 