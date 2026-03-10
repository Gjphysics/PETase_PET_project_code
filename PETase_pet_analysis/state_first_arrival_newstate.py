import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from config import get_config

def get_first_arrival_times(data):
    """从状态数据中提取首次到达各新状态的时间"""
    # 新状态映射
    state_map = {
        1: "Apo",
        2: "ES",
        3: "ES",
        4: "DS",
        5: "PS"
    }
    first_time = {"Apo": np.nan, "ES": np.nan, "DS": np.nan, "PS": np.nan}
    seen_states = set()
    
    for entry in data:
        time, state = entry[0], int(entry[1])
        state_name = state_map.get(state, None)
        if state_name is not None and state_name not in seen_states:
            first_time[state_name] = time
            seen_states.add(state_name)
    return first_time

def process_first_arrival(run_type):
    """处理指定类型的所有轨迹数据，只统计能到达state5的轨迹"""
    config = get_config(run_type)
    base_dir = config['base_dir']
    run_path = os.path.join(base_dir, "run-all")

    # 保存结果
    output_dir = os.path.join(base_dir, "result", "first_arrival_analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{run_type}_state_first_arrival_new.csv")
    if os.path.exists(output_file):
        # 如果文件存在，则直接读取
        results_df = pd.read_csv(output_file)
        return {
            "states": results_df["State"].tolist(),
            "times": results_df["AverageTime(ns)"].tolist(),
            "counts": results_df["Count"].tolist(),
            "run_type": run_type
        }
    
    state_times = {"Apo": [], "ES": [], "DS": [], "PS": []}
    state_counts = {"Apo": 0, "ES": 0, "DS": 0, "PS": 0}
    
    # 遍历所有可能的run目录
    for run in tqdm(range(1, 140), desc=f"处理 {run_type} 数据"):
        data_file = os.path.join(run_path, f"run-{run}", "process", "state-a.xvg")
        
        if not os.path.exists(data_file):
            continue
        
        try:
            data = np.loadtxt(data_file, comments=["#", "@"])
            # 检查是否到达state5
            if 5 not in data[:, 1]:
                continue
            first_times = get_first_arrival_times(data)
            
            for state in ["Apo", "ES", "DS", "PS"]:
                if not np.isnan(first_times[state]):
                    state_times[state].append(first_times[state])
                    state_counts[state] += 1
                    
        except Exception as e:
            print(f"Error processing {data_file}: {str(e)}")
            continue
    
    # 计算平均值（转换为纳秒）
    avg_times = {}
    for state in ["Apo", "ES", "DS", "PS"]:
        times = state_times[state]
        avg_times[state] = np.mean(times)/1000 if times else np.nan  # 转换为ns
    

    
    # 保存为CSV
    results_df = pd.DataFrame({
        'State': ["Apo", "ES", "DS", "PS"],
        'AverageTime(ns)': [f"{avg_times[s]:.1f}" if not np.isnan(avg_times[s]) else "N/A" for s in ["Apo", "ES", "DS", "PS"]],
        'Count': [state_counts[s] for s in ["Apo", "ES", "DS", "PS"]]
    })
    
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return {
        "states": ["Apo", "ES", "DS", "PS"],
        "times": [avg_times[s] for s in ["Apo", "ES", "DS", "PS"]],
        "counts": [state_counts[s] for s in ["Apo", "ES", "DS", "PS"]],
        "run_type": run_type
    }

def plot_state_comparison(data_list):
    """绘制状态对比折线图"""
    plt.figure(figsize=(10, 6), dpi=300)
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    state_labels = ["Apo", "ES", "DS", "PS"]
    x = list(range(len(state_labels)))
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '>']
    
    for i, data in enumerate(data_list):
        y = data["times"]
        plt.plot(x, y, 
                marker=markers[i % len(markers)], 
                linestyle='-', 
                linewidth=2,
                color=colors[i % len(colors)], 
                markersize=8, 
                label=data["run_type"])
        
        # 标注数据量
        for xi, time, count in zip(x, y, data["counts"]):
            if not np.isnan(time):
                plt.text(xi, time+5, f"n={count}", 
                        color=colors[i % len(colors)], 
                        ha='center', 
                        fontsize=10)
    
    plt.xticks(x, labels=state_labels)
    plt.xlabel("Target State", fontweight='bold')
    plt.ylabel("First Arrival Time (ns)", fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper left')
    plt.ylim(0, 500)
    
    output_dir = "/M-run-complex-FAST-10/result/first_arrival_analysis"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "state_first_arrival_comparison_new_Thermo.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='统计首次到达新状态时间（仅统计能到达state5的轨迹）')
    parser.add_argument("-t", type=str, 
                      choices=["WT", "FAST", "WT119", "WT141", "HOT", "Thermo","Thermo119"],
                      help="Run type")
    parser.add_argument("-compare", action="store_true",
                      help="Compare all run types in one plot")
    args = parser.parse_args()
    
    if args.compare:
        # 处理所有类型的数据
        data_list = []
        # for run_type in ["WT", "WT119", "WT141", "Thermo", "Thermo119","Thermo146", "FAST","HOT"]:
        # for run_type in ["WT", "WT119", "WT141"]:
        for run_type in ["Thermo", "Thermo119","Thermo146"]:
            data = process_first_arrival(run_type)
            data_list.append(data)
        # 生成对比图
        plot_state_comparison(data_list)
    else:
        # 只处理指定类型的数据
        process_first_arrival(args.t)

if __name__ == "__main__":
    main() 