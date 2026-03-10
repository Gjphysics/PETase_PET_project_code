import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from config import get_config



def save_rmsf_average_all_runs(wkdir, state_num, output_csv):
    """
    读取所有 round_num 个 round 内的所有 run 的 RMSF 数据，
    计算每个残基的平均 RMSF 值，并保存到 CSV 文件中。
    
    参数:
        round_num (int): round 的数量。
        state_num (int): 状态编号。
        output_csv (str): 输出的 CSV 文件路径。
    """
    # 存储所有数据
    all_data = []
    
    for i in range(1, 120):
        base_dir = os.path.join(wkdir, f"run-{i}")
        if not os.path.exists(os.path.join(base_dir, "process/rmsf")):
            continue
        file_path = f"{base_dir}/process/rmsf/rmsf_{state_num}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_data.append(df["Average_RMSF"].values)
    
    # 将所有数据转换为 NumPy 数组以便处理
    all_data = np.array(all_data, dtype=float)
    
    # 计算每个残基的平均 RMSF 值
    average_rmsf = np.mean(all_data, axis=0)
    
    # 创建结果 DataFrame
    result_df = pd.DataFrame({
        "# Residue_Index": df["# Residue_Index"],
        "Average_RMSF": average_rmsf
    })
    
    # 保存结果到 CSV 文件
    result_df.to_csv(output_csv, index=False)
    print(f"平均 RMSF 值已保存到 {output_csv}")


def read_xvg_file(file_path):
    """
    读取单个 .xvg 文件并解析其数据。
    
    参数:
        file_path (str): .xvg 文件的路径。
        
    返回:
        list: 包含 (残基索引, RMSF 值) 的列表。
    """
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 提取数据部分（跳过注释行和标题行）
    for line in lines:
        if not line.startswith(('#', '@')):  # 跳过注释和元信息
            columns = line.split()
            if len(columns) >= 2:  # 确保有至少两列数据
                residue_index = int(columns[0])  # 第一列：残基索引
                rmsf_value = float(columns[1])  # 第二列：RMSF 值
                data.append((residue_index + config["resi_add_num"], rmsf_value))  # 调整残基索引
    
    return data

def plot_rmsf(rmsf_file, output_plot,color_id):
    # 读取 RMSF 数据
    df = pd.read_csv(rmsf_file)

    # 绘制 RMSF 分布图
    plt.figure(figsize=(12, 6))
    plt.plot(df["# Residue_Index"], df["Average_RMSF"], color=color_id, linewidth=1)
    plt.xlabel("Residue Index", fontsize=12)
    plt.ylabel("Average RMSF (nm)", fontsize=12)
    plt.title("Average RMSF of State", fontsize=14)
    plt.ylim(0, 0.32)
    #plt.grid(linestyle="--")
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close()

# def plot_double_rmsf(rmsf_el,rmsf_go, color_id_el, color_id_go, output_plot,ylim,label_el = "Elastic",label_go = "Go-Martini"):
#     # 读取 RMSF 数据
#     df_el = pd.read_csv(rmsf_el)
#     df_go = pd.read_csv(rmsf_go)

#     # 绘制 RMSF 分布图
#     plt.figure(figsize=(12, 6))
#     plt.plot(df_el["# Residue_Index"], df_el["Average_RMSF"], color=color_id_el, linewidth=1.2, label=label_el)
#     plt.plot(df_go["# Residue_Index"], df_go["Average_RMSF"], color=color_id_go, linewidth=1.2, label=label_go)
#     plt.xlabel("Residue Index", fontsize=12)
#     plt.ylabel("Average RMSF (nm)", fontsize=12)
#     #plt.title("Average RMSF of State", fontsize=14)
#     plt.legend()
#     #plt.grid(linestyle="--")
#     plt.ylim(0, ylim)
#     plt.xlim(30, 292)
#     plt.savefig(output_plot, dpi=300, bbox_inches="tight")
#     plt.close()

def plot_difference_rmsf(rmsf_el, rmsf_go, color_id_el, color_id_go, output_plot,ylim):
    # 读取 RMSF 数据
    df_el = pd.read_csv(rmsf_el)
    df_go = pd.read_csv(rmsf_go)

    # 计算 RMSF 差异
    df_diff = df_el.copy()
    df_diff["Average_RMSF"] = df_el["Average_RMSF"] - df_go["Average_RMSF"]

    # 绘制 RMSF 差异分布柱状图，正值表示用color_id_el，负值表示用color_id_go
    plt.figure(figsize=(12, 6))
    plt.bar(df_diff["# Residue_Index"], df_diff["Average_RMSF"], color=np.where(df_diff["Average_RMSF"] > 0, color_id_el, color_id_go))
    plt.xlabel("Residue Index", fontsize=12)
    plt.ylabel("Difference in RMSF (nm)", fontsize=12)
    plt.ylim(-ylim, ylim)
    plt.xlim(30, 292)
    #plt.title("Difference in RMSF between Elastic and Go-Martini", fontsize=14)
    #plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close()

def plot_state_comparison(output_dir, state_num, ylim=None):
    # 设置全局字体参数
    plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # 使用tab10颜色映射
    colors = plt.cm.tab10.colors
    line_styles = ["-"] * len(colors)  # 所有线条使用实线
    labels = ["WT", "HOT", "Thermo", "WT119", "WT141"]  # 更新标签列表

    # 处理所有类型数据
    for i, run_type in enumerate(["WT", "HOT", "Thermo", "WT119", "WT141"]):
        config = get_config(run_type)
        csv_path = os.path.join(config['base_dir'], "result", "rmsf",  
                               f"rmsf_{run_type}_{state_num}.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            ax.plot(df['# Residue_Index'], df['Average_RMSF'], 
                    color=colors[i], 
                    linewidth=2,
                    linestyle=line_styles[i],
                    label=labels[i],
                    alpha=0.8)
        else:
            print(f"{run_type} RMSF data not found")

    # 设置标签和标题
    ax.set_xlabel("Residue Index", fontsize=14)
    ax.set_ylabel("Average RMSF (nm)", fontsize=14)
    ax.set_title(f"State {state_num} Comparison", fontsize=16)
    
    # 设置图例
    ax.legend()
    
    # 设置坐标轴范围
    if ylim:
        ax.set_ylim(0, ylim)
    ax.set_xlim(30, 292)
    
    # 设置x轴刻度
    ax.set_xticks(np.arange(30, 293, 20))
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 添加网格
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f"state_{state_num}_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {output_path}")

def plot_average_rmsf_by_state(output_dir, ylim=None):
    """
    计算并绘制所有运行类型在不同状态下的平均RMSF。
    
    参数:
        output_dir (str): 输出目录路径
        ylim (float): y轴上限
    """
    # 设置全局字体参数
    plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # 使用tab10颜色映射
    colors = plt.cm.tab10.colors
    line_styles = ["-"] * len(colors)
    run_types = ["WT", "WT119", "WT141", "HOT", "Thermo","FAST"]
    
    # 存储每个运行类型在不同状态下的平均RMSF
    state_averages = {run_type: [] for run_type in run_types}
    
    # 处理所有状态和运行类型
    for state_num in range(1, 6):
        for i, run_type in enumerate(run_types):
            config = get_config(run_type)
            csv_path = os.path.join(config['base_dir'], "result", "rmsf",  
                                   f"rmsf_{run_type}_{state_num}.csv")
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # 计算该状态下所有残基的平均RMSF
                avg_rmsf = df['Average_RMSF'].mean()
                state_averages[run_type].append(avg_rmsf)
            else:
                print(f"{run_type} RMSF data for state {state_num} not found")
                state_averages[run_type].append(None)
    
    # 绘制每个运行类型的平均RMSF随状态变化的折线图
    states = range(1, 6)
    for i, run_type in enumerate(run_types):
        if all(x is not None for x in state_averages[run_type]):
            ax.plot(states, state_averages[run_type], 
                    color=colors[i], 
                    linewidth=2,
                    linestyle=line_styles[i],
                    label=run_type,
                    marker='o',
                    markersize=8,
                    alpha=0.8)
    
    # 设置标签和标题
    ax.set_xlabel("State", fontsize=14)
    ax.set_ylabel("Average RMSF (nm)", fontsize=14)
    ax.set_title("Average RMSF Across States", fontsize=16)
    
    # 设置图例
    ax.legend()
    
    # 设置坐标轴范围

    ax.set_ylim(0.08, 0.13)
    ax.set_xlim(0.5, 5.5)
    
    # 设置x轴刻度
    ax.set_xticks(range(1, 6))
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 添加网格
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, "average_rmsf_by_state.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Average RMSF plot saved to {output_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Calculate total RMSF across all state segments")
    parser.add_argument("-t", 
                    type=str, 
                    choices=["WT", "FAST", "WT119", "WT141", "HOT", "Thermo"], 
                    help="Run type: FAST, WT119, or WT141 (case-sensitive)")
    parser.add_argument("-main", action="store_true", 
                    help="Main function to run")
    parser.add_argument("-compare", action="store_true", 
                      help="Compare all run types in one plot")
    parser.add_argument("-ave", action="store_true",
                      help="Plot average RMSF across states for all run types")
    parser.add_argument("-pylim", type=float, help="Line plot y limit", default=0.32)

    args = parser.parse_args()

    if args.ave:
        output_dir = "/M-run-complex-FAST-10/result/rmsf_compare_all"
        os.makedirs(output_dir, exist_ok=True)
        plot_average_rmsf_by_state(output_dir, args.pylim)

    if args.compare:
        output_dir = "/M-run-complex-FAST-10/result/rmsf_compare_all"
        os.makedirs(output_dir, exist_ok=True)
        
        for state_num in range(1, 6):
            plot_state_comparison(output_dir, state_num, args.pylim)

    if args.main:
        run_type = args.t
        config = get_config(run_type)
        mother_dir = config["base_dir"]
        wkdir = os.path.join(mother_dir, "run-all")
        color_id = "#FFBE7A"

        save_path = os.path.join(mother_dir, "result", "rmsf")
        os.makedirs(save_path, exist_ok=True)
        
        for state_number in tqdm(range(1, 6)):
            save_rmsf_average_all_runs(wkdir, state_number, f"{save_path}/rmsf_{run_type}_{state_number}.csv")
            plot_rmsf(f"{save_path}/rmsf_{run_type}_{state_number}.csv", f"{save_path}/rmsf_{run_type}_{state_number}.png", color_id)

            if state_number > 1:
                plot_difference_rmsf(f"{save_path}/rmsf_{run_type}_{state_number}.csv", 
                                   f"{save_path}/rmsf_{run_type}_{state_number-1}.csv", 
                                   color_id, "grey", 
                                   f"{save_path}/rmsf_diff_{run_type}_{state_number}.png", 
                                   0.05)

