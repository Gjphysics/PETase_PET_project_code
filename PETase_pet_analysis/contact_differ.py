import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from argparse import ArgumentParser
import pandas as pd
import seaborn as sns
from config import get_config

def combine_residue_contact_csv_single_round(wkdir,file_name_without_dir, state_num, output_file):
    """
    合并指定round和state下所有run目录中的单个残基接触CSV文件，生成一个总文件。
    """
    base_dir = wkdir
    all_dfs = []

    # 遍历所有可能的run目录（假设run从1到12）
    for run in range(1, 130):
        run_path = os.path.join(base_dir, f"run-{run}")
        csv_file = os.path.join(run_path, "process", "state_old", f"{file_name_without_dir}_{state_num}.csv")
        
        # 检查文件是否存在
        if not os.path.exists(csv_file):
            print(f"文件 {csv_file} 不存在，跳过。")
            continue
        
        # 读取CSV文件并添加run列
        df = pd.read_csv(csv_file)
        #df['run'] = run  # 添加run编号列以区分不同run的数据
        all_dfs.append(df)
    
    # 检查是否有数据可合并
    if not all_dfs:
        print("未找到任何CSV文件，无法合并。")
        return
    
    # 合并所有DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # 保存合并后的CSV文件
    combined_df.to_csv(output_file, index=False)
    print(f"合并后的数据已保存至 {output_file}")


def plot_csv_average(csv_file, output_file, color_id, state_num, ylim=None):
    """
    读取CSV文件，计算每个残基的平均contact数并绘制折线图
    
    参数说明：
        - csv_file: 输入的CSV文件路径（包含所有时间点的残基contact数据）
        - output_file: 输出图片路径
        - ylim: 可选，设置y轴范围
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 排除时间列（第一列）
    residue_columns = df.columns[1:]
    
    # 计算每个残基的平均contact数
    avg_contacts = df[residue_columns].mean(axis=0)
    
    # 提取残基编号并转换为整数
    res_ids = [int(col.split('_')[1]) for col in residue_columns]
    res_ids = [res + config["resi_add_num"] for res in res_ids]
    # 创建包含残基编号和平均值的DataFrame并按残基编号排序
    avg_df = pd.DataFrame({'Residue': res_ids, 'Average Contact': avg_contacts.values})
    avg_df = avg_df.sort_values('Residue').reset_index(drop=True)

    # 存储平均值数据到CSV文件
    avg_df.to_csv(output_file.replace(".png", ".csv"), index=False)

    # 创建画布
    plt.figure(figsize=(12, 6))
    
    # 绘制折线图
    plt.plot(avg_df['Residue'], avg_df['Average Contact'], linestyle='-', linewidth=2, markersize=8,
             color=color_id)  # 使用用户之前定义的color_id变量更合适
    
    # 设置标签和标题
    plt.xlabel("Residue Index", fontsize=14)
    plt.ylabel("Average Contact Number", fontsize=14)
    plt.title(f"Average Contact Number per Residue (State {state_num})", fontsize=16)
    
    # 设置坐标轴范围
    if ylim is not None:
        plt.ylim(0, ylim)
    plt.xlim(res_ids[0], res_ids[-1])
    # 设置x轴刻度，只显示每隔20个残基编号的刻度
    xticks = avg_df['Residue'][::20]  # 每隔20个残基编号
    plt.xticks(xticks, fontsize=12)

    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"折线图已保存至 {output_file}")

def plot_contact_difference(csv_current, csv_previous, color_id_current="red", color_id_previous="blue",output_file=None, ylim=None, save_diff_file=None):
    """
    计算两个状态间的残基接触数差异并绘制差异图
    
    参数说明：
        - csv_current: 当前状态（state_n）的CSV文件路径
        - csv_previous: 前一状态（state_n-1）的CSV文件路径
        - output_file: 可选，输出图片路径
        - ylim: 可选，设置y轴对称范围（例如设置10表示范围[-10, 10]）
        - save_diff_file: 可选，保存差异数据的文件路径。如果不提供则不保存。
    """
    # 读取两个状态的数据
    df_current = pd.read_csv(csv_current)
    df_previous = pd.read_csv(csv_previous)
    
    # 确保列名一致（排除时间列）
    current_cols = df_current.columns[1:]
    previous_cols = df_previous.columns[1:]
    assert set(current_cols) == set(previous_cols), "残基列不匹配"
    
    # 按列名排序确保顺序一致
    df_current = df_current.reindex(columns=df_current.columns[0:1].tolist() + sorted(current_cols, key=lambda x: int(x.split('_')[1])))
    df_previous = df_previous.reindex(columns=df_previous.columns[0:1].tolist() + sorted(previous_cols, key=lambda x: int(x.split('_')[1])))
    
    # 计算平均接触数
    avg_current = df_current[current_cols].mean(axis=0)
    avg_previous = df_previous[previous_cols].mean(axis=0)
    
    # 计算差异（当前状态 - 前一状态）
    difference = avg_current - avg_previous
    
    # 创建差异DataFrame
    res_ids = [int(col.split('_')[1]) for col in difference.index]
    res_ids = [res + config["resi_add_num"] for res in res_ids]
    diff_df = pd.DataFrame({'Residue': res_ids, 'Difference': difference.values})
    diff_df = diff_df.sort_values('Residue').reset_index(drop=True)
    
    # 如果提供了保存路径，则保存差异数据到文件
    if save_diff_file is not None:
        diff_df.to_csv(save_diff_file, index=False)
        print(f"差异数据已保存至 {save_diff_file}")
    if  output_file is not None:
        # 创建画布
        plt.figure(figsize=(12, 6))
        
        # 创建颜色映射（红色正差异，蓝色负差异）
        colors = np.where(diff_df['Difference'] >= 0, color_id_current, color_id_previous)
        
        # 绘制柱状图
        plt.bar(diff_df['Residue'], diff_df['Difference'], 
                color=colors, 
                edgecolor='black', 
                linewidth=0.5,
                width=0.8)
        
        # 设置标签和标题
        plt.xlabel("Residue Index", fontsize=14)
        plt.ylabel("Contact Difference", fontsize=14)
        #plt.title(f"Contact Difference between State {state_num} and State {state_num-1}", fontsize=16)
        
        # 设置对称y轴范围
        if ylim is not None:
            plt.ylim(-ylim, ylim)
        else:
            # 自动设置对称范围
            max_diff = np.abs(diff_df['Difference']).max()
            plt.ylim(-max_diff*1.1, max_diff*1.1)
        
        plt.xlim(res_ids[0], res_ids[-1])
        
        # 设置x轴刻度
        xticks = diff_df['Residue'][::20]
        plt.xticks(xticks, fontsize=12)
        
        # 添加零线参考
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        
        # 紧凑布局
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"差异图已保存至 {output_file}")

# 激活函数定义,单位是Å
def contact_function(x):
    a=1
    d=8
    return 0.5 - 0.5 * np.tanh(a * (x - d))

# 新增WT数据处理函数
def process_wt_data():
    wt_dir = "/M-run-complex-FAST-10/result/contact_differ/WT"
    output_dir = os.path.join(wt_dir, "averaged")
    os.makedirs(output_dir, exist_ok=True)
    
    for state_num in range(1, 6):
        input_file = os.path.join(wt_dir, f"single_residue_contact_{state_num}_all_go.csv")
        if not os.path.exists(input_file):
            print(f"WT state {state_num} data missing")
            continue
        print("start processing WT")
        df = pd.read_csv(input_file)
        residue_cols = [col for col in df.columns if col.startswith('resid_')]
        
        # 计算平均值
        avg_contacts = df[residue_cols].mean().reset_index()
        avg_contacts.columns = ['Residue', 'Average Contact']
        # 修改后（正确）
        avg_contacts['Residue'] = avg_contacts['Residue'].str.extract('(\d+)').astype(int) + config["resi_add_num"]
        
        # 保存平均值文件
        output_file = os.path.join(output_dir, f"single_residue_contact_average_{state_num}.csv")
        avg_contacts.to_csv(output_file, index=False)
        print(f"Processed WT state {state_num} average data")

# 修改后的对比绘图函数
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
        csv_path = os.path.join(config['base_dir'], "result","contact_differ",  
                               f"single_residue_contact_average_{state_num}.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            ax.plot(df['Residue'], df['Average Contact'], 
                    color=colors[i], 
                    linewidth=2,
                    linestyle=line_styles[i],
                    label=labels[i],
                    alpha=0.8)
        else:
            print(f"{run_type} average data not found")

    # 设置标签和标题
    ax.set_xlabel("Residue Index", fontsize=14)
    ax.set_ylabel("Average Contact Number", fontsize=14)
    ax.set_title(f"State {state_num} Comparison", fontsize=16)
    
    # 设置图例
    ax.legend()
    
    # 设置坐标轴范围
    if ylim:
        ax.set_ylim(0, ylim)
    ax.set_xlim(df['Residue'].min(), df['Residue'].max())
    
    # 设置x轴刻度
    ax.set_xticks(df['Residue'][::20])
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

if __name__ == "__main__":
    ArgumentParser(description="Contact difference between two states")
    argument = ArgumentParser()
    argument.add_argument("-t", 
                    type=str, 
                    choices=["WT", "FAST", "WT119", "WT141","HOT","Thermo","WT188","FAST119"], 
                    help="Run type: FAST, WT119, or WT141 (case-sensitive)")
    argument.add_argument("-main", action="store_true", 
                    help="Main function to run")
    argument.add_argument("-compare", action="store_true", 
                      help="Compare all three run types in one plot")
    argument.add_argument("-pylim", type=int, help="Line plot y limit", default=80)
    if argument.parse_args().compare:
        #process_wt_data()
        output_dir = "/M-run-complex-FAST-10/analysis/result_all/contact_compare_all"
        os.makedirs(output_dir, exist_ok=True)
        
        for state_num in range(1, 6):
            plot_state_comparison(output_dir, state_num, argument.parse_args().pylim)
    if argument.parse_args().main:
        run_type = argument.parse_args().t
        main_num = argument.parse_args().main

        config = get_config(argument.parse_args().t)

        color_id = "#FFBE7A"
        pet_range = config["group_b"]
        resid_PETase = config["protein_range"]
        mother_dir = config['base_dir']
        output_dir = os.path.join(mother_dir, "result", "contact_differ")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(mother_dir, exist_ok=True)

        for state_num in tqdm(range(1, 6)):
            wkdir = os.path.join(mother_dir, "run-all")

            combine_residue_contact_csv_single_round(wkdir, "single_residue_contact",state_num, 
            os.path.join(output_dir, f"single_residue_contact_{state_num}_all.csv"))
            
            # 画出折线图
            plot_csv_average(
                os.path.join(output_dir, f"single_residue_contact_{state_num}_all.csv"), 
                os.path.join(output_dir, f"single_residue_contact_average_{state_num}.png"), color_id, state_num, argument.parse_args().pylim)
            
            if state_num > 2:
                #计算差异
                plot_contact_difference(
                    csv_current=os.path.join(output_dir, f"single_residue_contact_{state_num}_all.csv"), 
                    csv_previous=os.path.join(output_dir, f"single_residue_contact_{state_num-1}_all.csv"), 
                    output_file=os.path.join(output_dir, f"contact_difference_{state_num}.png"))
        
