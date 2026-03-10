import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config import get_config

def reach_counting(run_path, state_num):
    reach_count = 0
    trj_count = 0
    for i in range(1, 151):
        file_path = os.path.join(run_path, f"run-{i}", "process")
        file_to_process = f"{file_path}/state-a.xvg"
        if os.path.exists(file_to_process):
            state_data = np.loadtxt(file_to_process, comments=["#", "@"])
            reach_count += count_whether_reach_five(state_data, state_num)
            trj_count += 1
    percentage = (reach_count / trj_count * 100) if trj_count else 0
    return percentage, trj_count

def count_whether_reach_five(state_data, target_state):
    return 1 if any(state_data[:, 1] == target_state) else 0

def save_to_csv(label, percentages, output_file, runs):
    # 构建 DataFrame
    data = {
        'Label': [label],
        'Runs': [runs],
        'Apo': [percentages['Apo']],
        'ES': [percentages['ES']],
        'DS': [percentages['DS']],
        'PS': [percentages['PS']]
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

def plot_comparison(csv_files, output_plot):
    plt.rcParams.update({'font.size': 14, 'font.family': 'Arial'})

    standard_order = ['Apo', 'ES', 'DS', 'PS']
    num_states = len(standard_order)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    x = np.arange(num_states)
    colors = plt.cm.tab10.colors[:len(csv_files)]
    markers = ['o', 's', '^', 'D', 'v', '>']

    for i, (csv_file, color) in enumerate(zip(csv_files, colors)):
        df = pd.read_csv(csv_file)
        label = df['Label'].iloc[0]
        runs = df['Runs'].iloc[0]
        values = [df[state].iloc[0] if state in df.columns else 0 for state in standard_order]
        values_rounded = [round(v, 2) for v in values]

        ax.plot(x, values_rounded, marker=markers[i % len(markers)], linestyle='-', linewidth=2, color=color, label=f"{label} (Runs: {runs})")

    # 设置 x 轴标签
    ax.set_xticks(x)
    ax.set_xticklabels(standard_order)

    # 设置 y 轴和标题
    ax.set_ylabel('Percentage (%)')
    ax.set_title('State Reach Percentage Comparison')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()
    
def main():
    parser = argparse.ArgumentParser(description="State reach statistics tool")
    parser.add_argument("-t", choices=["WT","FAST", "WT119", "WT141","HOT","Thermo"], help="Run type")
    parser.add_argument("-label", help="Label for CSV first column (required for stats mode)")
    parser.add_argument("-output", help="CSV output file (required for stats mode)")
    parser.add_argument("-plot", nargs="+", help="CSV files to compare")
    parser.add_argument("-plot_output", default="comparison.png", help="Output plot filename")
    parser.add_argument("-compare", action="store_true", help="Compare all run types in one plot")
    parser.add_argument("-recompute", action="store_true", help="Force recomputation even if CSV files exist")
    
    args = parser.parse_args()

    if args.compare:
        # 处理所有类型的数据
        csv_files = []
        for run_type in  ["WT", "HOT", "Thermo", "FAST","Thermo119","Thermo146"]:
            config = get_config(run_type)
            output_file = os.path.join(config['base_dir'], "result", "first_arrival_analysis", f"{run_type}_state_reach.csv")
            
            # 检查CSV文件是否存在，如果存在且没有使用-recompute标志，则直接使用现有文件
            if os.path.exists(output_file) and not args.recompute:
                print(f"Using existing CSV file: {output_file}")
                csv_files.append(output_file)
            else:
                # 重新计算并生成CSV文件
                if args.recompute:
                    print(f"Recomputing data for {run_type}...")
                else:
                    print(f"CSV file not found for {run_type}, computing...")
                
                run_path = os.path.join(config['base_dir'], "run-all")
                percentages = {}
                trj_count = 0
                for state in range(1, 6):
                    pct, trj_count = reach_counting(run_path, state)
                    if state == 1:
                        percentages['Apo'] = round(pct, 2)
                    elif state == 2 or state == 3:
                        percentages['ES'] = round(pct, 2)
                    elif state == 4:
                        percentages['DS'] = round(pct, 2)
                    elif state == 5:
                        percentages['PS'] = round(pct, 2)
                    if state == 1:
                        total_trj_count = trj_count
                percentages['Label'] = run_type
                percentages['Runs'] = total_trj_count
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                save_to_csv(run_type, percentages, output_file, total_trj_count)
                csv_files.append(output_file)
        
        # 生成对比图
        plot_comparison(csv_files, args.plot_output)
    elif args.plot:
        plot_comparison(args.plot, args.plot_output)
    else:
        if not args.label or not args.output:
            parser.error("Stats mode requires -label and -output parameters")

        config = get_config(args.t)
        run_path = os.path.join(config['base_dir'], "run-all")

        # 初始化状态百分比字典
        percentages = {}

        # 获取有效运行数（trj_count）
        trj_count = 0
        for state in range(1, 6):
            pct, trj_count = reach_counting(run_path, state)
            if state == 1:
                percentages['Apo'] = round(pct, 2)
            elif state == 2 or state == 3:
                percentages['ES'] = round(pct, 2)
            elif state == 4:
                percentages['DS'] = round(pct, 2)
            elif state == 5:
                percentages['PS'] = round(pct, 2)
            if state == 1:
                total_trj_count = trj_count  # 所有状态的 trj_count 是一致的

        # 添加 Label 和 Runs 到字典
        percentages['Label'] = args.label
        percentages['Runs'] = total_trj_count

        # 保存到 CSV
        save_to_csv(args.label, percentages, args.output, total_trj_count)

if __name__ == "__main__":
    main() 