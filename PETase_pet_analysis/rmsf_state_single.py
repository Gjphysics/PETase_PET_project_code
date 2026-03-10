import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
import subprocess
from config import get_config

def get_time(file_path):
    time_data = []
    with open(file_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            line = line.strip().split()
            time_data.append((int(line[0]), float(line[1]), float(line[2])))
    return time_data

def calculate_rmsf_gmx(wk_dir,index_file, output_xvg, time_control):
    gro_file = os.path.join(wk_dir, "2-eq.tpr")
    xtc_file = os.path.join(wk_dir, "2-eq-noPBC-PETase-PET.xtc")
    #print("1111")
    if not os.path.exists(xtc_file):
        origin_xtc = os.path.join(wk_dir, "2-eq.xtc")
        tpr_file = os.path.join(wk_dir, "2-eq.tpr")
        cmd = f"echo 20 20 | gmx trjconv -f {origin_xtc} -s {tpr_file} -o {xtc_file} -pbc mol -center -n {index_file}"
        subprocess.run(cmd, shell=True, check=True)

    cmd = f"echo 21 21 | gmx rmsf -s {gro_file} -f {xtc_file} -o {output_xvg} -res -n {index_file} -b {time_control[0]} -e {time_control[1]} -dt {time_control[2]}"
    subprocess.run(cmd, shell=True, check=True)

def calculate_single_run_gmx(wk_dir,state_num, index_file, state_time): 
    save_dir = os.path.join(wk_dir, f"process/rmsf/state-{state_num}")
    out_put_csv = os.path.join(wk_dir, f"process/rmsf/rmsf_{state_num}.csv")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(len(state_time)):
        time_control = (state_time[i][1], state_time[i][2], 5)

        output_xvg = os.path.join(save_dir, f"rmsf_{state_time[i][0]}_b_{state_time[i][1]}.xvg")
        if not os.path.exists(output_xvg):
            calculate_rmsf_gmx(wk_dir, index_file, output_xvg, time_control)
    
    save_rmsf_average_single_run(save_dir, state_num, out_put_csv)

def save_rmsf_average_single_run(save_dir, state_num, output_csv):
    """
    读取指定目录内的所有 .xvg 文件，检查它们是否属于同一个状态，
    计算每个残基的平均 RMSF 值，并保存到 CSV 文件中。
    
    参数:
        save_dir (str): 包含 .xvg 文件的目录路径。
        state_num (int): 状态编号，用于验证文件名前缀。
        output_csv (str): 输出的 CSV 文件路径。
    """
    # 检查目录是否存在
    if not os.path.exists(save_dir):
        print(f"目录 {save_dir} 不存在！")
        return
    
    # 获取目录内所有 .xvg 文件
    xvg_files = [f for f in os.listdir(save_dir) if f.endswith('.xvg')]
    if not xvg_files:
        print(f"目录 {save_dir} 中没有找到任何 .xvg 文件！")
        return
    
    # 验证文件名前缀是否一致
    prefix = f"rmsf_{state_num}_"
    for file_name in xvg_files:
        if not file_name.startswith(prefix):
            raise ValueError(f"文件 {file_name} 不属于状态 {state_num}！")
    
    # 存储所有文件的数据
    all_data = []
    
    # 使用抽象出的函数读取每个 .xvg 文件的数据
    for file_name in xvg_files:
        file_path = os.path.join(save_dir, file_name)
        data = read_xvg_file(file_path)
        all_data.append(data)
    
    # 将所有数据转换为 NumPy 数组以便处理
    all_data = np.array(all_data, dtype=object)
    
    # 检查所有文件的残基索引是否一致
    residue_indices = np.array([data[:, 0] for data in all_data])
    if not np.all(residue_indices == residue_indices[0]):
        raise ValueError("不同文件的残基索引不一致！")
    
    # 计算每个残基的平均 RMSF 值
    rmsf_values = np.array([data[:, 1] for data in all_data], dtype=float)
    average_rmsf = np.mean(rmsf_values, axis=0)
    
    # 创建结果 DataFrame
    result_df = pd.DataFrame({
        "# Residue_Index": residue_indices[0],
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

if __name__ == "__main__":
    parser = ArgumentParser(description="Calculate total RMSF across all state segments")
    parser.add_argument("-t", 
                    type=str, 
                    choices=["WT","FAST", "WT119", "WT141","HOT","Thermo","WT188","FAST119"], 
                    required=True,
                    help="Run type: FAST, WT119, or WT141 (case-sensitive)")
    parser.add_argument("-run", type=int, help="Run number", required=True)
    args = parser.parse_args()

    type_id = args.t
    run_num = args.run
    config = get_config(type_id)
    mother_dir = config["base_dir"]
    wkdir = os.path.join(mother_dir, "run-all")
    index_file = config["index_contact_rmsf"]
    
    for state_number in range(1, 6):
        base_dir = os.path.join(wkdir, f"run-{run_num}")
        if not os.path.exists(os.path.join(base_dir, "2-eq.gro")):
            print(f"Run {run_num} does not finish.")
            break
            
        time_file = f"{base_dir}/process/state_{state_number}_absolute_time.xvg"
        
        state_time_single_run = get_time(time_file)

        calculate_single_run_gmx(base_dir, state_number, index_file, state_time_single_run)



