import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from config import get_config

def combine(data_1, data_2, data_3,combine_xvg):
    data = []
    index_3 = 0  # 初始化 data_3 的索引
    for i in range(len(data_1)):
        # 仅在 data_3 的索引有效范围内时取值
        if index_3 < len(data_3):
            data.append((data_1[i][0], data_1[i][1], data_2[i][1], data_3[index_3][1]))
        else:
            data.append((data_1[i][0], data_1[i][1], data_2[i][1], float('nan'), float('nan')))  # 如果超出范围，用 NaN 填充
        index_3 += 5    # 每隔 5 个时间步长取一个值
    
    # 写入文件
    with open(combine_xvg, "w") as f:
        f.write("# time(ps) contact_protein(num) contact_pocked(num) distance_nearest(nm)\n")
        for t, d1, d2, d3 in data:
            f.write(f"{t:.2f} {d1:.3f} {d2:.3f} {d3:.3f} \n")
    return data

def find_state(combine_data,save_xvg):
    state = 0
    state_list = []
    last_state = 1
    debug_data = []
    for time, c_p, c_c, d_c_5 in combine_data:
        if c_p < 1 and c_c < 1:
            state = 1
        elif c_p >= 1 and c_c < 1:
            state = 2
            debug_data.append((time, c_p, c_c, d_c_5))
        elif c_p >= 1 and c_c >= 1:
            if  d_c_5 > 1.2:
                state = 3
            elif 1.2 >= d_c_5 > 0.5:
                state = 4
            elif 0.5 >= d_c_5:
                state = 5
        else:
            state = 0

        state_original = state

        num_up_c_p = 5
        num_up_c_c = 2
        num_down_c_p = 0
        num_down_c_c = 0

        #先给一个正向的过边界
        if state > last_state:
            if c_p < num_up_c_p and c_c < num_up_c_c:
                state = 1
            elif c_p >= num_up_c_p and c_c < num_up_c_c:
                state = 2
            elif c_p >= num_up_c_p and c_c >= num_up_c_c:
                if  d_c_5 > 1.0:
                    state = 3
                elif 1.0 >= d_c_5 > 0.4:
                    state = 4
                elif 0.4 >= d_c_5:
                    state = 5
        #再给一个反向的过边界
        elif state < last_state:
            if c_p <= num_down_c_p and c_c <= num_down_c_c:
                state = 1
            elif c_p > num_down_c_p and c_c <= num_down_c_c:
                state = 2
            elif c_p > num_down_c_p and c_c > num_down_c_c:
                if  d_c_5 > 2.0:
                    state = 3
                elif 2.0 >= d_c_5 > 0.8:
                    state = 4
                elif 0.8 >= d_c_5:
                    state = 5

        if state_original < last_state and state > last_state:
            state = max(state_original, last_state)
        elif state_original > last_state and state < last_state:
            state = min(state_original, last_state)

        last_state = state

        state_list.append((time, state))
    fix_num = 0
    fix_length = 5
    #处理一下一些突然数据
    state_list_fix1 = []
    if fix_num == 0:
        state_list_fix1 = state_list
    else:
        for i in range(len(state_list)):
            if i > fix_length :
                fix = round(np.mean([state_list[j][1] for j in range(i-fix_length, i)]))
            else:
                fix = state_list[i][1]
            state_list_fix1.append((state_list[i][0], fix))

    with open(save_xvg, "w") as f:
        f.write("# time(ps) state\n")
        for t, s in state_list_fix1:
            f.write(f"{t:.2f} {s:.0f}\n")
    return state_list_fix1

def plot_state(data, save_file):
    data = np.array(data)
    plt.figure(dpi=300)
    plt.plot(data[:, 0]/1000, data[:, 1], color = color_id)
    plt.xlabel("Time (ns)", fontsize=14)
    plt.ylabel("State", fontsize=14)
    plt.xlim(0, 1000)
    plt.ylim(0, 6)
    #plt.title("Distance between Residue and GroupB", fontsize=16)
    #plt.legend(fontsize=12)
    #plt.grid(True)
    plt.savefig(save_file)
    plt.close()

def backup_existing(file):
    if os.path.exists(file):
        os.rename(file, f"{file}.bak")

if __name__ == "__main__":
    ArgumentParser(description="Radar plot for distance between residue and groupB")
    argument = ArgumentParser()
    #argument.add_argument("-run", type=int, help="Run number", required=True)
    argument.add_argument("-t", 
                    type=str, 
                    choices=["WT","FAST", "WT119", "WT141","HOT","Thermo","FAST119","WT188","Thermo119","Thermo146"], 
                    required=True,
                    help="Run type: FAST, WT119, or WT141 (case-sensitive)")
    argument.add_argument("-test", action="store_true", help="Test mode")
    argument.add_argument("-name", type=str, help="Name of the file",required=True)
    argument.add_argument("-debug", action="store_true", help="Debug mode")
    args = argument.parse_args()

    #run_num = args.run
    run_type = args.t

    # 获取配置
    config = get_config(args.t)
    color_id = "#FFBE7A"
    for run_num in tqdm(range(1,151)):
        if not os.path.exists(os.path.join(config['base_dir'], "run-all", f"run-{run_num}", "process")):
            continue
        file_path = os.path.join(config['base_dir'], "run-all", f"run-{run_num}", "process")
        save_combine_file = f"{file_path}/combine_test.xvg"
        contact_protein = np.loadtxt(f"{file_path}/pbc_contact_protein.xvg",comments=["#","@"])
        contact_pocked = np.loadtxt(f"{file_path}/pbc_contact_binding_pose.xvg",comments=["#","@"])
        distance_nearest = np.loadtxt(f"{file_path}/pbc_rader.xvg",comments=["#","@"])

        combine_data = combine(contact_protein, contact_pocked, distance_nearest, save_combine_file)

        state_data = find_state(combine_data, f"{file_path}/state-{argument.parse_args().name}.xvg")
        if argument.parse_args().debug:
            plot_state(state_data, f"{file_path}/state-debug.png")
        else:
            if os.path.exists(f"{file_path}/state-debug.png"):
                os.remove(f"{file_path}/state-debug.png")
            plot_state(state_data, f"{file_path}/state-{argument.parse_args().name}.png")

    # if run_num:
    #     main(round_num, run_num)
    # elif not run_num:
    #     for i in tqdm(range(1,13)):
    #         main(round_num, i)

    # if argument.parse_args().test:
    #     """
    #     Test mode
    #     """
    #     state_data = find_state(combine_data, f"{file_path}/state-test.xvg")
    #     plot_state(state_data, f"{file_path}/state-test.png")
    #     exit()