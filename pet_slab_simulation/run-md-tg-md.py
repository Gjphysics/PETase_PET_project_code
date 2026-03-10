import os
import subprocess
# from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import sys
#运行指令：
# 设置命令行参数解析器###################################################
parser = argparse.ArgumentParser(description="运行脚本的工作目录和参数校验")
parser.add_argument('-mer', type=int, required=True, help="值必须是2, 4, 8, 10, 20, 100, 200, 400, 1000, 2000之一")
parser.add_argument('-gpu', type=int, required=True, help="请提供GPU_ID")
# 解析命令行参数
args = parser.parse_args()
# 验证参数值是否在指定范围内
if args.mer not in [2, 4, 8, 10, 20, 100, 200, 400, 1000, 2000]:
    print("参数值必须是2, 4, 8, 10, 20, 100, 200, 400, 1000, 2000之一")
    sys.exit(1)
#一系列变量的设置########################################################
gpu = args.gpu
mer = args.mer
chain_num = int(2000/mer)
file_name = f'{mer}-{chain_num}'
count = 0
interval = 50
home_dir = "/data/hcz/M-run-2000PET"
script_dir = f"{home_dir}/T-g-script"
#更改当前工作路径
os.chdir(f"{home_dir}/{file_name}")
# 获取当前工作路径
wkdir = os.getcwd()
print(wkdir)
#开始正式代码############################################################
for i in tqdm(range(600, 99,-interval)):
    #gpu = count % 4
    cmd = f'bash {script_dir}/run-md-tg.sh {wkdir}/{i} {gpu}'
    subprocess.run(cmd, shell=True)
    print(cmd)
    count += 1

#另一种解决方案
# def run_gmx(i, gpu):
#     cmd = f'cd {wkdir}/{i} && gmx mdrun -v -deffnm 3-run -gpu_id {gpu} -ntomp 4 -ntmpi 1 && echo -e "17 0" | gmx energy -f 3-run.edr -o density-md.xvg'
#     subprocess.run(cmd, shell=True)

# # 使用线程池来管理GPU任务，每个GPU一个线程
# with ThreadPoolExecutor(max_workers=4) as executor:
#     futures = []
#     for count, i in enumerate(range(600, 99, -interval)):
#         gpu = count % 4
#         futures.append(executor.submit(run_gmx, i, gpu))

#     for future in as_completed(futures):
#         future.result()  # 等待所有任务完成