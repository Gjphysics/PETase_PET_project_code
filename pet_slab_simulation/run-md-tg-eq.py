import os
import subprocess
from tqdm import tqdm
import argparse
import sys
#运行指令：
# 设置命令行参数解析器###################################################
parser = argparse.ArgumentParser(description="运行脚本的工作目录和参数校验")
parser.add_argument('-mer', type=int, required=True, help="值必须是2, 4, 8, 20, 200, 2000之一")
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
mdp_dir = f"{home_dir}/T-g-script/mdp-files"
script_dir = f"{home_dir}/T-g-script"
#更改当前工作路径
os.chdir(f"{home_dir}/{file_name}")
# 获取当前工作路径
wkdir = os.getcwd()
print(wkdir)
#函数声明区域############################################################
def check_former_file(name):
    if not os.path.exists(name):
        print(f'{name} 文件不存在,程序结束')
        sys.exit(1)
#开始正式代码############################################################
for i in tqdm(range(600, 99,-interval)):
    os.makedirs(f'{i}', exist_ok=True)
    #先检查一下运行是否完成，如果完成就继续下一次循环
    if os.path.exists(f'{wkdir}/{i}/2-eq.gro'):
        print(f'{i}K的模拟已完成')
        continue  

    #先把初始文件准备好
    if i == 600:
        cmd = f'cp {wkdir}/PET-{file_name}.gro {wkdir}/{i}/start.gro'
        subprocess.run(cmd, shell=True)
        print(cmd)
    else:
        cmd = f'cp {wkdir}/{i + interval}/2-eq.gro {wkdir}/{i}/start.gro'
        subprocess.run(cmd, shell=True)
        print(cmd)
    
    #开始npt平衡
    #gpu = count % 4
    #先检查一下上一步是否完成，防止生成过多无效文件
    check_former_file(f'{wkdir}/{i}/start.gro')

    if i == 600:
        with open(f'{wkdir}/{i}/output.log', 'w') as logfile:
            cmd = f'bash {script_dir}/run-md-tg-t-eq-600.sh {wkdir}/{i} {i} {gpu} {mdp_dir} {file_name}'
            subprocess.run(cmd, shell=True, stdout=logfile, stderr=logfile)
        
    else:
        cmd = f'cp {wkdir}/{i + interval}/2-eq.cpt {wkdir}/{i}/start.cpt'
        subprocess.run(cmd, shell=True)
        with open(f'{wkdir}/{i}/output.log', 'w') as logfile:
            # 运行命令并将输出重定向到日志文件
            cmd = f'bash {script_dir}/run-md-tg-t-eq.sh {wkdir}/{i} {i} {gpu} {mdp_dir} {file_name}'
            subprocess.run(cmd, shell=True, stdout=logfile, stderr=logfile)
        
    #计数器+1
    count += 1

    