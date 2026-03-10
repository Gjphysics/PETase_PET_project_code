#!/bin/bash

# 遍历run-1到run-12目录
for i in {1..100}; do
  dir="/M-run-complex-$Type_id-10/run-all/run-$i"
  # 检查2-eq.gro文件是否存在
  if [ ! -f "$dir/2-eq.gro" ]; then
    echo "文件 $dir/2-eq.gro 不存在，跳过run-$i"
    continue
  fi
  job_key="run-$i"
  mkdir -p $dir
  # 创建每个目录的Slurm作业脚本
  cat > "$dir/job_state_script.sh" <<EOF
#!/bin/bash
#SBATCH -p i64m512u
#SBATCH -J $job_key      # 唯一的作业名称
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH -o $dir/job.%j.out   # 输出文件按目录区分
#SBATCH -e $dir/job.%j.err

source /hpc2ssd/softwares/anaconda3/etc/profile.d/conda.sh  # commented out by conda initialize
conda activate plot  # commented out by conda initialize

module load mpi/mpich-4.1.2
module load gromacs/2022.2

cd $dir  # 切换到对应的工作目录

script_dir="/M-run-complex-$Type_id-10/analysis"
python \$script_dir/radar.py -t $Type_id -run $i
python \$script_dir/contact-protein.py -t $Type_id -run $i
python \$script_dir/contact-binding-pose.py -t $Type_id -run $i
# python \$script_dir/state.py -t $Type_id -run $i -name a
EOF

  # 赋予脚本执行权限并提交作业
  chmod +x "$dir/job_state_script.sh"
  sbatch "$dir/job_state_script.sh"
done