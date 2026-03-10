#!/bin/bash

# 遍历run-1到run-12目录
for i in {1..120}; do
  dir="/M-run-complex-WT-10/run-all/run-$i"
  # 检查2-eq.gro文件是否存在
  if [ ! -f "$dir/process/state-a.xvg" ]; then
    echo "路径 run-$i/下不存在state文件，跳过run-$i"
    continue
  fi
  job_key="run-$i"
  mkdir -p $dir
  # 创建每个目录的Slurm作业脚本
  cat > "$dir/job_contact_rmsf_script_BB.sh" <<EOF
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
export MODULEPATH=/opt/plumed-2.9.2/lib/plumed/:\$MODULEPATH
export PATH=/opt/gromacs-2024.2/bin:\$PATH
module load pl29mod

cd $dir  # 切换到对应的工作目录

script_dir="/M-run-complex-FAST-10/analysis"

# python \$script_dir/contact_differ_single.py -run $i -t WT
python \$script_dir/rmsf_state_single_BB.py -run $i -t WT

EOF

  # 赋予脚本执行权限并提交作业
  chmod +x "$dir/job_contact_rmsf_script_BB.sh"
  sbatch "$dir/job_contact_rmsf_script_BB.sh"
done