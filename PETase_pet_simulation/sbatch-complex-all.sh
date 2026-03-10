#!/bin/bash

# 遍历run-1到run-12目录
for i in {1..100}; do
  dir="M-run-complex-$Type_id-10/run-all/run-$i"
  job_key="run-$i"
  mkdir $dir
  # 创建每个目录的Slurm作业脚本
  cat > "$dir/job_script.sh" <<EOF
#!/bin/bash
#SBATCH -p $$$$$$$$$$
#SBATCH -J $job_key      # 唯一的作业名称
#SBATCH --nodes 1
#SBATCH --ntasks-per-node $$
#SBATCH -o $dir/job.%j.out   # 输出文件按目录区分
#SBATCH -e $dir/job.%j.err

module load mpi/mpich-4.1.2
export MODULEPATH=opt/plumed-2.9.2/lib/plumed/:\$MODULEPATH
export PATH=opt/gromacs-2024.2/bin:\$PATH
module load pl29mod

cd $dir  # 切换到对应的工作目录

# 定义路径（使用绝对路径确保正确性）
top_dir="M-run-complex-$Type_id-10/go-$Type_id/Complex.top"
gro_dir="M-run-complex-$Type_id-10/go-$Type_id/Complex-sol-ions.gro"
mdp_dir="M-run-complex-$Type_id-10/mdp-files"
plumed_dir="M-run-complex-$Type_id-10/plumed-files/virtual-wall-go.dat"
index_dir="M-run-complex-$Type_id-10/go-$Type_id/index_mdrun.ndx"

# 执行GROMACS命令
gmx grompp -p \$top_dir -c \$gro_dir -f \$mdp_dir/martini_em.mdp -o 1-min.tpr -maxwarn 1
\$nohup gmx mdrun -v -deffnm 1-min -ntomp 32

gmx grompp -p \$top_dir -c 1-min.gro -f \$mdp_dir/martini_eq_p_large.mdp -o 2-eq.tpr -maxwarn 1 -n \$index_dir
nohup gmx mdrun -v -deffnm 2-eq -dds 0.8 -ntomp 32 -plumed \$plumed_dir
EOF

  # 赋予脚本执行权限并提交作业
  chmod +x "$dir/job_script.sh"
  sbatch "$dir/job_script.sh"
done