#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Please provide a wkdir."
    exit 1
fi
wkdir=$1
if [ -z "$2" ]; then
    echo "Error: Please provide MD Temperature."
    exit 1
fi
temp=$2
if [ -z "$3" ]; then
    echo "Error: Please provide a GPU ID."
    exit 1
fi
gpu=$3
if [ -z "$4" ]; then
    echo "Error: Please provide a mdp_dir."
    exit 1
fi
mdp_dir=$4
if [ -z "$5" ]; then
    echo "Error: Please provide a file_name."
    exit 1
fi
file_name=$5
#更换wkdir
cd $wkdir
#初始文件处理

eq_mdp="run_eq_${2}_1.mdp"
md_mdp="run_md_${2}_1.mdp"
cp $mdp_dir/martini_eq_p.mdp $eq_mdp
cp $mdp_dir/martini_run.mdp $md_mdp

sed -i "s/TTTTTTTTTTTTTTTTT/$temp/g" $eq_mdp
sed -i "s/PPPPPPPPPPPPPPPPP/1/g" $eq_mdp
sed -i "s/TTTTTTTTTTTTTTTTT/$temp/g" $md_mdp
sed -i "s/PPPPPPPPPPPPPPPPP/1/g" $md_mdp

cp ../system-$file_name.top system.top
cp ../PET-$file_name.itp .

gmx grompp -p system.top -c start.gro   -f $eq_mdp  -o 2-eq.tpr -t start.cpt -maxwarn 1  || { echo "grompp failed"; exit 1; }
gmx mdrun -v -deffnm 2-eq -gpu_id $gpu -ntomp 8 -ntmpi 1

echo -e "17 0" | gmx energy -f 2-eq.edr -o density-npt.xvg

gmx grompp -p system.top -c 2-eq.gro   -f $md_mdp -o 3-run.tpr -t 2-eq.cpt -maxwarn 3

# gmx mdrun -v -deffnm 3-run -gpu_id $4 -ntomp 4 -ntmpi 1
