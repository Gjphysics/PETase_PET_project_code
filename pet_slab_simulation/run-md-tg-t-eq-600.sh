#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Please provide a wkdir."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: Please provide MD Temperature."
    exit 1
fi

if [ -z "$3" ]; then
    echo "Error: Please provide a GPU ID."
    exit 1
fi

if [ -z "$4" ]; then
    echo "Error: Please provide a mdp_dir."
    exit 1
fi

if [ -z "$5" ]; then
    echo "Error: Please provide a file_name."
    exit 1
fi
#更换wkdir
cd $1
#初始文件处理
cp $4/martini_em.mdp run_em.mdp
cp $4/martini_eq_p.mdp run_eq_${2}_1.mdp
cp $4/martini_run.mdp run_md_${2}_1.mdp

sed -i "s/TTTTTTTTTTTTTTTTT/$2/g" run_eq_${2}_1.mdp
sed -i "s/PPPPPPPPPPPPPPPPP/1/g" run_eq_${2}_1.mdp
sed -i "s/TTTTTTTTTTTTTTTTT/$2/g" run_md_${2}_1.mdp
sed -i "s/PPPPPPPPPPPPPPPPP/1/g" run_md_${2}_1.mdp

cp ../system-$5.top system.top
cp ../PET-$5.itp .

gmx grompp -p system.top -c start.gro -f run_em.mdp  -o 1-min.tpr -maxwarn 1
gmx mdrun -v -deffnm 1-min -gpu_id $3 -ntomp 8 -ntmpi 1

gmx grompp -p system.top -c 1-min.gro   -f run_eq_${2}_1.mdp  -o 2-eq.tpr -maxwarn 1
gmx mdrun -v -deffnm 2-eq -gpu_id $3 -ntomp 8 -ntmpi 1

echo -e "17 0" | gmx energy -f 2-eq.edr -o density-npt.xvg

gmx grompp -p system.top -c 2-eq.gro   -f run_md_${2}_1.mdp -o 3-run.tpr -t 2-eq.cpt -maxwarn 3

# gmx mdrun -v -deffnm 3-run -gpu_id $4 -ntomp 4 -ntmpi 1
