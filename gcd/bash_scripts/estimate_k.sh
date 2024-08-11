#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=b_search
#SBATCH --mem=50G
#SBATCH --cpus-per-task=15
#SBATCH --exclude=gnodek1,gnodec1
#SBATCH --output=/work/sagar/osr_novel_categories/slurm_outputs/myLog-%j.out
#SBATCH --chdir=/users/sagar/kai_collab/osr_novel_categories
#--------------------------

PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'

hostname

# Get unique log file
SAVE_DIR=/work/sagar/osr_novel_categories/dev_outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.estimate_k.estimate_k --max_classes 1000 --dataset_name herbarium_19 --search_mode other \
        > ${SAVE_DIR}logfile_${EXP_NUM}.out