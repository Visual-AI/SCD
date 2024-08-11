#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --job-name=new_setting
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=30
#SBATCH --exclude=gnodek1,gnodec1
#SBATCH --output=/work/sagar/osr_novel_categories/slurm_outputs/myLog-%j.out
#SBATCH --chdir=/users/sagar/kai_collab/osr_novel_categories
#--------------------------

PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file
SAVE_DIR=/work/sagar/osr_novel_categories/dev_outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.clustering.extract_features --dataset scars --use_best_model 'True' \
 --warmup_model_dir '/work/sagar/osr_novel_categories/metric_learn_with_linear/log/(07.02.2022_|_32.959)/checkpoints/model.pt' \
#> ${SAVE_DIR}logfile_${EXP_NUM}.out