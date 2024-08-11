#!/bin/bash
#SBATCH --time=72:00:00
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

export CUDA_VISIBLE_DEVICES=3

# Get unique log file
SAVE_DIR=/work/sagar/osr_novel_categories/dev_outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.cluster_and_classifier.k_means_with_classifier_init --dataset scars --use_best_model 'True' \
 --warmup_model_exp_id '(07.02.2022_|_32.959)' --max_kmeans_iter 1 \
#> ${SAVE_DIR}logfile_${EXP_NUM}.out