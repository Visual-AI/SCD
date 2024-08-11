#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --job-name=supcon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=30
#SBATCH --exclude=gnodek1,gnodec1
#SBATCH --output=/work/khan/osr_novel_category/slurm_outputs/myLog-%j.out
#SBATCH --chdir=/users/khan/osr_novel_category
#--------------------------

# PYTHON="/users/khan/anaconda2/envs/py37/bin/python3.7"
PYTHON="/work/khan/envs/OSRR/bin/python3.7"

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file
SAVE_DIR=/work/khan/osr_novel_category/dev_outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.metric_learn.learn_metric --dataset_name imagenet_100 --batch_size 128 \
                --grad_from_block 11 --sup_con_weight 0.5 --epochs 200 --base_model vit_dino \
                --exp_root "/work/khan/osr_novel_category"
#> ${SAVE_DIR}logfile_${EXP_NUM}.out

# ${PYTHON} -m methods.metric_learn.learn_metric --dataset_name imagenet_100 --batch_size 128 --save_best_epoch 12 \
# --grad_from_block 11 --sup_con_weight 0.5 --contrastive_weight 1.0 --epochs 200 --base_model vit_dino --num_workers 16 \
# > ${SAVE_DIR}logfile_${EXP_NUM}.out
