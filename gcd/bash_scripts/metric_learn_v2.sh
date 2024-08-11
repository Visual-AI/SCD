#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --job-name=metric
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=30
#SBATCH --exclude=gnodek1,gnodec1
#SBATCH --output=/work/khan/osr_novel_category/slurm_outputs/myLog-%j.out
#SBATCH --chdir=/users/khan/osr_novel_category
#--------------------------

# PYTHON="/users/khan/anaconda2/envs/py37/bin//python3.7"
PYTHON="/work/khan/envs/OSRR/bin/python3.7"


hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file
SAVE_DIR=/work/khan/osr_novel_category/dev_outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.metric_learn.learn_metric_v2 --dataset_name herbarium_19 \
 --sup_con_weight_gt 0.5 --sup_con_weight_pseudo 0.5 --contrastive_weight 1.0 --loss_weight_mode 'trade_off' --loss_switch_T 50 \
 --epochs 200 --base_model vit_dino --num_workers 16 \
 --exp_root "/work/khan/osr_novel_category"
#> ${SAVE_DIR}logfile_${EXP_NUM}.out

# cifar100 --loss_switch_T 50 393804
# cifar10 --loss_switch_T 50 393805
# imagenet_100 --loss_switch_T 10 393806
# cub --loss_switch_T 50 393806
# scars --loss_switch_T 50 393807
# herbarium_19 --loss_switch_T 50 393808