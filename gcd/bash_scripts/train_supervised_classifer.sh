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

export CUDA_VISIBLE_DEVICES=2

# Get unique log file
SAVE_DIR=/work/sagar/osr_novel_categories/dev_outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.cluster_and_classifier.train_supervised --dataset_name scars --label_smoothing=0.0 --num_workers 16 \
 --temperature=0.1 --grad_from_block 11 --epochs 200 --batch_size 256 --transform 'rand-augment' --rand_aug_m 15 --rand_aug_n 2 \
 > ${SAVE_DIR}logfile_${EXP_NUM}.out
#          --warmup_model_dir='/work/sagar/osr_novel_categories/metric_learn_gcd/log/(29.12.2021_|_48.847)/checkpoints/model_best.pt' \
