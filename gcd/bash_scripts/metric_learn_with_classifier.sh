#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=lin_supcon
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

# SLURM commands: #SBATCH --output=/work/sagar/osr_novel_categories/slurm_outputs/myLog-%j.out
# SLURM commands: #SBATCH --array=0-2

# Get unique log file,
SAVE_DIR=/work/sagar/osr_novel_categories/dev_outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

#dataset_names=('cifar10' 'cifar100' 'imagenet_100')
#dataset=${dataset_names[$SLURM_ARRAY_TASK_ID]}

${PYTHON} -m methods.metric_learn_with_classifier.learn_metric_with_classifier \
            --dataset_name scars \
            --batch_size 128 \
            --grad_from_block 11 \
            --epochs 200 \
            --ss_k_means_interval 1 \
            --base_model vit_dino \
            --num_workers 16 \
            --use_custom_fgvc_splits 'False' \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --max_ent_loss_weight 1.0 \
            --ce_loss_weight 0.0 \
            --contrast_unlabel_only 'True' \
            --transform 'imagenet' \
            --lr 0.1 \
            --eval_funcs 'v1' 'v2' 'v3'
#> ${SAVE_DIR}logfile_${EXP_NUM}.out