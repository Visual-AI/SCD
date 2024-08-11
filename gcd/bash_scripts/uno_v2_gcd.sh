#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=uno
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=30
#SBATCH --exclude=gnodek1,gnodel1
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

${PYTHON} -m methods.baselines.uno_v2_new_setting --exp_root='/work/sagar/osr_novel_categories/' \
        --base_lr 0.1 \
        --min_lr 0.001 \
        --temperature 0.1 \
        --weight_decay_opt 1e-4 \
        --batch_size 256 \
        --epochs 200 \
        --dataset_name imagenet_100 \
        --seed 0 \
        --model_name vit_dino \
        --num_workers=8 \
        --prop_train_labels=0.5 \
        --mode test \
        --pretrain_dir 'vit_dino' \
        --grad_from_block 11 \
        --eval_funcs 'v1' 'v2' 'v3' \
        --pretrain_path '/work/sagar/osr_novel_categories/uno_v2_gcd/log/(28.01.2022_|_55.096)/checkpoints/model_best.pt'
#        --mode train > ${SAVE_DIR}logfile_${EXP_NUM}.out