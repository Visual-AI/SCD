#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --job-name=auto_nov
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=30
#SBATCH --output=/work/sagar/osr_novel_categories/slurm_outputs/myLog-%j.out
#SBATCH --chdir=/users/sagar/kai_collab/osr_novel_categories
#--------------------------


#   ##SBATCH --exclude=gnodek1,gnodec1,gnodeb4

PYTHON='/users/sagar/miniconda3/envs/open_world_prototype/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=1

# Get unique log file
SAVE_DIR=/work/sagar/osr_novel_categories/dev_outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

DATASET='imagenet_100'
GRAD_FROM_BLOCK=11
WEIGHT_DECAY=1e-4
MODE='test'

if [ $DATASET == 'cifar10' ]; then

  ${PYTHON} -m methods.baselines.autonovel_new_setting --exp_root='/work/sagar/osr_novel_categories/' \
          --lr 0.1 \
          --gamma 0.1 \
          --weight_decay ${WEIGHT_DECAY} \
          --step_size 170 \
          --batch_size 128 \
          --epochs 200 \
          --rampup_length 50 \
          --rampup_coefficient 5.0 \
          --dataset_name $DATASET \
          --seed 0 \
          --num_workers=16 \
          --ce_loss 1.0 \
          --model vit_dino \
          --grad_from_block $GRAD_FROM_BLOCK \
          --eval_funcs 'v2' 'v1' \
          --mode $MODE \
          --mode train #> ${SAVE_DIR}logfile_${EXP_NUM}.out
fi

if [ $DATASET == 'cifar100' ] || [ $DATASET == 'imagenet_100' ]; then

  ${PYTHON} -m methods.baselines.autonovel_new_setting --exp_root='/work/sagar/osr_novel_categories/' \
          --lr 0.1 \
          --gamma 0.1 \
          --weight_decay ${WEIGHT_DECAY} \
          --step_size 170 \
          --batch_size 128 \
          --epochs 200 \
          --rampup_length 150 \
          --rampup_coefficient 50 \
          --dataset_name $DATASET \
          --seed 0 \
          --num_workers=16 \
          --ce_loss 1.0 \
          --model vit_dino \
          --grad_from_block $GRAD_FROM_BLOCK \
          --eval_funcs 'v2' 'v1' \
          --mode $MODE \
          --mode train #> ${SAVE_DIR}logfile_${EXP_NUM}.out

fi

if [ $DATASET == 'scars' ] || [ $DATASET == 'cub' ] || [ $DATASET == 'herbarium_19' ]; then

  ${PYTHON} -m methods.baselines.autonovel_new_setting --exp_root='/work/sagar/osr_novel_categories/' \
          --lr 0.001 \
          --gamma 0.1 \
          --weight_decay ${WEIGHT_DECAY} \
          --step_size 170 \
          --batch_size 128 \
          --epochs 200 \
          --rampup_length 150 \
          --rampup_coefficient 50 \
          --dataset_name $DATASET \
          --seed 0 \
          --num_workers=16 \
          --ce_loss 1.0 \
          --model vit_dino \
          --grad_from_block $GRAD_FROM_BLOCK \
          --eval_funcs 'v2' 'v1' \
          --mode $MODE \
          --mode train #> ${SAVE_DIR}logfile_${EXP_NUM}.out

fi
