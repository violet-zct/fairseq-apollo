#! /bin/bash

module load cuda/11.3
conda activate sru

seed=42
LR=3e-4
WARMUP=16000
WD=0.1
SENT_REP_TYPE=mp
DATA=/private/home/chuntinz/checkpoint/data/lra/pf32
SAVE_ROOT=saved_models/lra/pathfinder
exp_name=pf32_sru_postnorm_lr${LR}_warmup${WARMUP}_wd${WD}_seed${seed}
SAVE=${SAVE_ROOT}/${exp_name}
rm -rf ${SAVE}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

TOTAL_NUM_UPDATES=250000
model=sru_lra_pf32

CUDA_VISIBLE_DEVICES=0 python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task lra-image --input-type image --pixel-normalization 0.5 0.5 \
    --sen-rep-type ${SENT_REP_TYPE} \
    --encoder-hidden-dim 256 --encoder-layers 8 --z-dim 64 --encoder-embed-dim 128 --attention-every-n-layers 1 \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer radam --lr ${LR} --radam-betas '(0.9, 0.999)' --radam-eps 1e-8 --clip-norm 1.0 --clip-mode 'total' \
    --dropout 0.15 --attention-dropout 0.15 --act-dropout 0.15 --weight-decay $WD \
    --batch-size 128 --sentence-avg --update-freq 1 --max-update ${TOTAL_NUM_UPDATES} \
    --lr-scheduler 'cosine' --warmup-updates ${WARMUP} --warmup-init-lr '1e-07' --keep-last-epochs 1 --max-sentences-valid 512 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 | tee -a ${SAVE}/log.txt

date
wait

python fairseq_cli/validate.py $DATA --task lra-image --batch-size 512 --valid-subset test --path ${SAVE}/checkpoint_best.pt | tee -a ${SAVE}/log.txt

python fairseq_cli/validate.py $DATA --task lra-image --batch-size 512 --valid-subset test --path ${SAVE}/checkpoint_last.pt | tee -a ${SAVE}/log.txt

