# usage: bash EventSem/scripts/inference.sh <ckpt_path> <eval_path> [extra args...]
#   e.g. bash EventSem/scripts/inference.sh results_tacos/<run>/model_best.ckpt data/tacos/test.jsonl
# Model and data settings are read back from opt.json next to the checkpoint; --eval_path,
# --t_feat_dir and --semantic_t_feat_dir given here override the saved ones (see README, SRE).
ckpt_path=$1
eval_path=$2
PYTHONPATH=$PYTHONPATH:. python EventSem/inference.py \
data/MR.py \
--resume ${ckpt_path} \
--eval_split_name val \
--eval_path ${eval_path} \
"${@:3}"
