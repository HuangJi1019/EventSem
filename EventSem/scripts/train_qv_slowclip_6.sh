#!/bin/bash
. /opt/gridware/depots/54e7fb3c/el8/pkg/apps/anaconda3/2024.06/bin/etc/profile.d/conda.sh
conda activate FlashVTG
dset_name=hl
ctx_mode=video_tef
v_feat_types=clip_slowfast
t_feat_type=clip
results_root=results_all_qvhighlight
exp_id=qv_slowclip_512

######## data paths
train_path=/users/40448930/ji_code/FlashVTG/data/highlight_train_release.jsonl
# eval_path=data/highlight_val_release.jsonl
eval_path=/users/40448930/ji_code/FlashVTG/data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=/users/40448930/ji_code/FlashVTG/datasets/qvhighlight

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=64
max_v_l=75
max_q_l=32
eval_epoch=1
weight_decay=0.005
eval_bsz=1

enc_layers=3
t2v_layers=6
dummy_layers=2
num_dummies=40
kernel_size=5
num_conv_layers=1
num_mlp_layers=5

lw_reg=1
lw_cls=5
lw_sal=0.1
lw_saliency=0.8
label_loss_coef=4

PYTHONPATH=$PYTHONPATH:. python /users/40448930/ji_code/FlashVTG/FlashVTG/train.py \
data/MR.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--enc_layers ${enc_layers} \
--results_root ${results_root} \
--bsz ${bsz} \
--exp_id ${exp_id} \
--t2v_layers ${t2v_layers} \
--dummy_layers ${dummy_layers} \
--max_v_l ${max_v_l} \
--max_q_l ${max_q_l} \
--n_epoch 200 \
--lr_drop 50 \
--eval_epoch ${eval_epoch} \
--wd ${weight_decay} \
--eval_bsz ${eval_bsz} \
--lw_reg ${lw_reg} \
--lw_cls ${lw_cls} \
--lw_sal ${lw_sal} \
--lw_saliency ${lw_saliency} \
--nms_thd 0.7 \
--use_neg \
--num_dummies ${num_dummies} \
--kernel_size ${kernel_size} \
--num_conv_layers ${num_conv_layers} \
--num_mlp_layers ${num_mlp_layers} \
--label_loss_coef ${label_loss_coef} \
--use_SRM \
--clip_length 2.0 \
--max_event_spans 40 \
--lw_l1 0.1 \
--lw_giou 0.1 \
--score_weight 0.01 \
--event_sim_threshold 0.2 \
--lr 4e-4 \
--span_width_threshold 0.2 \
--semantic_t_feat_dir "/users/40448930/ji_code/FlashVTG/datasets/semantic_embeddings/qv_highlight_token_level_new" \
--n_semantic_proj 6 \
--gate -2 \
--sim_sharpness 5 \
--resume "/users/40448930/ji_code/FlashVTG/results_all_qvhighlight/hl-video_tef-qv_slowclip_512-2025-06-02-20-38-00/model_best.ckpt" \
# --resume_all \
# --resume "/users/40448930/ji_code/FlashVTG/results_all_qvhighlight/hl-video_tef-qv_slowclip_512-2025-06-02-20-38-00/model_best.ckpt" \

${@:1}
# 18,6,21