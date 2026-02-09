dset_name=tacos
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip
results_root=results_tacos
exp_id=50005


######## data paths
train_path=data/tacos/train.jsonl
eval_path=data/tacos/test_SRE.jsonl
 # eval_path=data/tacos/test.jsonl
eval_split_name=val

######## setup video+text features
feat_root=datasets/tacos

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
bsz=32 #32
max_v_l=-1
max_q_l=-1
eval_epoch=1
weight_decay=0.0001
eval_bsz=1

enc_layers=3
t2v_layers=8
dummy_layers=3
num_dummies=35
kernel_size=5
num_conv_layers=2
num_mlp_layers=5

lw_reg=1
lw_cls=5
lw_sal=0.05
lw_saliency=0.8
label_loss_coef=4
nms_type=normal

PYTHONPATH=$PYTHONPATH:. python EventSem/train.py \
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
--n_epoch 400 \
--lr_drop 200 \
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
--nms_type ${nms_type} \
--clip_length 2 \
--lr 3e-4 \
--dropout 0 \
--score_weight 0.5 \
--event_sim_threshold 0.25 \
--semantic_t_feat_dir "datasets/semantic_embeddings/tacos-token-level" \
--n_semantic_proj 2 \
${@:1}
