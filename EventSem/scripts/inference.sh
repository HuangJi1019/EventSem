tfl_config_path=EventSem/scripts/charades_sta/train_vgg.sh
ckpt_path=EventSem/results_charades/charadesSTA-video_tef-demo-2025-06-25-22-39-31/model_best.ckpt
eval_split_name=test
eval_path=data/charades_sta/charades_sta_test_tvr_format_copy.jsonl
echo ${ckpt_path}
echo ${eval_split_name}
echo ${eval_path}
PYTHONPATH=$PYTHONPATH:. python EventSem/inference.py \
${tfl_config_path} \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:4}
