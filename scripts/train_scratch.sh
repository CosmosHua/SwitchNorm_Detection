num_gpu=8
batch_size=16
task='scratch_e2e_mask_rcnn_R-50-FPN_3x_sn'

python -u tools/train_net_step.py \
  --dataset coco2017 \
  --cfg configs/sn_baselines/${task}.yaml \
  --use_tfboard \
  --bs ${batch_size} \
  --resume \
  2>&1|tee Outputs/${task}.log \
