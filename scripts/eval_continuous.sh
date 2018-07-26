num_gpu=1
task='e2e_mask_rcnn_R-50-FPN_2x_sn_(8,1)'

python -u tools/test_net.py \
  --dataset coco2017 \
  --cfg configs/sn_baselines/${task}.yaml \
  --load_dir Outputs/${task}/ckpt/ \
  --use_tfboard \
  2>&1|tee Outputs/eval-${task}.log \
#  --multi-gpu-testing \
