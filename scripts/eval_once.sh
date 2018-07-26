num_gpu=8
task='e2e_mask_rcnn_R-50-FPN_2x_sn_(8,1)'
ckpt='model/mask_rcnn_R-50-FPN_2x_sn_(8,1).pth'
#task='e2e_faster_rcnn_R-50-FPN_2x_sn_(8,1)'
#ckpt='model/faster_rcnn_R-50-FPN_2x_sn_(8,1).pth'
#task='e2e_faster_rcnn_R-50-FPN_2x_sn'
#ckpt='model/faster_rcnn_R-50-FPN_2x_sn.pth'
#task='e2e_faster_rcnn_R-50-C4_2x_sn'
#ckpt='model/faster_rcnn_R-50-C4_2x_sn.pth'

python -u tools/test_net.py \
  --dataset coco2017 \
  --cfg configs/sn_baselines/${task}.yaml \
  --load_ckpt ${ckpt} \
  --use_tfboard \
  --multi-gpu-testing \
  2>&1|tee Outputs/eval-${task}.log \
