num_gpu=8
batch_size=16
task='e2e_faster_rcnn_R-50-FPN_2x_sn_(8,1)'

python -u tools/train_net_step.py \
  --dataset coco2017 \
  --load_ckpt 'data/pretrained_model/resnet-sn-50-(8,1)-fpn.pth' \
  --cfg configs/sn_baselines/${task}.yaml \
  --use_tfboard \
  --bs ${batch_size} \
  --resume \
  2>&1|tee Outputs/${task}.log \

### mask rcnn R50 + FPN (pretrain model: R50-SN(8,1))
# task='e2e_mask_rcnn_R-50-FPN_2x_sn_(8,1)'
# --load_ckpt 'data/pretrained_model/resnet-sn-50-(8,1)-fpn.pth'

### faster rcnn R50 + FPN (pretrain model: R50-SN(8,1))
# task='e2e_faster_rcnn_R-50-FPN_2x_sn_(8,1)'
# --load_ckpt 'data/pretrained_model/resnet-sn-50-(8,1)-fpn.pth'

### faster rcnn R50 + FPN (pretrain model: R50-SN(8,2))
# task='e2e_faster_rcnn_R-50-FPN_2x_sn'
# --load_ckpt 'data/pretrained_model/resnet-sn-50-(8,2)-fpn.pth'

### faster rcnn R50 C4 (pretrain model: R50-SN(8,32))
# task='e2e_faster_rcnn_R-50-C4_2x_sn'
# --load_ckpt 'data/pretrained_model/resnet-sn-50-(8,32).pth'
