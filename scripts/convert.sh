PRETRAINED_CKPT='ResNet50v1+SN(8,32)-77.49.pth'
DST_FILENAME='data/pretrained_model/resnet-sn-50-(8,32).pth'

python -u tools/convert.py \
  --ckpt ${PRETRAINED_CKPT} \
  --dst ${DST_FILENAME} \
#  --use-FPN
