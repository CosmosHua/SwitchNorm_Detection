import os
import torch
import numpy as np
import argparse
from distutils.version import LooseVersion

def main(args):
    ckpt = torch.load(args.ckpt)
    prefix = 'Conv_Body.'
    if args.use_FPN:
        prefix += 'conv_body.'

    mapping = {}
    for sd in ckpt['state_dict'].keys():
        #print(sd)
        if 'module.sn1' in sd:
            mapping[prefix+'res1.'+sd[7:]] = sd
        elif 'module.conv1' in sd:
            mapping[prefix+'res1.'+sd[7:]] = sd
        elif 'module.fc' in sd:
            mapping[prefix+'classifier.'+sd[10:]] = sd
        else:
            mapping[prefix+'res'+str(int(sd[12])+1)+sd[13:]] = sd
    state_dict = {}
    for name in mapping:
        print('%s\t\t%s'%(name,mapping[name]))
        state_dict[name] = ckpt['state_dict'][mapping[name]]
    
    torch.save({'model':state_dict}, args.dst)
    print('Finish converting!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt', default='./ckpt.pth',
                        help='the checkpoint to be converted')
    parser.add_argument('--dst', default='./dst.pth',
                        help='the filename to output model with valid format')
    parser.add_argument('--use-FPN', action='store_true',
                        help='whether to use FPN')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:12} {}".format(key, val))

    assert os.path.isfile(args.ckpt)

    dst_dir = os.path.dirname(args.dst)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    main(args)
