# Switchable Normalization for Object Detection



This repository contains the code of using Swithable Normalization (SN) in object detection, proposed by the paper 
["Differentiable Learning-to-Normalize via Switchable Normalization"](https://arxiv.org/abs/1806.10779).

This is a re-implementation of the experiments presented in the above paper by using [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch).
SN is easy to plug into different codebases. Please refer to the paper that evaluated SN in another two codebases [Faster R-CNN.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) and fb's [Caffe2 Detectron](https://github.com/facebookresearch/Detectron).


## Update

- 2018/7/26: The code and trained models of object detection by using SN are released!
- More results and models will be released soon. 

## Citation

You are encouraged to cite the following paper if you use SN in research or wish to refer to the baseline results.

```
@article{SwitchableNorm,
  title={Differentiable Learning-to-Normalize via Switchable Normalization},
  author={Ping Luo and Jiamin Ren and Zhanglin Peng},
  journal={arXiv:1806.10779},
  year={2018}
}
```

## Getting Started

Use git to clone this repository:

```
git clone https://github.com/switchablenorms/SwitchNorm_Detection.git
```

### Environment

The code is tested under the following configurations.

- Hardware: 1-8 GPUs (with at least 12G GPU memories)
- Software: CUDA 9.0, Python 3.6, PyTorch 0.4.0

### Installation & Data Preparation

Please check the [Requirements](https://github.com/roytseng-tw/Detectron.pytorch#requirements), [Compilation](https://github.com/roytseng-tw/Detectron.pytorch#compilation), and [Data Preparation](https://github.com/roytseng-tw/Detectron.pytorch#data-preparation) subsection in the repo [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch) for the details of installation.


### Pre-trained Models

Download the pretrained model and put them into the `{repo_root}/data/pretrained_model`.

#### ImageNet pre-trained models

The backbone models with SN pretrained on ImageNet are available in the format used by Detectron.pytorch and this repo.

- ResNet50v1+SN(8,2)  [[w/ fpn](https://drive.google.com/file/d/1b0ia4AnSGL__8pAKG-n4w3CSeYCjlNwe/view?usp=sharing), [w/o fpn](https://drive.google.com/file/d/1U_nftWTUqcjfprSY8w9FRYTjwWnD5nLN/view?usp=sharing)]
- ResNet50v1+SN(8,1)* [[w/ fpn](https://drive.google.com/file/d/16-6MGOHFkwyRImGv3EI71nsOSF9lMRH9/view?usp=sharing), [w/o fpn](https://drive.google.com/file/d/1gDy06UAPKxENPyqobqY164uIcKu9P8Xv/view?usp=sharing)]

\* For (8,1), SN contains IN and SN without BN, as BN is the same as IN in training.

For more pretrained models with SN, please refer to the repo of [switchablenorms/Switchable-Normalization](https://github.com/switchablenorms/Switchable-Normalization).
The following script converts the model trained from [Switchable-Normalization](https://github.com/switchablenorms/Switchable-Normalization) into a valid format used by the detection codebase.
```
./scripts/convert.sh
```
Input arguments:
```
usage: convert.py [-h] [--ckpt CKPT] [--dst DST] [--use-FPN]

```

**NOTE:** The paramater keys in pretrained model checkpoint must match the keys in backbone model **EXACTLY**. The prefix of the backbone key in the model with FPN is `Conv_Body.conv_body`, while it is `Conv_Body` in the model without FPN. The parameters in [w/ fpn] and [w/o fpn] backbone models with SN above are the same, except for the keys. You should load the correct pretrained model according to your detection architechure.


### Training

- All sn config files are provided in the folder `configs/sn_baselines/*.yaml`. 
- The training script with ResNet-50-sn backbone can be found here:
    - from scratch: `./scripts/train_scratch.sh`
    - fine tune: `./scripts/train_finetune.sh`



Optional arguments (see full input arguments via `python tools/train_net_step.py -h`):

```
  --dataset DATASET     Dataset to use
  --cfg CFG_FILE        Config file for training (and optionally testing)
  --bs BATCH_SIZE       Explicitly specify to overwrite the value comed from
                        cfg_file.
  --resume              resume to training on a checkpoint
  --load_ckpt LOAD_CKPT
                        checkpoint path to load
  --use_tfboard         Use tensorflow tensorboard to log training info

```
**NOTE:** There is something different about the resume mode between [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch) and this repo. In this repo, `--resume` allows the training to resume from the lastest checkpoint in `output_dir`, which is generated in the training process automatically. You do not need to re-assign `--load_ckpt`. If `output_dir` is empty, `--resume` does not apply.


### Evaluation

- The evaluation script with ResNet-50-sn backbone can be found here:
    - evaluate once: `./scripts/eval_once.sh`
    - evaluate continuous: `./scripts/eval_continuous.sh`

Optional arguments (see full input arguments via `python tools/test_net.py -h`):

```
  --dataset DATASET     training dataset
  --cfg CFG_FILE        optional config file
  --load_dir LOAD_DIR   path of checkpoint to load
  --load_ckpt LOAD_CKPT
                        path of checkpoint to load
  --use_tfboard         Use tensorflow tensorboard to log training info

```

`--load_dir` enables continuous evaluation when training. When `--load_dir` is assigned, the lastest checkpoint under this path will be evaluated, and wait for the next checkpoint exists after the current evaluation is completed. If you want to test one model once, you can load this model from `--load_ckpt`.


## Main Results

### End-to-End Faster & Mask R-CNN with SN

|     backbone|norm           |   type  | lr schd |  im/gpu |  box AP | mask AP | download            |
| ---           |  :---:      |  :---:  |  :---:  |  :---:  |  :---:  |  :---:  | :---                |
|R-50-C4   | SN-(8,32) | faster   |   2x    |    2    |  38.07  |    --     | [model](https://drive.google.com/file/d/1Eu9XCNWZBpy31wZcyYS49Wtr6y83BwVz/view?usp=sharing) &#124 [boxes](https://drive.google.com/file/d/1LrOSy09aiOM6A2CwFV7tpiyHLh3jhtpG/view?usp=sharing) |
|R-50-FPN  | SN-(8,2) |  faster   |   2x    |    2    |  39.10  |    --     | [model](https://drive.google.com/file/d/1ZfqIhrugfQI0gb7i10AGVp6_uTEnqOkY/view?usp=sharing) &#124 [boxes](https://drive.google.com/file/d/1zb-6zXBKywnk2I2MoWe88NS8QKZCGtLo/view?usp=sharing) |
|R-50-FPN | SN-(8, 1)| faster   |   2x    |    2    |  38.99  |    --     | [model](https://drive.google.com/file/d/1N6WiPK52JJmx5Cwv1wgXu0a9NZGtnPFx/view?usp=sharing) &#124 [boxes](https://drive.google.com/file/d/1p5ZpX0vThaOrNvczigd3yzdnC40EpF65/view?usp=sharing) |
|R-50-FPN | SN-(8, 1)|  mask   |   2x    |    2    |  41.01  |  36.12  | [model](https://drive.google.com/file/d/12SCgBn3GZOfaNfKcbqWdWUr8A21xqFc7/view?usp=sharing) &#124 [boxes](https://drive.google.com/file/d/1sOk9At5Ev9UvEesmVsBH3Lh0KCRS4DJl/view?usp=sharing) &#124 [masks](https://drive.google.com/file/d/115BqRMdYc5ycj9pcOkrerE0eE7f7DlM8/view?usp=sharing) |

### Comparisons with BN and GN

We also implement GN by using the same codebase in this repo for reference. Several results of BN are borrowed from fb's Detectron.

|     backbone           |  norm |  type  | lr schd |  im/gpu |  box AP | mask AP | reference |
| ---          |  :---:  |     :---:  |  :---:  |  :---:  |  :---:  |  :---:  | :---  |
|R-50-C4   | BN|  faster   |   2x    |    2    |  36.5  |    --     | [Detectron](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md) |
|R-50-C4^  | GN| faster   |   2x    |    2    |  37.3  |       --  | |
|R-50-FPN  | BN| faster   |   2x    |    2    |  37.9  |     --    | [Detectron](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md) |
|R-50-FPN^  | GN| faster   |   2x    |    2    |  38.3  |      --   | |
|R-50-FPN |BN|   mask   |   2x    |    2    |  38.6  |     34.5    | [Detectron](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md) |
|R-50-FPN^ |GN|  mask   |   2x    |    2    |  40.4  |  35.6  ||
|R-50-FPN |GN|  mask   |   2x    |    2    |  40.3  |  35.7  | [Detectron](https://github.com/facebookresearch/Detectron/blob/master/projects/GN/README.md)|

^ reproduced results based on this repository 


