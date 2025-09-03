## VSLA-CLIP++: A Large-Scale Benchmark and Part-Level Feature Alignment for Cross-Platform Video Person ReID
### Installation

```
conda create -n vslaclip python=3.8
conda activate vslaclip
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install yacs
pip install timm
pip install scikit-image
pip install tqdm
pip install ftfy
pip install regex
```

### Training

For example, if you want to run for the ls-vid, you need to modify the config file to

```
DATASETS:
   NAMES: ('lsvid')
   ROOT_DIR: ('your_dataset_dir')
OUTPUT_DIR: 'your_output_dir'
```
Then, if you want to use weight of [VIFI-CLIP](https://github.com/muzairkhattak/ViFi-CLIP) to initialize model, you need to down the weight form [link](https://github.com/muzairkhattak/ViFi-CLIP) and modify config file as:

```
MODEL:
  VIFI_WEIGHT : 'your_dataset_dir/vifi_weight.pth'
  USE_VIFI_WEIGHT : True
```

if you want to run VSLA-CLIP++:

```
CUDA_VISIBLE_DEVICES=0 python train_vsla_part.py --config_file configs/vsla++/vsla++_g2av2.yml
```

### Evaluation

For example, if you want to test VSLA-CLIP++ for LS-VID

```
CUDA_VISIBLE_DEVICES=0 python test_vsla_part.py --config_file 'your_config_file' TEST.WEIGHT 'your_trained_checkpoints_path/ViT-B-16_120.pth'
```

### Citation
```
@inproceedings{zhang2024cross,
  title={Cross-platform video person reid: A new benchmark dataset and adaptation approach},
  author={Zhang, Shizhou and Luo, Wenlong and Cheng, De and Yang, Qingchun and Ran, Lingyan and Xing, Yinghui and Zhang, Yanning},
  booktitle={European Conference on Computer Vision},
  pages={270--287},
  year={2024},
  organization={Springer}
}
```

### Acknowledgement

Codebase from [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID), [TransReID](https://github.com/damo-cv/TransReID), [CLIP](https://github.com/openai/CLIP), and [CoOp](https://github.com/KaiyangZhou/CoOp).
