# LPEFormer
> [**LPEFormer: A transformer-based method for identification of human-virus protein-protein interactions**]

## Getting Started

### Installation
Setup conda environment:
```bash
# Create environment
conda create -n trans-ppi python=3.8 -y
conda activate trans-ppi

# Instaill requirements
conda install pytorch==1.8.1 torchvision==0.9.1 -c pytorch -y

```

### Training
To train LPEFormer for cross-validation:
```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --add_fea data_full/add_feature.json --output_dir outputs/pt5_rel_pos_folder_1/ --batch_size 256 \
    --folder_num 1 --width 320 --epochs 20 --lr 8e-4 --warmup-lr 5e-4 --method self_attention --relative_pos
```


