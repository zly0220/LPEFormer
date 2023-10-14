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
pip install bio-embeddings[all]

```

### Prepare datasets
[datasets]
[add_feature](https://drive.google.com/file/d/1fh3m2QNN2SNZorIioSPvBQBTrrF45eys/view?usp=drive_link)
```none
${DATA_ROOT}
├── 1
│   ├── cv_test_1.csv
|   ├── cv_train_1.csv
├── 2
    ...
├── 5
├── add_feature.json
├── pos_sample.csv
├── neg_sample.csv
├── independent.csv
```

### Training
To train LPEFormer for cross-validation:
```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py \
--add_fea data_full/add_feature.json --output_dir /path/to/save_dir/ --batch_size 256 \
--folder_num 1 --width 320 --epochs 20 --lr 8e-4 --warmup-lr 5e-4 --method self_attention --relative_pos
```
- `--nproc_per_node` : number of gpus
- `--batch_size` : batch_size_per_gpu

### Testing
To test LPEFormer for cross-validation:
```bash
python -m torch.distributed.launch --nproc_per_node=4 eval.py \
--add_fea /path/to/add_feature.json --output_dir /path/to/save_dir/ --batch_size 256 \
--folder_num 1 --width 320 --epochs 20 --lr 8e-4 --warmup-lr 5e-4 --method self_attention --relative_pos --ckpt /path/to/save_checkpoint_file
```

