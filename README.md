# GraphTransformer
An implementation for paper "AMR-to-text Generation with Graph Transformer" (accepted at TACL20)

coming soon...

## Requirements
* python 3.5
* tensorflow >= 1.5

## Data Preprocessing
We use the tools in https://github.com/Cartus/DCGCN (without anonymization) to preprocessing the data. Because AMR corpus has LDC license, we cannot distribute the preprocessed data. We upload some examples for format reference. If you have the license, feel free to contact us for getting the preprocessed data.

We use pretrained Glove vectors. Due to the limitation of filesize, we only upload part of pretrained vectors. You can extract it from "glove.840B.300d.txt" with the vocab file.

## Train
python train.py --enc_layers=8 --dec_layers=6 --num_heads=2 --num_units=256 --emb_dim=300  --train_dir=model/gt1/ --use_copy=1 --batch_size=64 --dropout_rate=0.2 --gpu_device=0 --max_src_len=90 --max_tgt_len=90

## Test
python infer.py --enc_layers=8 --dec_layers=6 --num_heads=2 --num_units=256 --emb_dim=300  --train_dir=model/gt1/ --use_copy=1 --batch_size=64 --dropout_rate=0.2 --gpu_device=0 --max_src_len=90 --max_tgt_len=90

#The output file can be found in the "output" folder directory.

## Citation

@article{wang2020amr,
  title={AMR-To-Text Generation with Graph Transformer},
  author={Wang, Tianming and Wan, Xiaojun and Jin, Hanqi},
  journal={Transactions of the Association for Computational Linguistics},
  volume={8},
  pages={19--33},
  year={2020},
  publisher={MIT Press}
}
