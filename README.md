# GraphTransformer
An implementation for paper "AMR-to-text Generation with Graph Transformer" (accepted at TACL20)

coming soon...

## Requirements
* python 3.5
* tensorflow >= 1.5

## Command
python train.py --enc_layers=8 --dec_layers=6 --num_heads=2 --num_units=256 --emb_dim=300 --output_dir=model/gt1/ --train_dir=model/gt1/ ---use_copy=1 --batch_size=64 --dropout_rate=0.2 --gpu_device=2 --max_src_len=90 --max_tgt_len=90
