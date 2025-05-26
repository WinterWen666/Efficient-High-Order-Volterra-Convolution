
## Usage examples
To train Resnet-50 on ImageNet-1k dataset with 8 GPUs:
```
python -m torch.distributed.run --nproc_per_node=8 train.py \
--data /path to data/  --net_type resnet \
--lr 0.6 \
--batch_size 128 \
--depth 50 \
--print-freq 10 \
--expname resnet-50 \
--dataset imagenet \
--epochs 100 \
```


