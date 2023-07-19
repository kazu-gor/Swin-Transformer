PYTHONPATH="./swin-transformer/:$PYTHONPATH" python3 -m torch.distributed.launch main.py --cfg ./configs/swin/swin_base_patch4_window7_224.yaml --local_rank 0
