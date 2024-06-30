python -m torch.distributed.launch --nproc_per_node=2 \
    src/train_gpt.py \
    --training-steps=50000 \
    --eval_every_n_steps=100 \
    --log-every-n-steps=1 \
    --per-device-training-batch-size=32 \
    --per-device-validation-batch-size=32 \
    --num-workers=16 \
    --num_chunks=32 \
    --chunk_len=500 \
    --chunk_ovlp=50 \
    --num-hidden-layers=6 \
    --num-encoder-layers=6 \
    --run-name=32clen2_embed1024_multi_gpu \
    --training-style=CSM_causal \
    --embedding-dim=1024 \
    --train-data-path=data/npy_tuh_eeg \
    --verbose=True