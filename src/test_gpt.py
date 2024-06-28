"""
test.py

Testing of models based on given data. See get_args() for
details on command line arguments

Give it n chunks, n<32.. test the n+1 chunk...
"""

import os
from typing import Dict
from safetensors import safe_open
from safetensors.torch import load_file
from train_gpt import make_model, get_config
from batcher.downstream_dataset import EEGDataset
import torch
from torch.utils.data import DataLoader
from numpy import random

if __name__ == '__main__':

    config = dict(get_config())
    model = make_model(config)

    root_path = os.getcwd()
    
    # results/models/upstream/32clen2_embed1024/model_final/model.safetensors
    model_path = os.path.join(os.getcwd(), config["log_dir"], "model_final")

    state_dict = load_file(model_path + "/model.safetensors")
    
    model.load_state_dict(state_dict=state_dict)

    # input_dataset = {'inputs': torch.ones(size=(1,32,68,256)),
    #                  'attention_mask': torch.zeros(size=(1,32,68,256)).numpy(),
    #                  'seq_on': 0,
    #                  'seq_len': 32
    #                 }
    
    
    train_data_path = config["train_data_path"]
    files = [os.path.join(train_data_path, f) for f in os.listdir(train_data_path) if f.endswith('.npy')]

    # # Remove files less than 0.2 MB
    files = [f for f in files if os.path.getsize(f) >= 0.2 * 1024 * 1024]

    random.shuffle(files)
    num_files = len(files)
    split_index = int(num_files * 0.9)
    train_files = files[:split_index]
    validation_files = files[split_index:]

    test_dataset = EEGDataset(validation_files, sample_keys=[
        'inputs',
        'attention_mask'
    ], chunk_len=config["chunk_len"],num_chunks=config["num_chunks"], ovlp=config["chunk_ovlp"], root_path=root_path, gpt_only=not config["use_encoder"], normalization=config["do_normalization"])

    model.eval()

    sample = DataLoader(test_dataset, batch_size=1)
    output = model(next(iter(sample)), prep_batch=True)

    print("Predictions: ", output['outputs'])
    print("Shape: ", output['outputs'].shape)