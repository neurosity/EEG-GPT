# test_train_gpt.py

import unittest
from unittest.mock import patch, MagicMock
from src.train_gpt import train

class TestTrainGPT(unittest.TestCase):

    @patch('src.train_gpt.make_trainer')
    @patch('src.train_gpt.make_model')
    @patch('src.train_gpt.EEGDataset')
    @patch('src.train_gpt.MotorImageryDataset')
    @patch('src.train_gpt.cv_split_bci')
    @patch('src.train_gpt.read_threshold_sub')
    def test_train(self, mock_read_threshold_sub, mock_cv_split_bci, mock_MotorImageryDataset, mock_EEGDataset, mock_make_model, mock_make_trainer):
        # Mock the functions and methods
        mock_read_threshold_sub.return_value = ['file1', 'file2', 'file3']
        mock_cv_split_bci.return_value = (['train_file1', 'train_file2'], ['test_file1', 'test_file2'])
        mock_MotorImageryDataset.return_value = MagicMock()
        mock_EEGDataset.return_value = MagicMock()
        mock_make_model.return_value = MagicMock()
        mock_make_trainer.return_value = MagicMock()

        # Call the function with a sample config
        config = {
            'training_style': 'decoding',
            'dst_data_path': '/path/to/data',
            'fold_i': 0,
            'chunk_len': 500,
            'num_chunks': 8,
            'chunk_ovlp': 50,
            'use_encoder': 'True',
            'do_normalization': 'True',
            'train_data_path': '/path/to/train/data',
            'seed': 1234,
            'set_seed': 'True',
            'do_train': 'True',
            'log_dir': '/path/to/log',
            'resume_from': None,
            'training_steps': 1000,
            'log_every_n_steps': 100,
            'eval_every_n_steps': 200,
            'fp16': 'True',
            'deepspeed': 'none',
            'run_name': 'test_run',
            'optim': 'AdamW',
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'adam_beta_1': 0.9,
            'adam_beta_2': 0.999,
            'adam_epsilon': 1e-8,
            'max_grad_norm': 1.0,
            'lr_scheduler': 'linear',
            'warmup_ratio': 0.1,
            'per_device_training_batch_size': 8,
            'per_device_validation_batch_size': 8,
            'num_workers': 4,
        }
        trainer = train(config)

        # Assert the mock functions and methods were called with the correct arguments
        mock_read_threshold_sub.assert_called_once_with('../inputs/sub_list2.csv', lower_bound=1000, upper_bound=1000000)
        mock_cv_split_bci.assert_called_once_with(['file1', 'file2', 'file3'][:18])
        mock_MotorImageryDataset.assert_called()
        mock_EEGDataset.assert_called()
        mock_make_model.assert_called()
        mock_make_trainer.assert_called()

if __name__ == '__main__':
    unittest.main()