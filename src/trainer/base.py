#!/usr/bin/env python3
from typing import Dict, List, Optional, Tuple

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
# from apex import amp
from tqdm.auto import tqdm
import torch
from torch import nn
from transformers import Trainer
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from transformers.integrations import (  # isort: split
    hp_params,
)
from transformers import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    TrainerState,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
)
from transformers.trainer_utils import (
    seed_worker
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_torch_tensorrt_fx_available,
    is_datasets_available,
    is_torch_tpu_available,
    is_torchdynamo_available,
    logging,
)
from transformers.utils.generic import ContextManagers

logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class Trainer(Trainer):
    def __init__(
        self,
        is_deepspeed: bool = False,
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.name = "Trainer"
        self.is_deepspeed = is_deepspeed
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        # if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        #     train_dataset = self._remove_unused_columns(train_dataset, description="training")
        # else:
        # data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        # pdb.set_trace()
        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            # if self.args.world_size > 1:
            #     train_dataset = IterableDatasetShard(
            #         train_dataset,
            #         batch_size=self._train_batch_size,
            #         drop_last=self.args.dataloader_drop_last,
            #         num_processes=self.args.world_size,
            #         process_index=self.args.process_index,
            #     )
            print("iterable dataset")
            # pdb.set_trace()
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                # collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=True,
            )

        train_sampler = self._get_train_sampler()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            # collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        print(next(iter(train_loader))['inputs'].shape)
        return train_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        # if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
        #     eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        # else:
        #     data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                # collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            # collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        # data_collator = self.data_collator

        if isinstance(test_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                test_dataset = IterableDatasetShard(
                    test_dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                test_dataset,
                batch_size=self.args.eval_batch_size,
                # collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            # collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def prediction_step(
        self,
        model,
        batch,
        prediction_loss_only: bool = False,
        ignore_keys: Optional[List[str]] = None
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        batch = self._move_batch_to_device(batch=batch)
        
        with torch.no_grad():
            (loss, outputs) = self.compute_loss(
                model=model,
                batch=batch,
                return_outputs=True
            )

        if not prediction_loss_only and 'labels' in batch:
            return (loss, outputs['decoding_logits'], batch['labels'])
        
        else:
            return (loss, outputs, None)

    def compute_loss(
        self,
        model,
        batch,
        return_outputs=False,
        **kwargs
        ):
        batch = self._move_batch_to_device(batch=batch)

        if isinstance(
            model,
            (
                torch.nn.DataParallel, 
                torch.nn.parallel.DistributedDataParallel
            )
        ) or self.is_deepspeed:
            (losses, outputs) = model.module.compute_loss(
                batch=batch,
                return_outputs=True
            )
        
        else:
            (losses, outputs) = model.compute_loss(
                batch=batch,
                return_outputs=True
            )
        
        loss = losses['loss'] if 'loss' in losses.keys() else sum(losses.values())
        
        return (loss, outputs) if return_outputs else loss

    def _move_batch_to_device(
        self,
        batch
        ) -> Dict[str, torch.tensor]:
        batch = self._prepare_inputs(batch)
        
        if "labels" in batch:
            batch["labels"] = batch["labels"].to(torch.long).to(batch["inputs"].device)
        
        return self._prepare_inputs(batch)
