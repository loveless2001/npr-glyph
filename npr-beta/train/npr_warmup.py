import os
import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional

import trl
import torch
import transformers
import torch.distributed as dist
from datasets import load_from_disk

from utils import (
    SequentialWarmupTrainer,
    add_and_init_special_tokens,
    NPRDataCollator,
)

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen3-8B")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="NativeParallelReasoner")
    wandb_entity: Optional[str] = field(default="")
    wandb_api_key: Optional[str] = field(default="")
    wandb_name: Optional[str] = field(default="NPR-Warmup-8B")
    train_file_path: Optional[str] = field(
        default="dataset/math/rejection_sampling/train"
    )

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project
        os.environ["WANDB_ENTITY"] = self.wandb_entity
        os.environ["WANDB_API_KEY"] = self.wandb_api_key
        os.environ["WANDB_NAME"] = self.wandb_name


def train():
    """Main training function for fine-tuning a language model using supervised fine-tuning (SFT)"""

    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "use_cache": False,
    }
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name, **kwargs
    )

    train_dataset = load_from_disk(config.train_file_path)
    dataset = {"train": train_dataset}

    # Dump dataset by split
    for split, ds in dataset.items():
        logging.info(f"Dataset split: {split}, size: {len(ds)}")
        datasplit_json = ds.to_dict()
        with open(f"dataset/tmp/{split}.json", "w") as f:
            json.dump(datasplit_json, f, indent=4, ensure_ascii=False)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, use_fast=True, max_length=config.block_size
    )

    instruction_template = "<|im_start|>user\n"
    response_template = "<|im_start|>assistant\n"
    tokenizer.pad_token = "<|fim_pad|>"

    add_and_init_special_tokens(model, tokenizer)

    training_collator = NPRDataCollator(
        instruction_template=instruction_template,
        response_template=response_template,
        max_length=config.block_size,
        tokenizer=tokenizer,
        mlm=False,
    )

    args.run_name = config.wandb_name
    args.dataset_text_field = "text"
    args.predict_with_generate = True

    # We will truncate the input in our collator, so we set a very large max_seq_length to avoid truncation in tokenizer
    trainer = SequentialWarmupTrainer(
        model,
        train_dataset=dataset["train"],
        args=args,
        processing_class=tokenizer,
        data_collator=training_collator,
        dataset_num_proc=8,
    )
    training_collator.trainer = trainer

    trainer.train()

    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()

    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logging.warning(f"Failed to destroy process group: {e}.")


if __name__ == "__main__":
    train()
