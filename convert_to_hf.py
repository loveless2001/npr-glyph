from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import fire
from collections import defaultdict


def main(
    fsdp_checkpoint_path, huggingface_model_path, output_path
):
    """
    Convert FSDP checkpoint to HuggingFace checkpoint
    Args:
        fsdp_checkpoint_path: path to the FSDP checkpoint
        huggingface_model_path: path to the HuggingFace model
        output_path: path to save the converted checkpoint
    Usage:
        python convert_to_hf.py \
            verl/experiments/ckpts/project_name/exp_name/global_step_x/actor \
            <STAGE_2_MODEL_PATH> \
            <TARGET_HF_MODEL_PATH>
    """
    state_dict = defaultdict(list)
    world_size = 8

    for rank in range(int(world_size)):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print("loading", filepath)
        this_state_dict = torch.load(filepath, weights_only=False, map_location=torch.device('cpu'))
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    fire.Fire(main)
