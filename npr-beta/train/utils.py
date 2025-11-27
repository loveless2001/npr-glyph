import time
import logging
from typing import List, Any, Dict, Union, Optional

import trl
import torch
from torch.utils.data import SequentialSampler

logger = logging.getLogger(__name__)


def add_and_init_special_tokens(
    model, tokenizer, new_special_tokens: Optional[List[str]] = None
):
    """
    Initialize structured reasoning tokens.
    """
    if new_special_tokens is None:
        # TOKEN HIERARCHY: guideline -> plan -> step -> takeaway
        new_special_tokens = [
            "<guideline>",
            "</guideline>",
            "<plan>",
            "</plan>",
            "<step>",
            "</step>",
            "<takeaway>",
            "</takeaway>",
        ]
        if model.config.model_type != "qwen3":
            new_special_tokens += ["<think>", "</think>"]

        logger.info(f"Using tokenzier of {model.config.model_type}")

    tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
    model.resize_token_embeddings(new_num_tokens=len(tokenizer), pad_to_multiple_of=64)

    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    tied = embed.weight.data_ptr() == lm_head.weight.data_ptr()

    for tok in new_special_tokens:
        base_word = tok.strip("<>")

        base_ids = tokenizer(base_word, add_special_tokens=False).input_ids

        if all(i != tokenizer.unk_token_id for i in base_ids):
            tokens_embed = embed(torch.tensor(base_ids, device=model.device))
            avg_tokens_embed = tokens_embed.mean(dim=0)
            special_id = tokenizer.convert_tokens_to_ids(tok)
            embed.weight.data[special_id] = avg_tokens_embed

            if not tied and lm_head.weight.shape == embed.weight.shape:
                avg_lm_logits = lm_head.weight.data[base_ids].mean(dim=0)
                lm_head.weight.data[special_id] = avg_lm_logits.clone()
        else:
            raise ValueError(
                f"Failed to initialize special token '{tok}': some base tokens are unknown."
                f" This indicates tokenizer incompatibility with the enhanced tag system."
            )


TAG_TOKEN_IDS = {
    # Guideline definition phase - sets the high-level reasoning objective
    "guideline_start": "<guideline>",
    "guideline_end": "</guideline>",
    # Plan planning phase - breaks down reasoning strategy
    "plan_start": "<plan>",
    "plan_end": "</plan>",
    # Step execution phase - parallel reasoning steps
    "step_start": "<step>",
    "step_end": "</step>",
    # Takeaway phase - synthesis of parallel steps
    "takeaway_start": "<takeaway>",
    "takeaway_end": "</takeaway>",
}


def construct_parallel_attention_mask(
    input_ids,
    tokenizer,
    device="cpu",
):
    """
    1. Unshared steps: Prevents different reasoning steps from attending to each other

    This enables fine-grained control over parallel reasoning isolation patterns.

    Args:
        input_ids: Token sequence to process
        tokenizer: Tokenizer with special tokens
        device: Compute device for tensor operations

    Returns:
        Float attention mask with -inf for masked positions, 0.0 for allowed attention
    """
    seq_len = len(input_ids)

    # FOUNDATION: Start with causal attention mask (lower triangular)
    bool_attention_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )  # Primary attention mask for parallel patterns

    # TOKEN ID RESOLUTION: Convert semantic tags to actual token IDs
    # ASSUMPTION: Single-token tags for efficient parsing (multi-token support would require
    # more complex sequence matching)
    guideline_start_id = tokenizer.convert_tokens_to_ids(
        TAG_TOKEN_IDS["guideline_start"]
    )
    guideline_end_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["guideline_end"])
    step_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["step_start"])
    step_end_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["step_end"])
    plan_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["plan_start"])
    plan_end_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["plan_end"])
    takeaway_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["takeaway_start"])
    takeaway_end_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["takeaway_end"])

    # locate the first "<|im_end|>" token
    i = input_ids.index(tokenizer.convert_tokens_to_ids("<|im_end|>")) + 1
    structure_stack = []  # Stack for tracking nested guideline/plan/step structures
    while i < seq_len:
        current_token_id = input_ids[i]

        # Strict nesting validation prevents malformed structure
        if current_token_id == guideline_start_id:
            if structure_stack:
                raise ValueError(
                    f"<guideline> found at index {i} while already in a block: {structure_stack[-1]}"
                )
            structure_stack.append(
                {
                    "type": "parallel",
                    "start_marker_index": i,
                    "step_spans": [],
                    "plans": [],
                }
            )
            i += 1
            continue

        if current_token_id in [plan_start_id, step_start_id]:
            # VALIDATION: Guidelines cannot be nested within other structures
            if not structure_stack or structure_stack[-1]["type"] != "parallel":
                raise ValueError(
                    f"Found <guideline>, <plan>, or <step> outside of a <takeaway> block at index {i}"
                )

            if current_token_id == plan_start_id:
                structure_stack.append({"type": "plan", "start_marker_index": i})

            if current_token_id == step_start_id:
                structure_stack.append({"type": "step", "start_marker_index": i})

            i += 1
            continue

        # Check </plan> end
        elif current_token_id == plan_end_id:
            plan_end_marker_index = i + 1

            # VALIDATION: Ensure proper plan block closure
            if not structure_stack or structure_stack[-1]["type"] != "plan":
                raise ValueError(
                    f"</plan> found at index {i} without a matching <plan> block on stack."
                )

            closed_plan_block = structure_stack.pop()

            # Find the nearest enclosing parallel block to add this plan span
            enclosing_parallel_block = None
            for block in reversed(structure_stack):
                if block["type"] == "parallel":
                    enclosing_parallel_block = block
                    break

            if enclosing_parallel_block is None:
                raise ValueError(
                    f"Plan block ending at {i} is not enclosed within any <guideline> block."
                )

            # SPAN TRACKING: Record plan boundaries for attention masking
            plan_start_marker_index = closed_plan_block["start_marker_index"]
            if plan_start_marker_index < plan_end_marker_index:
                enclosing_parallel_block["plans"].append(
                    (plan_start_marker_index, plan_end_marker_index)
                )

            i = plan_end_marker_index
            continue

        # Check </step> end - CRITICAL for parallel attention computation
        elif current_token_id == step_end_id:
            step_end_marker_index = i + 1

            # VALIDATION: Ensure proper step block closure
            if not structure_stack or structure_stack[-1]["type"] != "step":
                raise ValueError(
                    f"</step> found at index {i} without a matching <step> block on stack."
                )

            closed_step_block = structure_stack.pop()

            # Find the nearest enclosing guideline block to add this plan span
            enclosing_parallel_block = None
            for block in reversed(structure_stack):
                if block["type"] == "parallel":
                    enclosing_parallel_block = block
                    break

            if enclosing_parallel_block is None:
                raise ValueError(
                    f"Step block ending at {i} is not enclosed within any parallel block."
                )

            # SPAN TRACKING: Record step boundaries for attention masking
            step_start_marker_index = closed_step_block["start_marker_index"]
            if step_start_marker_index < step_end_marker_index:
                enclosing_parallel_block["step_spans"].append(
                    (step_start_marker_index, step_end_marker_index)
                )

            i = step_end_marker_index

            continue

        elif current_token_id == takeaway_start_id:
            takeaway_end_marker_index = i + 1

            if not structure_stack or structure_stack[-1]["type"] != "parallel":
                raise ValueError(
                    f"Parallel block found at index {i} without a matching parallel block."
                )

            closed_parallel_block = structure_stack.pop()

            step_spans_in_this_block = closed_parallel_block["step_spans"]

            # PRIMARY MASKING STRATEGIES: Apply configured attention isolation patterns
            num_steps = len(step_spans_in_this_block)

            # UNSHARED STEPS - Core parallel attention isolation
            # RATIONALE: Independent reasoning steps should not attend to each other
            if num_steps > 1:
                all_i_indices_to_mask = []  # Row indices for masking
                all_j_indices_to_mask = []  # Column indices for masking

                # PAIRWISE MASKING: Block attention between all step pairs
                for step_idx_a in range(num_steps):
                    start_a, end_a = step_spans_in_this_block[step_idx_a]
                    # SAFETY: Skip invalid spans
                    if start_a >= end_a:
                        continue

                    # Generate token indices for first step
                    indices_a = torch.arange(start_a, end_a, device=device)

                    for step_idx_b in range(step_idx_a + 1, num_steps):
                        start_b, end_b = step_spans_in_this_block[step_idx_b]
                        # SAFETY: Skip invalid spans
                        if start_b >= end_b:
                            continue

                        # Generate token indices for second step
                        indices_b = torch.arange(start_b, end_b, device=device)

                        # EFFICIENT MASKING: Use meshgrid for all (i,j) combinations
                        grid_i, grid_j = torch.meshgrid(
                            indices_a, indices_b, indexing="ij"
                        )

                        all_i_indices_to_mask.append(grid_i.flatten())
                        all_j_indices_to_mask.append(grid_j.flatten())

                # APPLY MASKING: Block cross-step attention
                if all_i_indices_to_mask:  # Only apply if there are indices to mask
                    final_i = torch.cat(all_i_indices_to_mask)
                    final_j = torch.cat(all_j_indices_to_mask)

                    # BIDIRECTIONAL MASKING: Block attention in both directions
                    # False = masked attention, True = allowed attention
                    bool_attention_mask[final_i, final_j] = False  # Step A -> Step B
                    bool_attention_mask[final_j, final_i] = False  # Step B -> Step A
            else:
                # SINGLE STEP: No cross-step masking needed
                pass

            i = takeaway_end_marker_index
            continue
        elif current_token_id == guideline_end_id:
            i += 1
            continue

        # ADVANCE PARSER: Move to next token if no structured reasoning tags matched
        i += 1

    if structure_stack:
        with open("logs/error-unclosed.log", "w") as f:
            f.write(str(structure_stack) + "\n")
            f.write(tokenizer.decode(input_ids) + "\n\n")
        unclosed_types = [block["type"] for block in structure_stack]
        raise ValueError(f"Input sequence ended with unclosed blocks: {unclosed_types}")

    # FINAL CONVERSION: Boolean mask to float attention scores
    # ATTENTION SEMANTICS: 0.0 = allowed attention, -inf = blocked attention
    # This format is compatible with PyTorch's scaled_dot_product_attention
    float_attention_mask = torch.full_like(
        bool_attention_mask, -torch.inf, dtype=torch.float
    )
    float_attention_mask = float_attention_mask.masked_fill(bool_attention_mask, 0.0)

    return float_attention_mask


def construct_parallel_position_ids(input_ids: List[int], tokenizer) -> List[int]:
    """Generates position IDs accounting for Plan and Step structure."""

    # Get special token IDs
    tag_ids = {
        tag: tokenizer.convert_tokens_to_ids(token)
        for tag, token in TAG_TOKEN_IDS.items()
    }

    position_ids = torch.arange(len(input_ids), device="cpu", dtype=torch.long)

    block_stack = []
    i = input_ids.index(tokenizer.convert_tokens_to_ids("<|im_end|>")) + 1
    while i < len(input_ids):
        token_id = input_ids[i]
        current_block_state = block_stack[-1] if block_stack else {}

        if token_id == tag_ids["guideline_start"]:
            block_stack.append(
                {
                    "guideline_end_pos_id": -1,
                    "max_step_len": 0,
                    "num_plans": 0,
                    "num_steps": 0,
                    "is_in_guideline": False,
                    "is_in_plan": False,
                    "is_in_step": False,
                }
            )
            block_stack[-1]["is_in_guideline"] = True
        elif (
            token_id == tag_ids["guideline_end"]
            and current_block_state
            and current_block_state["is_in_guideline"]
        ):
            current_block_state["guideline_end_pos_id"] = position_ids[i]
            current_block_state["is_in_guideline"] = False
        elif (
            token_id == tag_ids["step_start"]
            and current_block_state
            and current_block_state["guideline_end_pos_id"] != -1
        ):
            current_block_state["is_in_step"] = True
            position_ids[i:] -= position_ids[i] - (
                current_block_state["guideline_end_pos_id"] + 1
            )
        elif (
            token_id == tag_ids["step_end"]
            and current_block_state
            and current_block_state["is_in_step"]
        ):
            current_block_state["max_step_len"] = max(
                current_block_state["max_step_len"],
                position_ids[i] - current_block_state["guideline_end_pos_id"],
            )
            # Reset step state
            current_block_state["num_steps"] += 1
            current_block_state["is_in_step"] = False

        elif (
            token_id == tag_ids["takeaway_start"]
            and current_block_state
            and current_block_state["guideline_end_pos_id"] != -1
        ):
            position_ids[i:] -= position_ids[i] - (
                current_block_state["guideline_end_pos_id"]
                + current_block_state["max_step_len"]
                + 1
            )
            block_stack.pop()

        i += 1

    # Final check for unclosed blocks (optional, for robustness)
    if block_stack:
        # Depending on requirements, either raise error or handle gracefully
        raise ValueError("Input sequence ended with unclosed execution blocks.")

    # Sanity check length
    if len(position_ids) != len(input_ids):
        raise ValueError("Position ID generation length mismatch!")

    return position_ids


class NPRDataCollator(trl.DataCollatorForCompletionOnlyLM):
    def __init__(self, *args, max_length=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = max_length

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # First, we generate full attention masks and position ids for complete sequences
        attention_masks = []
        position_ids = []
        valid_examples = []
        for example in examples:
            # Get the complete input_ids (before any truncation)
            if isinstance(example, dict):
                input_ids = example["input_ids"]
            else:
                input_ids = example

            # Generate full attention mask and position ids based on complete sequence
            try:
                attention_mask = construct_parallel_attention_mask(
                    input_ids, self.tokenizer
                )
                position_id = construct_parallel_position_ids(input_ids, self.tokenizer)
            except ValueError as e:
                logger.error(e)
                text = self.tokenizer.decode(example["input_ids"])
                with open(f"logs/error-{time.time()}.log", "w") as f:
                    f.write(str(e) + "\n\n")
                    f.write(text)
                continue
            attention_masks.append(attention_mask)
            position_ids.append(position_id)
            valid_examples.append(example)

        # Apply the standard collation with truncated examples
        batch = super().torch_call(valid_examples)

        # Get the final sequence length after truncation
        final_seq_len = batch["input_ids"].shape[1]

        # Create custom attention masks and position ids with the same truncation
        batch["attention_mask"] = torch.zeros(
            len(valid_examples),
            1,
            final_seq_len,
            final_seq_len,
            dtype=torch.float,
            device="cpu",
        )
        batch["position_ids"] = torch.zeros(
            len(valid_examples), final_seq_len, dtype=torch.long, device="cpu"
        )

        for i in range(len(valid_examples)):
            # Apply the same truncation to attention mask and position ids
            batch["attention_mask"][i, 0] = attention_masks[i][
                :final_seq_len, :final_seq_len
            ]
            batch["position_ids"][i] = position_ids[i][:final_seq_len]
            batch["input_ids"][i] = batch["input_ids"][i][:final_seq_len]
            batch["labels"][i] = batch["labels"][i][:final_seq_len]

        return batch


class SequentialWarmupTrainer(trl.SFTTrainer):
    """
    SEQUENTIAL SFT TRAINER: Deterministic training order for reproducibility.
    """

    def _get_train_sampler(self, dataset) -> Optional[torch.utils.data.Sampler]:
        """Override sampler method to use sequential sampling instead of random sampling"""
        if self.train_dataset is None or not hasattr(self.train_dataset, "__len__"):
            return None

        # HYBRID APPROACH: Preserve length grouping when specified for efficiency
        if self.args.group_by_length:
            return super()._get_train_sampler(dataset)
        else:
            return SequentialSampler(self.train_dataset)
