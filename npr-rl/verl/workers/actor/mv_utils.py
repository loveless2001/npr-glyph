import torch

from typing import List


TAG_TOKEN_IDS = {
    "guideline_start": "<guideline>",
    "guideline_end": "</guideline>",
    "plan_start": "<plan>",
    "plan_end": "</plan>",
    "step_start": "<step>",
    "step_end": "</step>",
    "takeaway_start": "<takeaway>",
    "takeaway_end": "</takeaway>",
}


def generate_parallel_attention_mask(
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
    """

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    seq_len = len(input_ids)
    # FOUNDATION: Start with causal attention mask (lower triangular)
    # True = allowed attention, False = masked attention
    bool_attention_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )  # Primary attention mask for parallel patterns

    step_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["step_start"])
    step_end_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["step_end"])
    takeaway_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["takeaway_start"])

    # locate the first "<|im_end|>" token
    i = input_ids.index(tokenizer.convert_tokens_to_ids("<|im_end|>")) + 1
    structure_stack = []
    step_spans = []

    while i < seq_len:
        current_token_id = input_ids[i]

        if current_token_id == step_start_id:
            if structure_stack:
                return None, True

            structure_stack.append({"type": "step", "start_marker_index": i})

            i += 1
            continue

        # Check </step> end - CRITICAL for parallel attention computation
        elif current_token_id == step_end_id:
            step_end_marker_index = i + 1

            if not structure_stack or structure_stack[-1]["type"] != "step":
                return None, True

            closed_step_block = structure_stack.pop()

            # SPAN TRACKING: Record step boundaries for attention masking
            step_start_marker_index = closed_step_block["start_marker_index"]
            if step_start_marker_index < step_end_marker_index:
                step_spans.append((step_start_marker_index, step_end_marker_index))
            else:
                return None, True

            i = step_end_marker_index
            continue

        elif current_token_id == takeaway_start_id:
            if structure_stack:
                return None, True

            num_steps = len(step_spans)

            # UNSHARED steps - Core parallel attention isolation
            # RATIONALE: Independent reasoning steps should not attend to each other
            # This is the KEY innovation enabling true takeaway reasoning
            if num_steps > 1:
                all_i_indices_to_mask = []  # Row indices for masking
                all_j_indices_to_mask = []  # Column indices for masking

                # PAIRWISE MASKING: Block attention between all step pairs
                for step_idx_a in range(num_steps):
                    start_a, end_a = step_spans[step_idx_a]
                    # SAFETY: Skip invalid spans
                    if start_a >= end_a:
                        return None, True

                    # Generate token indices for first step
                    indices_a = torch.arange(start_a, end_a, device=device)

                    for step_idx_b in range(step_idx_a + 1, num_steps):
                        start_b, end_b = step_spans[step_idx_b]
                        # SAFETY: Skip invalid spans
                        if start_b >= end_b:
                            return None, True

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
                    bool_attention_mask[final_i, final_j] = False  # step A -> step B
                    bool_attention_mask[final_j, final_i] = False  # step B -> step A
            else:
                # SINGLE step: No cross-step masking needed
                pass

            i += 1
            step_spans = []
            continue

        # ADVANCE PARSER: Move to next token if no structured reasoning tags matched
        i += 1

    if structure_stack:
        return None, True

    # FINAL CONVERSION: Boolean mask to float attention scores
    # ATTENTION SEMANTICS: 0.0 = allowed attention, -inf = blocked attention
    # This format is compatible with PyTorch's scaled_dot_product_attention
    float_attention_mask = torch.full_like(
        bool_attention_mask, -torch.inf, dtype=torch.float
    )
    float_attention_mask = float_attention_mask.masked_fill(bool_attention_mask, 0.0)

    return float_attention_mask, False


def generate_parallel_position_ids(input_ids: List[int], tokenizer) -> List[int]:
    """Generates position IDs accounting for guideline and step structure."""

    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

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
                    # "num_plans": 0,
                    "num_steps": 0,
                    "is_in_guideline": False,
                    # "is_in_plan": False,
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
            position_ids[i:] -= position_ids[i] - (current_block_state["guideline_end_pos_id"] + 1)  # make step overlap \n

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
            )  # make takeaway overlap \n
            block_stack.pop()

        i += 1

    # Final check for unclosed blocks
    if block_stack or len(position_ids) != len(input_ids):
        return None, True

    return position_ids, False
