import torch

from typing import List


TAG_TOKEN_IDS = {
    "guideline_start": "🜞",
    # "guideline_end": "</guideline>", # Implicit
    "plan_marker": "🜆", # 🜆 Plan content ...
    # "plan_end": "</plan>",
    "step_marker": "🜂", # 🜂 Step content ...
    # "step_end": "</step>",
    "takeaway_marker": "🜃", # 🜃 Takeaway content ...
    # "takeaway_end": "</takeaway>",
    "final_marker": "🝞"
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

    guideline_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["guideline_start"])
    step_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["step_marker"])
    takeaway_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["takeaway_marker"])

    # locate the first "<|im_end|>" token if possible, or start from 0
    try:
        start_idx = input_ids.index(tokenizer.convert_tokens_to_ids("<|im_end|>")) + 1
    except ValueError:
        start_idx = 0
        
    i = start_idx
    structure_stack = [] # Stores active blocks (parallel/guideline)
    
    # Helper to close current block (step)
    # But here we need to construct step_spans for the parallel block.
    # structure_stack elements: {"type": "parallel", "step_spans": [(start, end), ...], "current_step_start": int or None}
    
    while i < seq_len:
        current_token_id = input_ids[i]

        if current_token_id == guideline_start_id:
            # Start parallel block
            # If we were in a step (implicit?), close it?
            # Nested guidelines not standard but let's assume one level for now or strict nesting.
            # If we see 🜞, we start a new context.
            structure_stack.append({"type": "parallel", "step_spans": [], "current_step_start": None, "guideline_start": i})
            i += 1
            continue
            
        elif current_token_id == step_start_id:
             if not structure_stack or structure_stack[-1]["type"] != "parallel":
                 # Step outside guideline? Ignore or error.
                 pass
             else:
                 # Closing previous step if exists
                 block = structure_stack[-1]
                 if block["current_step_start"] is not None:
                     block["step_spans"].append((block["current_step_start"], i))
                 
                 # Start new step
                 block["current_step_start"] = i
        
             i += 1
             continue
             
        elif current_token_id == takeaway_start_id:
             if not structure_stack or structure_stack[-1]["type"] != "parallel":
                 pass
             else:
                 block = structure_stack[-1]
                 # Close last step
                 if block["current_step_start"] is not None:
                     block["step_spans"].append((block["current_step_start"], i))
                     block["current_step_start"] = None
                 
                 # Now process the block
                 step_spans = block["step_spans"]
                 num_steps = len(step_spans)
                 
                 if num_steps > 1:
                     all_i_indices_to_mask = []
                     all_j_indices_to_mask = []

                     for step_idx_a in range(num_steps):
                         start_a, end_a = step_spans[step_idx_a]
                         if start_a >= end_a: continue
                         indices_a = torch.arange(start_a, end_a, device=device)

                         for step_idx_b in range(step_idx_a + 1, num_steps):
                             start_b, end_b = step_spans[step_idx_b]
                             if start_b >= end_b: continue
                             indices_b = torch.arange(start_b, end_b, device=device)

                             grid_i, grid_j = torch.meshgrid(indices_a, indices_b, indexing="ij")
                             all_i_indices_to_mask.append(grid_i.flatten())
                             all_j_indices_to_mask.append(grid_j.flatten())

                     if all_i_indices_to_mask:
                         final_i = torch.cat(all_i_indices_to_mask)
                         final_j = torch.cat(all_j_indices_to_mask)
                         bool_attention_mask[final_i, final_j] = False
                         bool_attention_mask[final_j, final_i] = False
                 
                 # Close parallel block? 
                 # Takeaway is part of the flow after steps.
                 # Usually matches with Guideline.
                 structure_stack.pop()
        
             i += 1
             continue
        
        else:
             i += 1

    # End loop
    
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
    # Using explicit names for clarity
    guideline_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["guideline_start"])
    step_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["step_marker"])
    takeaway_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["takeaway_marker"])

    position_ids = torch.arange(len(input_ids), device="cpu", dtype=torch.long)

    # State
    # We need to track:
    # - current_parallel_block: {guideline_pos_id: int, steps: [{start_idx: int, end_idx: int}], current_step_idx: int}
    
    block_stack = []

    try:
        start_idx = input_ids.index(tokenizer.convert_tokens_to_ids("<|im_end|>")) + 1
    except ValueError:
        start_idx = 0
        
    i = start_idx
    
    while i < len(input_ids):
        token_id = input_ids[i]
        
        if token_id == guideline_id:
            block_stack.append({
                "type": "parallel",
                "guideline_start_pos": i, # Raw index
                "step_start_pos": None, # Normalized pos ID for start of steps
                "current_step_start_idx": None,
                "step_lengths": [],
                "base_pos": position_ids[i].item() # The pos ID at the start of guideline
            })
            
        elif token_id == step_id:
            if block_stack:
                block = block_stack[-1]
                
                # If we were in a step, close it
                if block["current_step_start_idx"] is not None:
                     # Calculate length of previous step
                     # It ended at i
                     prev_len = position_ids[i].item() - position_ids[block["current_step_start_idx"]].item()
                     block["step_lengths"].append(prev_len)
                
                # Start new step
                # The position of this step should reset to match others.
                # Where do steps start?
                # Usually after the guideline content (plans).
                # But guideline content length varies.
                # Wait, parallel position IDs mean steps share the same position range.
                # So Step 2 starts at same pos ID as Step 1.
                
                if block["step_start_pos"] is None:
                    # First step. Define the anchor.
                    # This step starts at current i.
                    block["step_start_pos"] = position_ids[i].item()
                    block["current_step_start_idx"] = i
                else:
                    # Subsequent step. Reset position.
                    # Current pos ID at i should become step_start_pos
                    # So pivot = position_ids[i] - step_start_pos
                    # position_ids[i:] -= pivot
                    
                    pivot = position_ids[i].item() - block["step_start_pos"]
                    position_ids[i:] -= pivot
                    block["current_step_start_idx"] = i

        elif token_id == takeaway_id:
            if block_stack:
                block = block_stack[-1]
                # Close last step
                if block["current_step_start_idx"] is not None:
                     prev_len = position_ids[i].item() - position_ids[block["current_step_start_idx"]].item()
                     block["step_lengths"].append(prev_len)
                
                # Takeaway should start after the longest step.
                # Max step end pos = step_start_pos + max(step_lengths)
                # Current pos id at i (takeaway start) is effectively step_start_pos (since we reset or continued).
                # Wait, if we just finished a step, position_ids[i] is step_start + length_of_last_step.
                # But we want it to be step_start + max(all_step_lengths).
                
                current_pos_at_takeaway = position_ids[i].item()
                if block["step_lengths"]:
                    max_len = max(block["step_lengths"])
                    target_pos = block["step_start_pos"] + max_len
                    
                    adjustment = target_pos - current_pos_at_takeaway
                    position_ids[i:] += adjustment
                
                block_stack.pop()
        
        i += 1

    return position_ids, False
