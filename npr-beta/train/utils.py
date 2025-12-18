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
            # Add Glyph tokens
            new_special_tokens += ["🜞", "🜆", "🜂", "🜃", "🝞"]

        logger.info(f"Using tokenzier of {model.config.model_type}")

    tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
    model.resize_token_embeddings(new_num_tokens=len(tokenizer), pad_to_multiple_of=64)

    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    tied = embed.weight.data_ptr() == lm_head.weight.data_ptr()

    for tok in new_special_tokens:
        base_word = tok.strip("<>").strip() 
        if not base_word: # Handle case where strip might leave nothing or just whitespace
             base_word = tok

        # Fallback for glyphs if they are not in base vocab (likely)
        # We initialize them from 'step', 'plan', etc. or just random/mean.
        if tok in ["🜞", "🜆", "🜂", "🜃", "🝞"]:
             # Map glyphs to semantic equivalents for initialization
             semantic_map = {
                 "🜞": "guideline",
                 "🜆": "plan",
                 "🜂": "step",
                 "🜃": "takeaway",
                 "🝞": "final"
             }
             base_word = semantic_map.get(tok, "step")
        
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
            logger.warning(
                f"Failed to initialize special token '{tok}': some base tokens are unknown. Using random init."
            )


TAG_TOKEN_IDS = {
    # Guideline definition phase - sets the high-level reasoning objective
    "guideline_start": "🜞",
    # Plan planning phase - breaks down reasoning strategy
    "plan_start": "🜆",
    # Step execution phase - parallel reasoning steps
    "step_start": "🜂",
    # Takeaway phase - synthesis of parallel steps
    "takeaway_start": "🜃",
    "final_start": "🝞"
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
    guideline_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["guideline_start"])
    plan_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["plan_start"])
    step_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["step_start"])
    takeaway_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["takeaway_start"])
    final_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS["final_start"])

    # locate the first <|im_start|>user and skip to <|im_start|>assistant or just find start
    # Simplified: finding start of generation area (usually after <|im_end|> of user prompt)
    try:
        start_index = input_ids.index(tokenizer.convert_tokens_to_ids("<|im_end|>")) + 1
    except ValueError:
        start_index = 0

    i = start_index
    structure_stack = []  # Stack for tracking nested guideline/plan/step structures
    
    # Helper to close current block
    def close_current_block(end_index, expected_type=None):
        if not structure_stack:
            return None
        
        current_block = structure_stack[-1]
        
        if expected_type and current_block["type"] != expected_type:
             # If we expect to close a step but we are in a plan, that's an issue with implicit logic or input
             # For robustness, we might want to pop until we find it, or error.
             # Strict: Error.
             # raise ValueError(f"Expected to close {expected_type}, but found {current_block['type']}")
             pass # Let the caller handle strictness

        structure_stack.pop()
        
        # logic for recording spans
        start_marker_index = current_block["start_marker_index"]
        
        # If closing a step or plan, record it in the parent parallel block
        if current_block["type"] in ["step", "plan"]:
             # Find enclosing parallel block
             enclosing_parallel_block = None
             for block in reversed(structure_stack):
                  if block["type"] == "parallel":
                       enclosing_parallel_block = block
                       break
             
             if enclosing_parallel_block:
                  if current_block["type"] == "step":
                       enclosing_parallel_block["step_spans"].append((start_marker_index, end_index))
                  elif current_block["type"] == "plan":
                       enclosing_parallel_block["plans"].append((start_marker_index, end_index))
        
        return current_block

    while i < seq_len:
        current_token_id = input_ids[i]

        # 1. Guideline Start (🜞)
        if current_token_id == guideline_start_id:
            # If we are already in a guideline (parallel block), this is nested or error. 
            # NPR usually implies one main block or sequential blocks.
            # If we strictly follow NPR, we shouldn't nest 🜞 inside 🜞.
            # If we see a new 🜞, we assume the previous one ended (implicit close).
            if structure_stack and structure_stack[-1]["type"] == "parallel":
                 close_current_block(i, "parallel")

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

        # 2. Plan Start (🜆) or Step Start (🜂)
        elif current_token_id in [plan_start_id, step_start_id]:
            # Must be inside a parallel block (guideline)
            if not structure_stack or structure_stack[-1]["type"] != "parallel":
                 # If we see a step/plan outside strictly, we could implicitly start a parallel block or error.
                 # Updated NPR logic often is loose. Let's assume strict for now based on previous code.
                 # actually, if we just saw <guideline>.... <step>, then stack[-1] is parallel.
                 # BUT if we saw <step>... <step>, stack[-1] is "step". IMPORTANT.
                 
                 # Logic: If current top is "step" or "plan", CLOSE IT first.
                 if structure_stack and structure_stack[-1]["type"] in ["step", "plan"]:
                      close_current_block(i)
                 
                 # Now check again
                 if not structure_stack or structure_stack[-1]["type"] != "parallel":
                       # Error or implicit parallel? Original code raised ValueError.
                       # raise ValueError(f"Found 🜆/🜂 outside 🜞 at index {i}")
                       pass 

            if current_token_id == plan_start_id:
                structure_stack.append({"type": "plan", "start_marker_index": i})

            if current_token_id == step_start_id:
                structure_stack.append({"type": "step", "start_marker_index": i})

            i += 1
            continue

        # 3. Takeaway Start (🜃)
        elif current_token_id == takeaway_start_id:
            # This marks the end of parallel execution.
            # Close any open step/plan
            if structure_stack and structure_stack[-1]["type"] in ["step", "plan"]:
                 close_current_block(i)
            
            # Now we should be at "parallel" level
            if not structure_stack or structure_stack[-1]["type"] != "parallel":
                 raise ValueError("Found 🜃 (Takeaway) without matching 🜞 (Guideline)")
            
            closed_parallel_block = structure_stack.pop() # Close the parallel block itself?? 
            # Wait, 🜃 is *part* of the flow. Usually it comes *after* steps.
            # The original code closed the parallel block at takeaway_start? 
            # Looking at original: `elif current_token_id == takeaway_start_id: closed_parallel_block = structure_stack.pop()`
            # Yes, it seems 🜃 closes the "parallel" region where masking applies.
            
            step_spans_in_this_block = closed_parallel_block["step_spans"]
            num_steps = len(step_spans_in_this_block)

            # APPLY MASKING LOGIC (Identical to original)
            if num_steps > 1:
                all_i_indices_to_mask = []
                all_j_indices_to_mask = []

                for step_idx_a in range(num_steps):
                    start_a, end_a = step_spans_in_this_block[step_idx_a]
                    if start_a >= end_a: continue
                    indices_a = torch.arange(start_a, end_a, device=device)

                    for step_idx_b in range(step_idx_a + 1, num_steps):
                        start_b, end_b = step_spans_in_this_block[step_idx_b]
                        if start_b >= end_b: continue
                        indices_b = torch.arange(start_b, end_b, device=device)

                        grid_i, grid_j = torch.meshgrid(indices_a, indices_b, indexing="ij")
                        all_i_indices_to_mask.append(grid_i.flatten())
                        all_j_indices_to_mask.append(grid_j.flatten())

                if all_i_indices_to_mask:
                    final_i = torch.cat(all_i_indices_to_mask)
                    final_j = torch.cat(all_j_indices_to_mask)
                    # Block cross-attention
                    bool_attention_mask[final_i, final_j] = False
                    bool_attention_mask[final_j, final_i] = False
            
            i += 1
            continue
        
        elif current_token_id == final_start_id:
             # Just ensures everything is closed
             while structure_stack:
                  close_current_block(i)
             i += 1
             continue

        i += 1

    # Close any remaining blocks at the end of sequence (Implicit closing at EOS)
    while structure_stack:
         close_current_block(seq_len)

    # FINAL CONVERSION
    float_attention_mask = torch.full_like(
        bool_attention_mask, -torch.inf, dtype=torch.float
    )
    float_attention_mask = float_attention_mask.masked_fill(bool_attention_mask, 0.0)

    return float_attention_mask


def construct_parallel_position_ids(input_ids: List[int], tokenizer) -> List[int]:
    """Generates position IDs accounting for Plan and Step structure."""

    # Get special token IDs
    # Get special token IDs
    tag_ids = {
        tag: tokenizer.convert_tokens_to_ids(token)
        for tag, token in TAG_TOKEN_IDS.items()
    }
    
    # Try finding im_end for start, else 0
    try:
         start_idx = input_ids.index(tokenizer.convert_tokens_to_ids("<|im_end|>")) + 1
    except ValueError:
         start_idx = 0

    position_ids = torch.arange(len(input_ids), device="cpu", dtype=torch.long)

    block_stack = []
    i = start_idx
    
    # Implicit closing state
    # We need to track implicit "is_in_step" based on the stack
    
    while i < len(input_ids):
        token_id = input_ids[i]
        
        # Check current context
        current_block_state = block_stack[-1] if block_stack else None
        
        # 1. Guideline Start (🜞)
        if token_id == tag_ids["guideline_start"]:
            # If previous guideline open, it implicitly closed? Position IDs reset?
            # Usually strict NPR resets pos IDs relative to guideline start or previous.
            # Here we just start a new block.
            
            # Note: Explicitly handling implicit close in pos IDs is harder without a full parse.
            # But we can assume clean structure or just push new state.
            
            block_stack.append(
                {
                    "guideline_end_pos_id": -1, # Set when we exit guideline (takeaway start)
                    "max_step_len": 0,
                    "num_steps": 0,
                    "is_in_step": False,
                    "step_start_pos_id": -1
                }
            )
        
        # 2. Step Start (🜂)
        elif token_id == tag_ids["step_start"] and current_block_state:
             # Implicitly close previous step if open
             if current_block_state["is_in_step"]:
                  # logic for closing step
                  step_len = position_ids[i] - current_block_state["step_start_pos_id"]
                  current_block_state["max_step_len"] = max(current_block_state["max_step_len"], step_len)
                  current_block_state["num_steps"] += 1
                  current_block_state["is_in_step"] = False
             
             # Start new step
             current_block_state["is_in_step"] = True
             current_block_state["step_start_pos_id"] = position_ids[i]
             
             # Adjust position IDs to be relative to the END of guideline init 
             # (Wait, original code used guideline_end_pos_id, but that was optional closing tag `</guideline>`)
             # In Glyph, 🜞 starts it. The "Guideline" text follows. Then 🜂 starts.
             # We want all steps to start from the same position relative to Guideline text??
             # Or relative to where they branched? 
             # Original: `position_ids[i:] -= position_ids[i] - (guideline_end_pos_id + 1)`
             # If we don't have explicit </guideline>, where does it end?
             # It effectively ends at the first 🜂.
             
             if current_block_state["guideline_end_pos_id"] == -1:
                  # First step defines the end of guideline text
                  current_block_state["guideline_end_pos_id"] = position_ids[i] - 1
             
             # Verify logic:
             # If step 1 starts at 20. guideline_end_pos_id = 19.
             # shift = 20 - (19 + 1) = 0. No shift for first step? Correct.
             # step 1 ends at 30.
             # step 2 starts at 31.
             # shift = 31 - (19 + 1) = 11.
             # pos ids at 31 become 31 - 11 = 20. 
             # So step 2 also starts at 20. Parallel! Correct.
             
             position_ids[i:] -= position_ids[i] - (current_block_state["guideline_end_pos_id"] + 1)

        # 3. Takeaway Start (🜃)
        elif token_id == tag_ids["takeaway_start"] and current_block_state:
             # Close last step
             if current_block_state["is_in_step"]:
                  step_len = position_ids[i] - current_block_state["step_start_pos_id"]
                  current_block_state["max_step_len"] = max(current_block_state["max_step_len"], step_len)
                  current_block_state["num_steps"] += 1
                  current_block_state["is_in_step"] = False

             # Shift takeaway to start AFTER the longest step
             # current pos is i.
             # guideline end is G.
             # max step len is L.
             # we want takeaway to start at G + L + 1.
             # shift = i - (G + L + 1)
             
             shift = position_ids[i] - (
                current_block_state["guideline_end_pos_id"]
                + current_block_state["max_step_len"]
                + 1
            )
             position_ids[i:] -= shift
             
             # Pop block, as we are leaving parallel region
             block_stack.pop()

        i += 1

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
