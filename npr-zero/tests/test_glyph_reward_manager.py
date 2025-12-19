
import unittest
from unittest.mock import MagicMock
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

# Mock ray before importing verl
sys.modules["ray"] = MagicMock()

from verl.workers.reward_manager.naive import NaiveRewardManager

class MockDataProto:
    def __init__(self, batch, non_tensor_batch):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
    
    def __len__(self):
        return len(self.batch["responses"])
    
    def __getitem__(self, idx):
        item_batch = {k: v[idx] if isinstance(v, torch.Tensor) else v[idx] for k, v in self.batch.items()}
        
        item_non_tensor = {}
        for k, v in self.non_tensor_batch.items():
            if isinstance(v, list):
                item_non_tensor[k] = v[idx]
            elif isinstance(v, dict):
                item_non_tensor[k] = {vk: vv[idx] if isinstance(vv, list) else vv for vk, vv in v.items()}
            else:
                item_non_tensor[k] = v
                
        class MockItem:
            def __init__(self, b, nb):
                self.batch = b
                self.non_tensor_batch = nb
                
        return MockItem(item_batch, item_non_tensor)

class TestGlyphRewardManager(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MagicMock()
        # Mock encoding behavior for Glyphs
        self.glyph_map = {
            "🜞": [1001],
            "🜆": [1002],
            "🜂": [1003],
            "🜃": [1004],
            "🝞": [1005]
        }
        
        def encode_side_effect(text, add_special_tokens=False):
            return self.glyph_map.get(text, [])

        self.tokenizer.encode.side_effect = encode_side_effect
        self.tokenizer.decode.return_value = "Mock Response"

    def test_glyph_reward_valid(self):
        # Valid Sequence IDs: 🜞 🜆 🜂 🜃 🝞
        valid_ids = torch.tensor([[1001, 1002, 1003, 1004, 1005]])
        
        rm = NaiveRewardManager(tokenizer=self.tokenizer, num_examine=1, compute_score=lambda **kwargs: 1.0)
        
        batch = {
            "prompts": torch.zeros((1, 5), dtype=torch.long),
            "responses": valid_ids,
            "attention_mask": torch.ones((1, 10), dtype=torch.long)
        }
        
        non_tensor = {
            "data_source": ["test_source"],
            "reward_model": {"ground_truth": ["test_gt"]},
            "extra_info": {}
        }
        
        data = MockDataProto(batch=batch, non_tensor_batch=non_tensor)
        
        # Run
        reward_tensor = rm(data)
        
        last_token_idx = 4
        self.assertEqual(reward_tensor[0, last_token_idx].item(), 2.0)

    def test_glyph_reward_invalid(self):
        # Invalid Sequence IDs: 1001, 1005 (Skip steps/plans) -> Should fail logic
        invalid_ids = torch.tensor([[1001, 1005, 0, 0, 0]])
        
        rm = NaiveRewardManager(tokenizer=self.tokenizer, num_examine=1, compute_score=lambda **kwargs: 1.0)
        
        batch = {
            "prompts": torch.zeros((1, 5), dtype=torch.long),
            "responses": invalid_ids,
            "attention_mask": torch.cat([torch.ones((1, 5)), torch.tensor([[1, 1, 0, 0, 0]])], dim=1).long()
        }
        
        non_tensor = {
            "data_source": ["test_source"],
            "reward_model": {"ground_truth": ["test_gt"]},
            "extra_info": {}
        }
        
        data = MockDataProto(batch=batch, non_tensor_batch=non_tensor)
        
        reward_tensor = rm(data)
        
        # Base 1.0 + Format 0.0 = 1.0
        self.assertEqual(reward_tensor[0, 1].item(), 1.0)

if __name__ == '__main__':
    unittest.main()
