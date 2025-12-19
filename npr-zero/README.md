# VeRL: Volcano Engine Reinforcement Learning for LLM (NPR-Zero Fork)

This directory (`npr-zero`) houses the Reinforcement Learning framework for the **Native Parallel Reasoner (NPR)** project, derived from [VeRL](https://github.com/volcengine/verl).

It has been customized to support **Glyph-based Parallel Reasoning**, enabling the training of models that utilize structured atomic glyphs (`🜞`, `🜆`, `🜂`, `🜃`, `🝞`) for multi-path reasoning.

## Key Adaptations for Glyph-NPR

*   **Reward Managers**: The `NaiveRewardManager` has been updated to integrate `glyph_format_reward`, ensuring that models are incentivized to follow the strict Glyph structure.
*   **Format Validation**: Utilities in `verl/utils/reward_score/` include logic to valid Glyph sequences and parse parallel reasoning paths.
*   **Tokenization**: Compatible with tokenizers that have been augmented with Glyph special tokens.

## Installation

To install this package in editable mode:

```bash
pip install -e .
```

## Structure

*   `verl/trainer`: Core PPO/GRPO training loops.
*   `verl/workers`: Actor, Critic, and Reward Manager implementations.
*   `examples/`: Scripts for running RL experiments (e.g., `ppo_trainer`).

