# Local PPO Trainer for npr-rl
# Based on RayPPOTrainer but for single-node local execution

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.local import LocalWorkerGroup
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = type[Worker]

class Role(Enum):
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6

class LocalPPOTrainer:
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.role_worker_mapping = role_worker_mapping
        
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.device_name = device_name if device_name else self.config.trainer.device
        
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GPG,
        ]:
            self.use_critic = False
        else:
             self.use_critic = True

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def init_workers(self):
        """Initialize workers using LocalWorkerGroup directly."""
        
        # 1. ActorRollout Group
        self.actor_rollout_wg = LocalWorkerGroup(
            cls_with_init=self.role_worker_mapping[Role.ActorRollout],
            name_prefix='ActorRollout'
        )
        self.actor_rollout_wg.init_model()

        # 2. Critic Group
        if self.use_critic:
            self.critic_wg = LocalWorkerGroup(
                cls_with_init=self.role_worker_mapping[Role.Critic],
                name_prefix='Critic'
            )
            self.critic_wg.init_model()

        # 3. Ref Policy
        if self.use_reference_policy:
            self.ref_policy_wg = LocalWorkerGroup(
                cls_with_init=self.role_worker_mapping[Role.RefPolicy],
                name_prefix='RefPolicy'
            )
            self.ref_policy_wg.init_model()

        # 4. Reward Model
        if self.use_rm:
            self.rm_wg = LocalWorkerGroup(
                cls_with_init=self.role_worker_mapping[Role.RewardModel],
                name_prefix='RewardModel'
            )
            self.rm_wg.init_model()
            
    def fit(self):
        """
        Main training loop.
        """
        self.global_steps = 0
        
        # Simplified Training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_stats = {}
                
                # ... Populate with actual training logic
                # For brevity, placeholders.
                
                self.global_steps += 1
                
                # Validation
                if self.global_steps % self.config.trainer.val_check_interval == 0:
                     self._validate()

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            collate_fn=collate_fn,
            sampler=train_sampler if train_sampler else None
        )
        
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size,
            collate_fn=collate_fn
        )

    def _validate(self):
         pass
