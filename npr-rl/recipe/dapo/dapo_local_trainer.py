# Local DAPO Trainer
# Based on RayDAPOTrainer but for single-node local execution

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
# Note: LocalPPOTrainer is our local base
from verl.single_controller.local.trainer import LocalPPOTrainer
from verl.utils.profiler import marked_timer
from verl.utils.tracking import Tracking


class LocalDAPOTrainer(LocalPPOTrainer):
    """
    Local version of RayDAPOTrainer.
    """

    def fit(self):
        """
        The training loop of PPO.
        """
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            run_id=self.config.trainer.run_id,
        )

        if not os.path.exists(f"experiments/record/{self.config.trainer.experiment_name}"):
            os.makedirs(f"experiments/record/{self.config.trainer.experiment_name}", exist_ok=True)

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint (if implemented in LocalPPOTrainer, otherwise skip or implement)
        # self._load_checkpoint() 

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate() # Assumes _validate is implemented in LocalPPOTrainer or here
            # assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            if val_metrics:
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        # We don't have self.total_training_steps in LocalPPOTrainer yet, need to compute it
        self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        bad_case_num = 0
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                do_profile = False # Simplify profiling for local

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                
                # pop keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)

                    # REMAX logic omitted for brevity unless strict requirement. 
                    # Assuming standard PPO for now or copying if needed.
                    
                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    
                    # Filtering
                    # Assuming always filter or based on config
                    mask = gen_batch_output.batch["keep_indices"].bool()
                    gen_batch_output = gen_batch_output[mask]
                    new_batch = new_batch[mask]
                    bad_case_num = (~mask).sum().item()
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        reward_extra_infos_dict = {}
                        try:
                            # Local reward_fn call
                            if self.reward_fn:
                                reward_result = self.reward_fn(new_batch, return_dict=True)
                                reward_tensor = reward_result["reward_tensor"]
                                reward_result["reward_extra_info"]["bad_case_num"] = np.array([bad_case_num] * new_batch.batch.batch_size[0])
                                reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                                new_batch.batch["token_level_scores"] = reward_tensor
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            # Fallback?
                        
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        if self.config.algorithm.use_kl_in_reward:
                             new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                             metrics.update(kl_metrics)
                        else:
                             new_batch.batch["token_level_rewards"] = new_batch.batch.get("token_level_scores", torch.zeros_like(new_batch.batch["input_ids"], dtype=torch.float32))

                    # Logic for batch accumulation (simplified)
                    batch = new_batch
                    true_traj_bsz = batch.batch.batch_size[0] // 8 * 8 
                    batch = batch[:true_traj_bsz]

                    # Updating
                    if batch.batch.batch_size[0] > 0:
                        batch.batch["response_mask"] = compute_response_mask(batch)
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        # Recompute old log probs
                        # (omitted generic optimization logic for brevity, assuming minimal working path)
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                        # Values
                        if self.use_critic:
                             values = self.critic_wg.compute_values(batch)
                             batch = batch.union(values)

                        # Advantages
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                             norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        )

                        # Update Critic
                        if self.use_critic:
                             critic_output = self.critic_wg.update_critic(batch)
                             metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                        # Update Actor
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                # Metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics["train/num_gen_batches"] = num_gen_batches
                
                logger.log(data=metrics, step=self.global_steps)
                
                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
                
                # Checkpointing
                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                     # self._save_checkpoint()
                     pass

        progress_bar.close()
