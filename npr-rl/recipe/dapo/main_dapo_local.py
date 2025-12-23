
import os
import socket
import hydra
from omegaconf import OmegaConf
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.fs import copy_to_local
from verl.utils import hf_processor, hf_tokenizer

# Local imports
from verl.single_controller.local.worker_group import LocalWorkerGroup, ClassWithInitArgs
from .dapo_local_trainer import LocalDAPOTrainer, Role
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker, RewardModelWorker

@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    run_ppo_local(config)

def run_ppo_local(config):
    print(f"Running Local DAPO PPO on {socket.gethostname()}")
    
    # Resolve config
    OmegaConf.resolve(config)
    config.actor_rollout_ref.rollout.exp_name = config.trainer.experiment_name

    # Download/Copy model
    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)

    # Worker definitions
    # Assuming FSDP/FSDP2 strategy
    
    # We must access the original classes, not Ray wrappers if they are wrapped.
    # In npr-rl, they might be just classes.
    
    role_worker_mapping = {}
    
    # ActorRollout
    role_worker_mapping[Role.ActorRollout] = ClassWithInitArgs(
        ActorRolloutRefWorker,
        config=config,
        role="actor_rollout"
    )
    
    # Critic
    if config.critic.strategy != "none":
        role_worker_mapping[Role.Critic] = ClassWithInitArgs(
            CriticWorker,
            config=config,
            role="critic"
        )
        
    # Reward Model
    if config.reward_model.enable:
        role_worker_mapping[Role.RewardModel] = ClassWithInitArgs(
            RewardModelWorker,
            config=config,
            role="reward_model" # Check verify role name
        )
        
    # Ref Policy
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ClassWithInitArgs(
            ActorRolloutRefWorker,
            config=config,
            role="ref"
        )
        
    # Reward Managers
    reward_fn = load_reward_manager(
        config,
        tokenizer,
        0,
        max_resp_len=config.data.max_response_length,
        overlong_buffer_cfg=config.reward_model.overlong_buffer,
    )
    
    val_reward_fn = load_reward_manager(
        config,
        tokenizer,
        1,
        max_resp_len=config.data.max_response_length,
        overlong_buffer_cfg=config.reward_model.overlong_buffer,
    )

    # Dataset
    # Assuming the Trainer creates it or we need to pass it.
    # RayDAPOTrainer inherits RayPPOTrainer which creates dataset inside __init__ if not passed?
    # No, usually passed or created.
    # In main_dapo.py, it doesn't create datasets explicitly. 
    # Let's check RayPPOTrainer.__init__. 
    # It calls create_rl_dataset if not passed.
    
    # LocalPPOTrainer skeleton I wrote expects passed datasets or creates them?
    # I wrote: self._create_dataloader(train_dataset, ...)
    # I need to implement data loading here or in Trainer.
    
    from verl.trainer.main_ppo import create_rl_dataset
    # Note: create_rl_dataset might need adaptation if it expects specific args
    
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)

    trainer = LocalDAPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    trainer.init_workers()
    trainer.fit()

if __name__ == "__main__":
    main()
