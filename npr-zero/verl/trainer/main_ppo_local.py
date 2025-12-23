
import os
import hydra
import torch
from omegaconf import OmegaConf

from verl.utils import hf_tokenizer, hf_processor
from verl.single_controller.base.worker_group import ClassWithInitArgs
from verl.single_controller.local.trainer import LocalPPOTrainer, Role
# Import your worker classes here. You might need to change imports if they heavily depend on Ray
# For now, let's assume we can use the FSDP workers or similar, or we need to strip Ray decorators?
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker, RewardModelWorker
# Warning: These existing workers might decorate classes with @ray.remote.
# If they do, we need to access the underlying class. 
# Ray decorators usually put the original class in .__ray_actor_class__ or similar if it's a remote function,
# but for classes it returns a ActorClass.
# We might need to unwrap them or import the raw class if it's available.

@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo_local(config)

def run_ppo_local(config):
    print("Running PPO Local...")
    
    # 1. Prepare Tokenizer
    local_path = config.actor_rollout_ref.model.path # Assuming local path or HF path
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path)
    
    # 2. Define Workers (Unwrap Ray if needed)
    # The existing worker files imports Ray. We must ensure that doesn't break things if Ray is not installed/init.
    # Ideally, we should have a 'LocalWorker' class if the existing ones are too tied to Ray.
    # But let's try to reuse `ActorRolloutRefWorker`.
    
    # Check if ActorRolloutRefWorker is a Ray ActorClass
    if hasattr(ActorRolloutRefWorker, '__ray_actor_class__'):
        ActorRolloutRefWorkerCls = ActorRolloutRefWorker.__ray_actor_class__
    else:
        ActorRolloutRefWorkerCls = ActorRolloutRefWorker

    if hasattr(CriticWorker, '__ray_actor_class__'):
        CriticWorkerCls = CriticWorker.__ray_actor_class__
    else:
        CriticWorkerCls = CriticWorker
        
    # 3. Setup Role Mapping
    role_worker_mapping = {}
    
    # Initialize ActorRollout
    # We pass ClassWithInitArgs so the LocalPPOTrainer can instantiate it
    role_worker_mapping[Role.ActorRollout] = ClassWithInitArgs(
        ActorRolloutRefWorkerCls,
        config=config,
        role="actor_rollout"
    )
    
    # Initialize Critic
    role_worker_mapping[Role.Critic] = ClassWithInitArgs(
        CriticWorkerCls,
        config=config,
        role="critic"
    )

    # 4. Data
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
    
    # 5. Trainer
    trainer = LocalPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        processor=processor
    )
    
    trainer.init_workers()
    trainer.fit()

if __name__ == "__main__":
    main()
