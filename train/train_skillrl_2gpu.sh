#!/usr/bin/env bash
# SkillRL ALFWorld GRPO on 2 GPUs with a FIXED skill bank (dynamic o3 skill updates off).
# Based on SkillRL examples/grpo_trainer/run_alfworld_skills.sh; changed lines are marked [2gpu]/[fixed]/[ckpt].
#
# Usage (inside the SkillRL conda env, from anywhere):
#   export SKILLRL_ROOT=~/SkillRL
#   export CUDA_VISIBLE_DEVICES=0,1                 # the two GPUs you reserved
#   MODEL_PATH=~/ckpt/rl_merged EXP=cont_from_rl80 bash train_skillrl_2gpu.sh      # continue from the public RL ckpt
#   MODEL_PATH=~/ckpt/sft/checkpoint-140 EXP=from_sft bash train_skillrl_2gpu.sh   # or start from SFT
#
# Knobs: EPOCHS (default 150), TEST_FREQ (10), SAVE_FREQ (10), GPU_MEM (0.4), FREE_CACHE (False; set True on OOM),
#        CKPT_DIR (default ~/ckpt/train/$EXP)
set -x
: "${SKILLRL_ROOT:?set SKILLRL_ROOT}"; : "${MODEL_PATH:?set MODEL_PATH}"
cd "$SKILLRL_ROOT"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
EXP=${EXP:-grpo_7b_skills_fixed_2gpu}
CKPT_DIR=${CKPT_DIR:-$HOME/ckpt/train/$EXP}
train_data_size=16
val_data_size=64
group_size=8
num_cpus_per_env_worker=0.1

python3 -m examples.data_preprocess.prepare --mode 'text' --train_data_size $train_data_size --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 `# [2gpu] was 4` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM:-0.4} `# [2gpu] was 0.5` \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE:-False} \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path=memory_data/alfworld/claude_style_skills.json \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.enable_dynamic_update=False `# [fixed] was True (needs Azure o3)` \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] `# was console+wandb` \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=$EXP \
    trainer.default_local_dir=$CKPT_DIR `# [ckpt]` \
    trainer.n_gpus_per_node=2 `# [2gpu] was 4` \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-10} \
    trainer.test_freq=${TEST_FREQ:-10} `# was 5` \
    trainer.total_epochs=${EPOCHS:-150} \
    trainer.val_before_train=True `# [ckpt] baseline at step 0` "$@"
