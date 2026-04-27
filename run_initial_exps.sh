#!/bin/bash

# run conversion
# conda deactivate
# conda deactivate 

# export HF_TOKEN=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 

export $HF_LEROBOT_HOME="/store/real/maxjdu/repos/LiberoContinualLearning/datasets/libero_lerobot" # used to search for the lerobot 


# exit 

# this script trains the ER algorithm 
# CUDA_VISIBLE_DEVICES=6 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run scripts/train.py pi0_libero_low_mem_finetune-libero_goal_sequential \
#   --exp-name=my_libero_experiment  \
#   --overwrite \
#   --fsdp-devices=1



# this needs to be run on the outer directory.
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python examples/libero/eval_continual.py \
  /store/real/maxjdu/repos/continual-openpi/runs/pi0_libero_low_mem_finetune-libero_goal_sequential/my_libero_experiment \
  libero_goal 

# python -c "import sys; print(f'Python version: {sys.version}'); print(f'Executable: {sys.executable}')"

# exit 

# CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run examples/libero/eval_continual.py \
#   /store/real/maxjdu/repos/continual-openpi/runs/pi0_libero_low_mem_finetune-libero_goal_sequential/my_libero_experiment/ \
#   libero_goal 

# uv run scripts/serve_policy.py --port 8000 --task_idx 0 policy:checkpoint --policy.config pi0_libero_low_mem_finetune-libero_goal_sequential --policy.dir /store/real/maxjdu/repos/continual-openpi/runs/pi0_libero_low_mem_finetune-libero_goal_sequential/my_libero_experiment/9999

# CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 python examples/libero/eval_continual.py \
#   libero_goal 







# deactivate 

# remaining questions:
# - how the ER buffer is implemented (and how I can modify it) 




####### SERVING THE POLICY ####

# default policy 
# CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 uv run scripts/serve_policy.py --port 8000 --env LIBERO policy:default 

# uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero 

        # "python", "examples/libero/main_parallel.py",
        # "--args.task_suite_name", TASK_SUITE,
        # "--args.eval_upto_task", str(task_idx),
        # "--args.num_trials_per_task", str(TRIALS_PER_TASK),
        # "--args.base_dir", ckpt_dir,
# running 


#############

##### CONVERSION OF DATA #### 

# uv run examples/libero/convert_libero_data_to_lerobot.py \
#     --data_dir="/store/real/maxjdu/repos/LiberoContinualLearning/datasets/libero_noops" \
#     --output_dir="/store/real/maxjdu/repos/LiberoContinualLearning/datasets/libero_lerobot" \
#     --idx=0

# uv run examples/libero/convert_libero_data_to_lerobot.py \
#     --data_dir="/store/real/maxjdu/repos/LiberoContinualLearning/datasets/libero_noops" \
#     --output_dir="/store/real/maxjdu/repos/LiberoContinualLearning/datasets/libero_lerobot" \
#     --idx=1

# uv run examples/libero/convert_libero_data_to_lerobot.py \
#     --data_dir="/store/real/maxjdu/repos/LiberoContinualLearning/datasets/libero_noops" \
#     --output_dir="/store/real/maxjdu/repos/LiberoContinualLearning/datasets/libero_lerobot" \
#     --idx=2

# uv run examples/libero/convert_libero_data_to_lerobot.py \
#     --data_dir="/store/real/maxjdu/repos/LiberoContinualLearning/datasets/libero_noops" \
#     --output_dir="/store/real/maxjdu/repos/LiberoContinualLearning/datasets/libero_lerobot" \
#     --idx=3
# exit 

# computing normalization 
# CUDA_VISIBLE_DEVICES=6 uv run scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune-libero_goal_sequential \





## parallelization attmpet 
# source examples/libero/.venv/bin/activate
# export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
# # export MUJOCO_GL=glx
# MUJOCO_GL=egl python examples/libero/main_parallelized.py \
#   --host 127.0.0.1 \
#   --port 8765 \
#   --task-suite-name libero_goal \
#   --eval-upto-task 0 \
#   --num-trials-per-task 10 \
#   --num-parallel-envs 5 \
#   --num-steps-wait 0 \
#   --replan-steps 8 \
#   --base-dir runs/parallel_smoke_test

  # --use-batched-inference \



# 436 seconds with 25 parallel environments 
# 3600 seconds with 1 seqeutnail envirnments 

# there is an explicit feailure when we try to grab more than one environment 
  # --sequence \

# exit 

# python examples/libero/dummy_policy_server.py --port 8765

# This script evaluates it 