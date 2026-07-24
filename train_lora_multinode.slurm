#!/bin/bash
#SBATCH --job-name=flux-lora-multinode
#SBATCH --partition=gpu
#SBATCH --nodes=2                      # <-- set number of nodes
#SBATCH --ntasks-per-node=1            # torchrun itself is 1 task/node; it spawns the GPU processes
#SBATCH --gres=gpu:2                   # <-- set GPUs per node
#SBATCH --cpus-per-task=16             # <-- adjust to your node's CPU count / 4 dataloader workers per GPU is a reasonable target
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err



# ── Cluster / networking setup ────────────────────────────────
# Number of GPUs per node — must match --gres=gpu:N above
GPUS_PER_NODE=2

# Pick the first node in the allocation as the rendezvous host
NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
MASTER_ADDR=${NODES[0]}
MASTER_PORT=29500   # any free port; change if it collides with another job on the same node

echo "Nodes in job: ${NODES[@]}"
echo "Master node : $MASTER_ADDR:$MASTER_PORT"
echo "Total GPUs  : $((SLURM_NNODES * GPUS_PER_NODE))"

# ── NCCL settings ──────────────────────────────────────────────
# Uncomment / adjust these if your cluster's inter-node network needs it
# (common on clusters without RDMA/InfiniBand, or with multiple NICs):
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

# ── Activate environment ───────────────────────────────────────
source ~/.bashrc
conda activate matclip

# ── Launch: one srun task per node, each running torchrun ─────
# torchrun reads SLURM's per-task env (SLURM_NODEID, SLURM_PROCID, etc. are
# NOT what torchrun uses directly — we pass --node_rank explicitly per node
# via $SLURM_NODEID so each node knows its rank in the rendezvous).
srun --ntasks-per-node=1 bash -c "
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=\$SLURM_NODEID \
    --rdzv_id=\$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_flux_lora_multigpu.py
"

