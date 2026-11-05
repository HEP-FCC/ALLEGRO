#!/bin/bash -l
#SBATCH --job-name=TRAPPIST-train
#SBATCH --output=/gpfs/scratch/ehpc1013/jfanini/slurm-logs/%j.out
#SBATCH --error=/gpfs/scratch/ehpc1013/jfanini/slurm-logs/%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc1013
#SBATCH --mail-type=all
#SBATCH --mail-user=jacopo.fanini@cern.ch

# modules to load the environment
module load miniforge/24.3.0-0
source "/gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh"
conda activate /gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/envs/mlpf
source "/gpfs/scratch/ehpc1013/lena/envs/hitpf-overlay/bin/activate"
cd "/gpfs/projects/ehpc1013/jfanini/TRAPPIST-train"
pwd

# nodes and gpus
NODES=${NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}

# --- Paths / configs (edit if needed) ---
TRAINING_ID="TRAPPIST-train"
DATA_DIR="/gpfs/scratch/ehpc1013/jfanini/datasets/TRAPPIST/202606-SW/strips2/"
CFG_NET="src/models/wrapper/example_mode_gatr.py"
N_PI0_EVENTS=567508
N_GAMMA_EVENTS=818194
MODEL_PREFIX="/gpfs/scratch/ehpc1013/jfanini/trained-models/TRAPPIST/${TRAINING_ID}/"
mkdir -p "${MODEL_PREFIX}"


# --- Weights & Biases (offline at BSC) ---
WANDB_NAME=${TRAINING_ID}
export WANDB_MODE=offline
export WANDB_PROJECT=allegro_photonpion
export WANDB_ENTITY=ml4hep
export WANDB_DIR=/home/cern/cern923515/wandb/${WANDB_NAME}
mkdir -p "${WANDB_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_CPU_BIND=none
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export NODE_RANK=$SLURM_PROCID
export NUM_PROCS=$((SLURM_NNODES * GPUS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "NODE_RANK: $NODE_RANK"
echo "NNODES: $SLURM_NNODES"
echo "NUM_PROCS: $NUM_PROCS"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

echo "We are about to train!"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  python -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" \
    --node_rank="$SLURM_PROCID" \
    -m src.train_lightning1 \
    --data-train "${DATA_DIR}" \
    --network-config "${CFG_NET}" \
    --model-prefix "${MODEL_PREFIX}" \
    --gpus 0,1,2,3 \
    --num-workers 32 \
    --batch-size 128  \
    --start-lr 0.001 \
    --num-epochs 2 \
    --optimizer ranger \
    --log-wandb \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-projectname "${WANDB_PROJECT}" \
    --wandb-displayname "${WANDB_NAME}" \
    --train-val-split 0.98 \
    --fetch-step 0.01 \
    --pi0-events "${N_PI0_EVENTS}" \
    --gamma-events "${N_GAMMA_EVENTS}"
    #--file-fraction 0.2


echo "Training season's over!"