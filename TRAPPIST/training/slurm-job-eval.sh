#!/bin/bash -l
#SBATCH --job-name=TRAPPIST-train
#SBATCH --output=/gpfs/scratch/ehpc1013/jfanini/slurm-logs/%j.out
#SBATCH --error=/gpfs/scratch/ehpc1013/jfanini/slurm-logs/%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:1
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
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

# --- Paths / configs (edit if needed) ---
EVAL_ID="TRAPPIST-eval"
DATA_DIR="/gpfs/scratch/ehpc1013/jfanini/datasets/TRAPPIST/202606-SW/strips2-test-reduced/*.parquet"
CFG_NET="src/models/wrapper/example_mode_gatr.py"
# n_pi0 and n_gamma are not used in evaluation, but they have to be defined in order to run it
N_PI0_EVENTS=567508 
N_GAMMA_EVENTS=818194
CKPT_FOLDER="/gpfs/scratch/ehpc1013/jfanini/trained-models/TRAPPIST/"
CKPT_FILE="TRAPPIST-test/_epoch=1_step=988.ckpt"
CKPT="${CKPT_FOLDER}${CKPT_FILE}"
MODEL_PREFIX="/gpfs/scratch/ehpc1013/jfanini/evaluation/TRAPPIST/${EVAL_ID}/"
mkdir -p "${MODEL_PREFIX}"


# --- Weights & Biases (offline at BSC) ---
WANDB_NAME=${EVAL_ID}
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
    --predict \
    --data-test "${DATA_DIR}" \
    --network-config "${CFG_NET}" \
    --gpus 0 \
    --num-workers 128  \
    --load-model-weights "${CKPT}" \
    --fetch-step 0.1 \
    --model-prefix "${MODEL_PREFIX}" \
    --gamma-events "${N_GAMMA_EVENTS}" \
    --pi0-events "${N_PI0_EVENTS}" 


echo "Evaluation is completed!"