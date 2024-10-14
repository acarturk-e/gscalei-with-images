#!/bin/bash -i
#SBATCH --job-name=gscalei_run1x # create a short name for your job
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=2      # set this equals to the number of gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:2             # number of allocated gpus per node
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=acarte@rpi.edu    # change this to your email!

get_random_seed() {
  awk 'BEGIN {
    # seed from CPU time in seconds
    srand()
    print int(rand() * 32767)
  }'
}

usage () {
  cat <<-END
Run GSCALEI once in NPL cluster in CCI @ RPI using slurm and conda.

$0 -h    Display this help message
$0 [-G] [-G|-g EPOCHS] [-G|-g EPOCHS] [-s SEED] DATA_DIR
         Run GSCALEI once with DATA_DIR as data directory

Options:
  -G        Skip generate data step
  -L        Skip LDR training step
  -l EPOCHS Load LDR checkpoint and do EPOCH epochs for ALL pairs
  -A        Skip autoenc training step
  -a EPOCHS Load autoenc checkpoint and do EPOCH epochs
  -s SEED   Send a specific PRNG seed to Python scripts

END
}

# export your rank 0 information (its address and port)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR

## IMPORTANT NOTE: I don't do `module load xxx` at all, which is intentional!

DATA_DIR=""
NUM_WORKERS="8"
# Global batch size is (4 nodes x 2 gpus) x local batch size.
GLOBAL_BATCH_SIZE="128"
GLOBAL_SEED=$(get_random_seed)

# Generate data
NUM_BALLS="5"
NUM_SAMPLES="25000"
GRAPH_DEGREE="2"
IMAGE_SIZE="64"

## Train LDR
LDR_LOAD_CHECKPOINT=""
LDR_LR="1e-5"
LDR_WEIGHT_DECAY="0.01"
LDR_MAX_EPOCHS="10"
LDR_CHECKPOINT_EPOCHS="10"

## Train autoencoder
LAMBDA1="1.0"
MAIN_LOSS_EPOCHS="4"
AE_LOAD_CHECKPOINT=""
AE_LR="3e-6"
AE_WEIGHT_DECAY="0.01"
AE_MAX_EPOCHS="250"
AE_CHECKPOINT_EPOCHS="10"

# Parse args to script
skip_data_gen=false
skip_ldr_train=false
skip_ae_train=false

# Parse options
while getopts ":hGLl:Aa:s:" option; do
  case $option in
    h)
      echo $USAGE
      exit 0
      ;;
    G)
      skip_data_gen=true
      ;;
    L)
      skip_ldr_train=true
      ;;
    l)
      LDR_LOAD_CHECKPOINT="--load-checkpoint"
      LDR_MAX_EPOCHS=$OPTARG
      ;;
    A)
      skip_ae_train=true
      ;;
    a)
      AE_LOAD_CHECKPOINT="--load-checkpoint"
      AE_MAX_EPOCHS=$OPTARG
      ;;
    s)
      GLOBAL_SEED=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      echo $USAGE
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      echo $USAGE
      exit 1
      ;;
  esac
done

# After parsing options, shift to process additional arguments
shift $((OPTIND - 1))

# Any remaining arguments after options are positional.
# We _require_ data dir as a positional argument
if [ -z "$1" ]; then
  echo "Error: Required positional argument (data dir) is missing." >&2
  exit 1
fi
DATA_DIR=$1


echo "$(date) Initializing conda"
conda activate python-2024-09 || exit

if [ "$skip_data_gen" = true ]; then
  echo "$(date) Skipping data generation"
else
  echo "$(date) Starting data generation"
  # Note that the first part (i.e., data generation) should not be run distributed
  srun --nodes=1 --ntasks-per-node=1 --gres=gpu:1 \
      python generate_data.py $NUM_BALLS $NUM_SAMPLES $DATA_DIR \
      --graph-degree $GRAPH_DEGREE --image-size $IMAGE_SIZE \
      --global-seed $GLOBAL_SEED || exit
fi

if [ "$skip_ldr_train" = true ]; then
  echo "$(date) Skipping LDR training"
else
  echo "$(date) Starting LDR training"
  srun \
      python train_ldr_cci.py $DATA_DIR $LDR_LOAD_CHECKPOINT \
      --lr $LDR_LR --weight_decay $LDR_WEIGHT_DECAY --max-epochs $LDR_MAX_EPOCHS \
      --checkpoint-epochs $LDR_CHECKPOINT_EPOCHS --num-workers $NUM_WORKERS \
      --global-batch-size $GLOBAL_BATCH_SIZE --global-seed $GLOBAL_SEED || exit
fi

if [ "$skip_ae_train" = true ]; then
  echo "$(date) Skipping autoencoder training"
else
  echo "$(date) Starting autoencoder training"
  srun \
      python train_autoenc_cci.py $DATA_DIR $AE_LOAD_CHECKPOINT \
      --lr $AE_LR --weight_decay $AE_WEIGHT_DECAY --max-epochs $AE_MAX_EPOCHS \
      --lambda1 $LAMBDA1 --main-loss-epochs $MAIN_LOSS_EPOCHS \
      --checkpoint-epochs $AE_CHECKPOINT_EPOCHS --num-workers $NUM_WORKERS \
      --global-batch-size $GLOBAL_BATCH_SIZE --global-seed $GLOBAL_SEED || exit
fi

echo "$(date) All done"

