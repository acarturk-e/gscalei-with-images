#!/bin/bash -i

get_random_seed() {
  awk 'BEGIN {
    # seed from CPU time in seconds
    srand()
    print int(rand() * 32767)
  }'
}

usage () {
  cat <<-END
Run GSCALEI with a 2-step autoencoder once in CPU.

$0 -h    Display this help message
$0 [-G] [-R | -r EPOCHS] [-x] [-L | -l EPOCHS] [-D | -d EPOCHS] [-s SEED] DATA_DIR LATENT_DIM
         Run GSCALEI once with DATA_DIR as data directory
         and LATENT_DIM as the output of the first encoder layer

Options:
  -G        Skip generate new data step
  -R        Skip autoenc step 1 training
  -r EPOCHS Load autoenc step 1 checkpoint and do EPOCH epochs
  -R        Skip autoenc step 1 training
  -r EPOCHS Load autoenc step 1 checkpoint and do EPOCH epochs
  -x        Train LDR on original input instead of step 1 output
  -L        Skip LDR training step
  -l EPOCHS Load LDR checkpoint and do EPOCH epochs for ALL pairs
  -D        Skip autoenc step 2 training
  -d EPOCHS Load autoenc step 2 checkpoint and do EPOCH epochs
  -s SEED   Send a specific PRNG seed to Python scripts

END
}


# We imitate slurm, except with world size 1
export SLURM_PROCID=0
export WORLD_SIZE=1


DATA_DIR=""
LATENT_DIM=0
NUM_WORKERS="8"
GLOBAL_BATCH_SIZE="16"
GLOBAL_SEED=$(get_random_seed)

# Generate data
NUM_BALLS="3"
NUM_SAMPLES="10000"
GRAPH_DEGREE="2"
IMAGE_SIZE="64"

## Train autoencoder step 1: Dim reduction (or, reconstruct)
AE1_LOAD_CHECKPOINT=""
AE1_LR="1e-3"
AE1_WEIGHT_DECAY="0.01"
AE1_MAX_EPOCHS="100"
AE1_CHECKPOINT_EPOCHS="5"

## Train LDR
LDR_LOAD_CHECKPOINT=""
LDR_LR="1e-5"
LDR_WEIGHT_DECAY="0.01"
LDR_MAX_EPOCHS="10"
LDR_CHECKPOINT_EPOCHS="5"

## Train autoencoder step 2: Disentangle
LAMBDA1="1"
AE2_LOAD_CHECKPOINT=""
AE2_LR="1e-3"
AE2_WEIGHT_DECAY="0.01"
AE2_MAX_EPOCHS="15"
AE2_CHECKPOINT_EPOCHS="5"

# Parse args to script
skip_data_gen=false
skip_ae1_train=false
ldr_on_x=false
skip_ae2_train=false


# Parse options
while getopts ":hGRr:xLl:Dd:s:" option; do
  case $option in
    h)
      usage
      exit 0
      ;;
    G)
      skip_data_gen=true
      ;;
    R)
      skip_ae1_train=true
      ;;
    r)
      AE1_LOAD_CHECKPOINT="--load-checkpoint"
      AE1_MAX_EPOCHS=$OPTARG
      ;;
    x)
      ldr_on_x=true
      ;;
    L)
      skip_ldr_train=true
      ;;
    l)
      LDR_LOAD_CHECKPOINT="--load-checkpoint"
      LDR_MAX_EPOCHS=$OPTARG
      ;;
    D)
      skip_ae2_train=true
      ;;
    d)
      AE2_LOAD_CHECKPOINT="--load-checkpoint"
      AE2_MAX_EPOCHS=$OPTARG
      ;;
    s)
      GLOBAL_SEED=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage
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

if [ -z "$2" ]; then
  echo "Error: Required positional argument (latent dim) is missing." >&2
  exit 1
fi
LATENT_DIM=$2


echo "$(date) Initializing conda"
eval "$(conda shell.bash hook)"
conda activate python-2024-10 || exit

if [ "$skip_data_gen" = true ]; then
  echo "$(date) Skipping data generation"
else
  echo "$(date) Starting data generation"
  python generate_data.py $NUM_BALLS $NUM_SAMPLES $DATA_DIR \
    --graph-degree $GRAPH_DEGREE --image-size $IMAGE_SIZE \
    --global-seed $GLOBAL_SEED || exit
fi

if [ "$skip_ae1_train" = true ]; then
  echo "$(date) Skipping autoencoder training step 1: Dim reduction"
else
  echo "$(date) Starting autoencoder training step 1: Dim reduction"
  python train_autoenc_reconstruct_cpu.py $DATA_DIR $LATENT_DIM \
    $AE1_LOAD_CHECKPOINT \
    --lr $AE1_LR --weight_decay $AE1_WEIGHT_DECAY --max-epochs $AE1_MAX_EPOCHS \
    --checkpoint-epochs $AE1_CHECKPOINT_EPOCHS --num-workers $NUM_WORKERS \
    --global-batch-size $GLOBAL_BATCH_SIZE --global-seed $GLOBAL_SEED || exit
  echo "$(date) Applying dim reduction to all x related data"
  python x_to_y.py $DATA_DIR $LATENT_DIM || exit
fi

if [ "$skip_ldr_train" = true ]; then
  echo "$(date) Skipping LDR training"
else
  if [ "$ldr_on_x" = true ]; then
    echo "$(date) Starting LDR training on original images"
    python train_ldr_cpu.py $DATA_DIR $LDR_LOAD_CHECKPOINT \
      --lr $LDR_LR --weight_decay $LDR_WEIGHT_DECAY --max-epochs $LDR_MAX_EPOCHS \
      --checkpoint-epochs $LDR_CHECKPOINT_EPOCHS --num-workers $NUM_WORKERS \
      --global-batch-size $GLOBAL_BATCH_SIZE --global-seed $GLOBAL_SEED || exit
    python dsx_to_dsy.py $DATA_DIR $LATENT_DIM || exit
  else
    echo "$(date) Starting LDR training on step 1 outputs"
    python train_ldr_on_y_cpu.py $DATA_DIR $LATENT_DIM $LDR_LOAD_CHECKPOINT \
      --lr $LDR_LR --weight_decay $LDR_WEIGHT_DECAY --max-epochs $LDR_MAX_EPOCHS \
      --checkpoint-epochs $LDR_CHECKPOINT_EPOCHS --num-workers $NUM_WORKERS \
      --global-batch-size $GLOBAL_BATCH_SIZE --global-seed $GLOBAL_SEED || exit
  fi
fi

if [ "$skip_ae2_train" = true ]; then
  echo "$(date) Skipping autoencoder training step 2: Disentangle"
else
  echo "$(date) Starting autoencoder training step 2: Disentangle"
  python train_autoenc_disentangle_cpu.py $DATA_DIR $LATENT_DIM \
    --lambda1 $LAMBDA1 \
    $AE2_LOAD_CHECKPOINT \
    --lr $AE2_LR --weight_decay $AE2_WEIGHT_DECAY --max-epochs $AE2_MAX_EPOCHS \
    --checkpoint-epochs $AE2_CHECKPOINT_EPOCHS --num-workers $NUM_WORKERS \
    --global-batch-size $GLOBAL_BATCH_SIZE --global-seed $GLOBAL_SEED || exit
fi

echo "$(date) All done"
