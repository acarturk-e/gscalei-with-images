"""TODO: We are using checkpoints, but they don't have metadata.
More importantly, since there are multiple models per run, we need a
include-when-exists logic---this is currently missing."""

import argparse
import logging
import math
import os

# For DDP utils
from socket import gethostname

import numpy as np

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy

# Setting these flags True makes A100 training a lot faster:
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# For ddp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# For data
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

# Local imports
from cdsde import LdrNn, class_density_from_ldr, score_diff_from_ldr


def get_y(data_dir: str, latent_dim: int) -> tuple[Tensor, Tensor, Tensor]:
    """Returns tuple `(ys_obs, ys_hard_1, ys_hard_2)`"""
    ys_obs = torch.load(os.path.join(data_dir, f"ys_obs_{latent_dim}.pth"), weights_only=True)
    ys_hard_1 = torch.load(os.path.join(data_dir, f"ys_hard_1_{latent_dim}.pth"), weights_only=True)
    ys_hard_2 = torch.load(os.path.join(data_dir, f"ys_hard_2_{latent_dim}.pth"), weights_only=True)
    return ys_obs, ys_hard_1, ys_hard_2


def create_logger(log_dir: str, latent_dim: int) -> logging.Logger:
    """Create a logger that writes to a log file (DEBUG level) and stdout (INFO level)"""
    os.makedirs(log_dir, exist_ok=True)
    h_out = logging.StreamHandler()
    h_out.setLevel(logging.INFO)
    h_file = logging.FileHandler(os.path.join(log_dir, f"train_ldr_on_y_{latent_dim}_cpu.log"), mode="w")
    h_file.setLevel(logging.DEBUG)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(module)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[h_out, h_file])
    logger = logging.getLogger(__name__)
    return logger


def train_ldr(
    rank: int,
    args: argparse.Namespace,
    logger: logging.Logger | None,
    x1: Tensor, x2: Tensor,
    ldr_name: str,
) -> torch.nn.Module:
    """Learns the log density ratio between the data sets `x1` and `x2`

    Log density ratio model is the `LdrNn` class.
    Requires balanced data. Training is distributed.

    `ldr_name` is the identifier appended to the parameter save file names
    for distinguishing between different LDR models during a single run."""
    if rank == 0:
        assert logger is not None
        logger.info(f"Starting LDR for identifier {ldr_name}")

    n_samples1 = x1.shape[0]
    n_samples2 = x2.shape[0]
    n_samples = n_samples1 + n_samples2
    image_shape = x1.shape[1:]
    assert image_shape == x2.shape[1:]

    xs = torch.cat((x1, x2), 0)
    ys = torch.zeros((n_samples,), dtype=xs.dtype)
    ys[n_samples1:] = 1.0

    [train_dataset, valid_dataset] = random_split(TensorDataset(xs, ys), [0.9, 0.1])

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed)
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False, # IMPORTANT set this to "False" since sampler's shuffle is True
        sampler=train_sampler,
        num_workers=args.num_workers, # This should be equal to the number of CPUs set per task
        pin_memory=True,
        drop_last=True) # Set "True" to prevent uneven splits
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=valid_sampler,
        num_workers=args.num_workers, # This should be equal to the number of CPUs set per task
        pin_memory=True,
        drop_last=True) # Set "True" to prevent uneven splits


    # Create model
    ldr_model = DDP(LdrNn(math.prod(image_shape)))

    if args.load_checkpoint:
        ldr_model.load_state_dict(torch.load(os.path.join(args.data_dir, "ldr_model_" + ldr_name + ".pth"), weights_only=True))

    opt = torch.optim.Adam(
        ldr_model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay)
    # lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)

    def loss_fn(xb: Tensor, yb: Tensor) -> Tensor:
        # Our model estimates log density ratios
        ldr = ldr_model(xb)
        # ... which can be used to compute class probabilities via softmax
        cd_est_b = class_density_from_ldr(ldr)
        # ... which lets us compute cross entropy
        loss = binary_cross_entropy(cd_est_b, yb, reduction="sum")
        return loss


    for epoch in range(args.max_epochs):
        # Training
        ldr_model.train()
        running_loss = log_steps = 0
        train_sampler.set_epoch(epoch)
        for (xb, yb) in train_loader:
            opt.zero_grad()
            assert isinstance(xb, Tensor)
            assert isinstance(yb, Tensor)

            loss = loss_fn(xb, yb)
            loss.backward()
            opt.step()

            log_steps += 1
            running_loss += loss.item()

        avg_loss = torch.tensor(running_loss / log_steps)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / dist.get_world_size()

        # End of training
        dist.barrier()

        if rank == 0:
            assert logger is not None
            logger.info(f"(step={epoch}), Train Loss: {avg_loss:.5f}")

        # Validation
        ldr_model.eval()
        running_loss = log_steps = 0
        valid_sampler.set_epoch(epoch)
        for (xb, yb) in valid_loader:
            assert isinstance(xb, Tensor)
            assert isinstance(yb, Tensor)

            with torch.no_grad():
                loss = loss_fn(xb, yb)

            log_steps += 1
            running_loss += loss.item()

        avg_loss = torch.tensor(running_loss / log_steps)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss.item() / dist.get_world_size()

        # End of validation
        dist.barrier()

        if rank == 0:
            assert logger is not None
            logger.info(f"(step={epoch}), Validation Loss: {avg_loss:.5f}")
            if args.checkpoint_epochs != -1 and epoch % args.checkpoint_epochs == 0:
                torch.save(ldr_model.state_dict(), os.path.join(args.data_dir, "ldr_model_" + ldr_name + ".pth"))

    dist.barrier()
    ldr_model.eval()
    ldr_model.requires_grad_(False)
    if rank == 0:
        assert logger is not None
        torch.save(ldr_model.state_dict(), os.path.join(args.data_dir, "ldr_model_" + ldr_name + ".pth"))
        logger.info("Done!")

    return ldr_model


def main(
    rank: int,
    args: argparse.Namespace
):
    assert args.global_batch_size % dist.get_world_size() == 0, "Global batch size must split evenly among ranks."

    # Set your random seed for experiment reproducibility.
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)

    logger = None
    if rank == 0:
        logger = create_logger(args.data_dir, args.latent_dim)
        logger.info(f"Experiment directory created at {args.data_dir}")
        logger.info(f"Batch size per rank: {args.global_batch_size // dist.get_world_size()}")

    # Read data and create distributive data loader
    ys_obs, ys_hard_1, ys_hard_2 = get_y(args.data_dir, args.latent_dim)
    n = ys_hard_1.shape[0]
    assert n == ys_hard_2.shape[0]
    image_shape = ys_obs.shape[1:]
    assert image_shape == ys_hard_1.shape[2:] and image_shape == ys_hard_2.shape[2:]

    ## Compute log density ratio between

    # 1. Hard intervention pairs
    ldr_bw_hards = list[torch.nn.Module]()
    for env_idx in range(n):
        ldr_bw_hards.append(train_ldr(
            rank, args, logger,
            ys_hard_1[env_idx], ys_hard_2[env_idx],
            f"bw_hards_{args.latent_dim}_{env_idx}"))

    # 2. Hard interventions (one set suffices) and observational domain
    ldr_hard_obs = list[torch.nn.Module]()
    for env_idx in range(n):
        ldr_hard_obs.append(train_ldr(
            rank, args, logger,
            ys_obs, ys_hard_1[env_idx],
            f"hard_obs_{args.latent_dim}_{env_idx}"))

    # We no longer need parallelism
    if rank != 0: return

    assert logger is not None

    ## Use the LDR models to estimate the score difference function
    ## on observational data points.
    logger.info("Starting score difference computation.")

    # 1. Hard intervention pairs
    dsys_bw_hards = torch.zeros((n,) + ys_obs.shape)
    for env_idx in range(n):
        ldr_model = ldr_bw_hards[env_idx]
        with torch.no_grad():
            dsys_bw_hards[env_idx] = score_diff_from_ldr(ldr_model.forward, ys_obs)
    torch.save(dsys_bw_hards, os.path.join(args.data_dir, f"dsys_bw_hards_{args.latent_dim}.pth"))

    # 2. Hard interventions (one set suffices) and observational domain
    dsys_hard_obs = torch.zeros((n,) + ys_obs.shape)
    for env_idx in range(n):
        ldr_model = ldr_bw_hards[env_idx]
        with torch.no_grad():
            dsys_hard_obs[env_idx] = score_diff_from_ldr(ldr_model.forward, ys_obs)
    torch.save(dsys_hard_obs, os.path.join(args.data_dir, f"dsys_hard_obs_{args.latent_dim}.pth"))

    logger.info("Computed and saved the score difference samples.")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_dir", type=str, metavar="DIR", help="Directory to store data and logs.")
    parser.add_argument("latent_dim", type=int, metavar="DIM", help="Dimension of the autoencoder latent space.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate of optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01, metavar="LAMBDA", help="Weight decay.")
    parser.add_argument("--max-epochs", type=int, default=10, metavar="EPOCHS", help="Number of epochs to run for each LDR model.")
    parser.add_argument("--checkpoint-epochs", type=int, default=-1, metavar="EPOCHS", help="Epoch period of checkpoint saves. Set to -1 to not save checkpoints.")
    parser.add_argument("--load-checkpoint", action="store_true", help="Loads all model parameters from checkpoints.")
    parser.add_argument("--num-workers", type=int, default=8, metavar="N", help="Number of CPUs per process")
    parser.add_argument("--global-batch-size", type=int, default=128, metavar="SIZE", help="Global, i.e., across all processes, batch size")
    parser.add_argument("--global-seed", type=int, default=9724)
    args = parser.parse_args()

    rank          = int(os.environ["SLURM_PROCID"])
    world_size    = int(os.environ["WORLD_SIZE"])
    file_store    = dist.FileStore(os.path.join(args.data_dir, "_train_ldr_file_store"), 1)  # type: ignore

    print(f"Hello from rank {rank} of {world_size} on {gethostname()}", flush=True)

    dist.init_process_group("gloo", store=file_store, rank=rank, world_size=world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    main(rank, args)
