import numpy as np
import awkward as ak
import tqdm
import time
import torch
from collections import defaultdict, Counter
from src.utils.metrics import evaluate_metrics
from src.data.tools import _concat
from src.logger.logger import _logger
import wandb
from src.utils.nn.tools import (
    _check_scales_centers,
    upd_dict,
    update_dict,
    getEffSigma,
    log_losses_wandb,
    log_losses_wandb_tracking,
    update_and_log_scheduler,
    lst_nonzero,
    clip_list,
    turn_grads_off,
    log_step_time,
    log_betas_hist,
    _flatten_label,
    _flatten_preds,
)


def train_regression(
    model,
    opt,
    scheduler,
    train_loader,
    dev,
    epoch,
    steps_per_epoch=None,
    grad_scaler=None,
    local_rank=0,
    current_step=0,
    args=None,
):
    model.train()
    num_batches = 0
    step_count = current_step
    model.module.mod.on_train_epoch_start()
    model.module.mod.current_epoch = epoch
    model.module.mod.local_rank = local_rank

    with tqdm.tqdm(train_loader) as tq:
        for batch_idx, batch in enumerate(tq):
            step_count += 1
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                loss = model.module.mod.training_step(batch, batch_idx)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, "_update_per_step", False):
                scheduler.step()

            loss = loss.item()

            num_batches = num_batches + 1
            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break
    model.module.mod.on_train_epoch_end()
    if scheduler and getattr(scheduler, "_update_per_step") == False:
        if args.lr_scheduler == "reduceplateau":
            scheduler.step(loss)  # loss
        else:
            scheduler.step()  # loss
        if local_rank == 0:
            if args.lr_scheduler == "reduceplateau":
                wandb.log({"lr": opt.param_groups[0]["lr"]})
            else:
                wandb.log({"lr": scheduler.get_last_lr()[0]})
    return step_count


def evaluate_regression(
    model,
    test_loader,
    dev,
    epoch,
    for_training=True,
    steps_per_epoch=None,
    local_rank=0,
    args=None,
):
    model.eval()
    total_loss = 0
    num_batches = 0
    model.module.mod.dev = dev
    model.module.mod.on_validation_epoch_start()
    model.module.mod.current_epoch = epoch
    model.module.mod.local_rank = local_rank
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for batch_index, batch in enumerate(tq):
                loss = model.module.mod.validation_step(batch, batch_index)

                total_loss = total_loss + loss.item()
                num_batches += 1
                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break
    model.module.mod.on_validation_epoch_end()
    if for_training:
        return total_loss / num_batches
