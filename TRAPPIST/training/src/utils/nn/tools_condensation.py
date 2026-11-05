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
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pathlib import Path
import os
import pickle
from src.models.gravnet_calibration import object_condensation_loss2
from src.layers.inference_oc import create_and_store_graph_output
#from src.layers.object_cond import onehot_particles_arr
from src.utils.logger_wandb import plot_clust
from src.utils.nn.tools import (
    lst_nonzero,
    clip_list,
    turn_grads_off,
    log_betas_hist,
    log_step_time,
    update_and_log_scheduler,
)
from src.utils.nn.tools import log_losses_wandb, update_dict

# class_names = ["other"] + [str(i) for i in onehot_particles_arr]  # quick fix


def train_regression(
    model,
    loss_func,
    opt,
    scheduler,
    train_loader,
    dev,
    epoch,
    steps_per_epoch=None,
    grad_scaler=None,
    tb_helper=None,
    logwandb=False,
    local_rank=0,
    current_step=0,  # current_step: used for logging correctly
    loss_terms=[],  # whether to only optimize the clustering loss
    args=None,
    args_model=None,
    alternate_steps=None,  # alternate_steps: after how many steps to switch between beta and clustering loss
    finetune_model=False,
):
    model.train()
    if finetune_model:
        model = turn_grads_off(model)

    clust_loss_only = loss_terms[0]
    add_energy_loss = loss_terms[1]  # whether to add energy loss to the clustering loss
    total_loss = 0
    num_batches = 0
    sum_abs_err = 0
    sum_sqr_err = 0
    count = 0
    step_count = current_step
    start_time = time.time()
    prev_time = time.time()
    loss_epoch_total, losses_epoch_total = [], []

    with tqdm.tqdm(train_loader) as tq:
        for batch_g, y in tq:
            label = y
            load_end_time = time.time()
            step_count += 1
            num_examples = label.shape[0]
            label = label.to(dev)
            opt.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                batch_g = batch_g.to(dev)
                if args.loss_regularization:
                    model_output, loss_regularizing_neig, loss_ll = model(batch_g)
                else:
                    if local_rank == 0:
                        model_output, e_cor, loss_ll = model(batch_g, step_count)
                    else:
                        model_output, e_cor, loss_ll = model(batch_g, 1)
                preds = model_output.squeeze()

                (loss, losses, loss_E, loss_E_frac_true,) = object_condensation_loss2(
                    batch_g,
                    model_output,
                    e_cor,
                    y,
                    clust_loss_only=clust_loss_only,
                    add_energy_loss=add_energy_loss,
                    calc_e_frac_loss=False,
                    q_min=args.qmin,
                    frac_clustering_loss=args.frac_cluster_loss,
                    attr_weight=args.L_attractive_weight,
                    repul_weight=args.L_repulsive_weight,
                    fill_loss_weight=args.fill_loss_weight,
                    use_average_cc_pos=args.use_average_cc_pos,
                    hgcalloss=args.hgcalloss,
                )
                loss = (
                    loss + 0.01 * loss_ll
                )  # + 1 / 20 * loss_E  # add energy loss # loss +
                if args.loss_regularization:
                    loss = loss + loss_regularizing_neig + loss_ll
                betas = (
                    torch.sigmoid(
                        torch.reshape(preds[:, args.clustering_space_dim], [-1, 1])
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                log_betas_hist(logwandb, local_rank, num_batches, betas, args)

            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()
            step_end_time = time.time()

            log_step_time(
                logwandb,
                num_batches,
                local_rank,
                load_end_time,
                prev_time,
                step_end_time,
            )
            loss = loss.item()

            num_batches += 1
            count += num_examples
            total_loss += loss

            update_and_log_scheduler(scheduler, args, loss, logwandb, local_rank, opt)

            log_losses_wandb(logwandb, num_batches, local_rank, losses, loss, loss_ll)
            if (local_rank == 0) and (num_batches % 500) == 0:
                dirname = os.path.dirname(args.model_prefix)
                if dirname and not os.path.exists(dirname):
                    os.makedirs(dirname)
                state_dict = (
                    model.module.state_dict()
                    if isinstance(
                        model,
                        (
                            torch.nn.DataParallel,
                            torch.nn.parallel.DistributedDataParallel,
                        ),
                    )
                    else model.state_dict()
                )
                PATH = args.model_prefix + "_checkpoint-%d.pt" % num_batches

                torch.save(
                    {
                        "model_state_dict": state_dict,
                        "optimizer_state_dict": opt.state_dict(),
                        "loss": loss,
                        "epoch": epoch,
                    },
                    PATH,
                )
            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break
            prev_time = time.time()

    time_diff = time.time() - start_time
    _logger.info(
        "Processed %d entries in total (avg. speed %.1f entries/s)"
        % (count, count / time_diff)
    )
    if count > 0 and num_batches > 0:
        _logger.info(
            "Train AvgLoss: %.5f, AvgMSE: %.5f, AvgMAE: %.5f"
            % (total_loss / num_batches, sum_sqr_err / count, sum_abs_err / count)
        )

        if scheduler and getattr(scheduler, "_update_per_step") == False:
            if args.lr_scheduler == "reduceplateau":
                scheduler.step(total_loss / num_batches)  # loss
                if logwandb and local_rank == 0:
                    wandb.log({"total_loss batch": total_loss / num_batches})
            else:
                scheduler.step()  # loss
            if logwandb and local_rank == 0:
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
    loss_func=None,
    steps_per_epoch=None,
    eval_metrics=[
        "mean_squared_error",
        "mean_absolute_error",
        "median_absolute_error",
        "mean_gamma_deviance",
    ],
    tb_helper=None,
    logwandb=False,
    energy_weighted=False,
    local_rank=0,
    step=0,
    loss_terms=[],
    args=None,
):
    model.eval()

    total_loss = 0
    num_batches = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    all_val_loss, all_val_losses = [], []
    step = 0
    df_showers = []
    df_showers_pandora = []
    df_showes_db = []
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for batch_g, y in tq:
                batch_g = batch_g.to(dev)
                label = y
                num_examples = label.shape[0]
                label = label.to(dev)
                if args.loss_regularization:
                    model_output, loss_regularizing_neig, loss_ll = model(batch_g)
                else:
                    if args.predict:
                        step_plotting = 0
                    else:
                        step_plotting = 1
                    model_output, e_corr, loss_ll = model(batch_g, step_plotting)
                (
                    loss,
                    losses,
                    loss_E_frac,
                    loss_E_frac_true,
                ) = object_condensation_loss2(
                    batch_g,
                    model_output,
                    e_corr,
                    y,
                    frac_clustering_loss=0,
                    q_min=args.qmin,
                    clust_loss_only=args.clustering_loss_only,
                    use_average_cc_pos=args.use_average_cc_pos,
                    hgcalloss=args.hgcalloss,
                )
                step += 1

                num_batches += 1
                count += num_examples
                total_loss += loss * num_examples

                tq.set_postfix(
                    {
                        "Loss": "%.5f" % loss,
                        "AvgLoss": "%.5f" % (total_loss / count),
                    }
                )
                log_losses_wandb(
                    logwandb, num_batches, local_rank, losses, loss, loss_ll, val=True
                )
                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break
                if args.predict:
                    model_output1 = torch.cat((model_output, e_corr.view(-1, 1)), dim=1)
                    (
                        df_batch,
                        df_batch_pandora,
                        df_batch1,
                    ) = create_and_store_graph_output(
                        batch_g,
                        model_output1,
                        y,
                        local_rank,
                        step,
                        epoch,
                        path_save=args.model_prefix + "showers_df_evaluation",
                        store=True,
                        predict=True,
                        tracks=args.tracks,
                    )
                    df_showers.append(df_batch)
                    df_showers_pandora.append(df_batch_pandora)
                    df_showes_db.append(df_batch1)
    # calculate showers at the end of every epoch
    if logwandb and local_rank == 0:
        if args.predict:
            from src.layers.inference_oc import store_at_batch_end
            import pandas as pd

            df_showers = pd.concat(df_showers)
            df_showers_pandora = pd.concat(df_showers_pandora)
            df_showes_db = pd.concat(df_showes_db)
            store_at_batch_end(
                path_save=args.model_prefix + "showers_df_evaluation",
                df_batch=df_showers,
                df_batch_pandora=df_showers_pandora,
                df_batch1=df_showes_db,
                step=0,
                predict=True,
            )
        else:
            model_output1 = torch.cat((model_output, e_corr.view(-1, 1)), dim=1)
            create_and_store_graph_output(
                batch_g,
                model_output1,
                y,
                local_rank,
                step,
                epoch,
                path_save=args.model_prefix + "showers_df_evaluation",
                store=True,
                predict=False,
                tracks=args.tracks,
            )
    if logwandb and local_rank == 0:

        wandb.log(
            {
                "loss val end regression": loss,
                "loss val end lv": losses[0],
                "loss val end beta": losses[1],
                "loss val end E": losses[2],
                "loss val end  X": losses[3],
                "loss val end attractive": losses[12],
                "loss val end repulsive": losses[13],
            }
        )

    time_diff = time.time() - start_time
    _logger.info(
        "Processed %d entries in total (avg. speed %.1f entries/s)"
        % (count, count / time_diff)
    )

    if count > 0:
        if for_training:
            return total_loss / count
        else:
            # convert 2D labels/scores
            observers = {k: _concat(v) for k, v in observers.items()}
            return total_loss / count, scores, labels, observers
