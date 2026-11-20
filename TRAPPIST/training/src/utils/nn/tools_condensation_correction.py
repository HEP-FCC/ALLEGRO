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
from src.layers.object_cond import (
    onehot_particles_arr,
    get_clustering,
    calc_LV_Lbeta_inference,
)
from src.utils.logger_wandb import plot_clust
from src.layers.inference_oc import obtain_clustering
from src.utils.nn.tools_condensation import lst_nonzero, clip_list
class_names = ["other"] + [str(i) for i in onehot_particles_arr]  # quick fix


def train_regression(
    model1,
    model2, 
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
):
    model1.eval()
    model2.train()
    # print("starting to train")
    iterator = iter(train_loader)
    g, y = next(iterator)
    iterator = iter(train_loader)
    # print("LEN DATALOADER", g)
    # print(y)
    data_config = train_loader.dataset.config
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
            load_end_time = time.time()
            label = y
            step_count += 1
            num_examples = label.shape[0]
            label = label.to(dev)
            opt.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                batch_g = batch_g.to(dev)
                calc_e_frac_loss = (num_batches % 250) == 0
                if local_rank == 0:
                    model_output = model1(batch_g, step_count)
                    labels_clustering = obtain_clustering(batch_g, model_output, y)
                    correction_factors = model2(batch_g,labels_clustering)
    
                else:
                    model_output = model1(batch_g, 1)
                    labels_clustering = obtain_clustering(batch_g, model_output, y)
                    correction_factors = model2(batch_g,labels_clustering)
                
                
                loss = loss 
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()
            step_end_time = time.time()
            loss = loss.item()

            num_batches += 1
            count += num_examples
            total_loss += loss
            if scheduler and getattr(scheduler, "_update_per_step"):
                if args.lr_scheduler == "reduceplateau":
                    scheduler.step(loss)  # loss
                else:
                    scheduler.step()  # loss
                if logwandb and local_rank == 0:
                    if args.lr_scheduler == "reduceplateau":
                        wandb.log({"lr": opt.param_groups[0]["lr"]})
                    else:
                        wandb.log({"lr": scheduler.get_last_lr()[0]})

            if tb_helper:
                print("tb_helper!", tb_helper)
                tb_helper.write_scalars(
                    [
                        ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ]
                )
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(
                            model_output=model_output,
                            model=model,
                            epoch=epoch,
                            i_batch=num_batches,
                            mode="train",
                        )

            if logwandb and ((num_batches - 1) % 10) == 0 and local_rank == 0:
                pid_true, pid_pred = losses[7], losses[8]
                loss_epoch_total.append(loss)
                losses_epoch_total.append(losses)
                
                wandb.log(
                    {
                        "loss regression": loss,
                        "loss lv": losses[0],
                        "loss beta": losses[1],
                        "loss E": losses[2],
                        "loss X": losses[3],
                        "loss PID": losses[4],
                        "loss momentum": losses[5],
                        "loss mass (not us. for opt.)": losses[6],
                        "inter-clustering loss": losses[10],
                        "filling loss": losses[11],
                        "loss attractive": losses[12],
                        "loss repulsive": losses[13],
                        "loss alpha coord": losses[14],
                        "loss beta zeros": losses[15],
                    }
                )  # , step=step_count)

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

        if tb_helper:
            tb_helper.write_scalars(
                [
                    ("Loss/train (epoch)", total_loss / num_batches, epoch),
                    ("MSE/train (epoch)", sum_sqr_err / count, epoch),
                    ("MAE/train (epoch)", sum_abs_err / count, epoch),
                ]
            )
            if tb_helper.custom_fn:
                with torch.no_grad():
                    tb_helper.custom_fn(
                        model_output=model_output,
                        model=model,
                        epoch=epoch,
                        i_batch=-1,
                        mode="train",
                    )
            # update the batch state
            tb_helper.batch_train_count += num_batches
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


def update_dict(dict1, dict2):
    for key in dict2:
        if key not in dict1:
            dict1[key] = 0.0
        dict1[key] += dict2[key]
    return dict1


def getEffSigma(data_for_hist, percentage=0.683, bins=1000):
    bins = np.linspace(0, 200, bins + 1)
    theHist, bin_edges = np.histogram(data_for_hist, bins=bins, density=True)
    wmin = 0.2
    wmax = 1.0
    epsilon = 0.01
    point = wmin
    weight = 0.0
    points = []
    sums = []
    # fill list of bin centers and the integral up to those point
    for i in range(len(bin_edges) - 1):
        weight += theHist[i] * (bin_edges[i + 1] - bin_edges[i])
        points.append([(bin_edges[i + 1] + bin_edges[i]) / 2, weight])
        sums.append(weight)

    low = wmin
    high = wmax
    width = 100
    for i in range(len(points)):
        for j in range(i, len(points)):
            wy = points[j][1] - points[i][1]
            # print(wy)
            if abs(wy - percentage) < epsilon:
                # print("here")
                wx = points[j][0] - points[i][0]
                if wx < width:
                    low = points[i][0]
                    high = points[j][0]
                    # print(points[j][0], points[i][0], wy, wx)
                    width = wx
                    ii = i
                    jj = j
    # print(low, high)
    return 0.5 * (high - low), low, high

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

    data_config = test_loader.dataset.config

    center, scales = _check_scales_centers(iter(test_loader.dataset))
    total_loss = 0
    num_batches = 0
    sum_sqr_err = 0
    sum_abs_err = 0
    clust_loss_only = loss_terms[0]
    add_energy_loss = loss_terms[1]  # whether to add energy loss to the clustering loss
    count = 0
    scores = []
    results = []  # resolution results
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    all_val_loss, all_val_losses = [], []
    step = 0
    df_showers = []
    df_showers_pandora = []
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for batch_g, y in tq:
                calc_e_frac_loss = num_batches % 10 == 0
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
                    model_output = model(batch_g, step_plotting)
                (
                    loss,
                    losses,
                    loss_E_frac,
                    loss_E_frac_true,
                ) = object_condensation_loss2(
                    batch_g,
                    model_output,
                    y,
                    frac_clustering_loss=0,
                    q_min=args.qmin,
                    clust_loss_only=args.clustering_loss_only,
                    use_average_cc_pos=args.use_average_cc_pos,
                    hgcalloss=args.hgcalloss,
                )
                #! create output graph with shower id ndata and store it for each event
                # if args.store_output:
                # print("calculating clustering and matching showers")
                # if step == 0 and local_rank == 0:
                #     create_and_store_graph_output(
                #         batch_g,
                #         model_output,
                #         y,
                #         local_rank,
                #         step,
                #         epoch,
                #         path_save=args.model_prefix + "/showers_df",
                #         store=True,
                #     )
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

                losses_cpu = [
                    x.detach().to("cpu") if isinstance(x, torch.Tensor) else x
                    for x in losses
                ]
                all_val_losses.append(losses_cpu)
                all_val_loss.append(loss.detach().to("cpu").item())
                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break
                if args.predict:
                    df_batch, df_batch_pandora = create_and_store_graph_output(
                        batch_g,
                        model_output,
                        y,
                        local_rank,
                        step,
                        epoch,
                        path_save=args.model_prefix + "/showers_df_evaluation",
                        store=False,
                        predict=True,
                    )
                    df_showers.append(df_batch)
                    df_showers_pandora.append(df_batch_pandora)
    # calculate showers at the end of every epoch
    if logwandb and local_rank == 0:
        if args.predict:
            from src.layers.inference_oc import store_at_batch_end
            import pandas as pd

            df_showers = pd.concat(df_showers)
            df_showers_pandora = pd.concat(df_showers_pandora)
            store_at_batch_end(
                path_save=args.model_prefix + "/showers_df_evaluation",
                df_batch=df_showers,
                df_batch_pandora=df_showers_pandora,
                step=0,
                predict=True,
            )
        else:
            create_and_store_graph_output(
                batch_g,
                model_output,
                y,
                local_rank,
                step,
                epoch,
                path_save=args.model_prefix + "/showers_df_evaluation",
                store=True,
                predict=False,
            )
    if logwandb and local_rank == 0:
        # pid_true, pid_pred = torch.cat(
        #     [torch.tensor(x[7]) for x in all_val_losses]
        # ), torch.cat([torch.tensor(x[8]) for x in all_val_losses])
        # pid_true, pid_pred = pid_true.tolist(), pid_pred.tolist()
        wandb.log(
            {
                "loss val regression": loss,
                "loss val lv": losses[0],
                "loss val beta": losses[1],
                "loss val E": losses[2],
                "loss val X": losses[3],
                "loss val attractive": losses[12],
                "loss val repulsive": losses[13],
                # "conf_mat_val": wandb.plot.confusion_matrix(
                #     y_true=pid_true, preds=pid_pred, class_names=class_names
                # ),
            }
        )  # , step=step)
        # if clust_loss_only and calc_e_frac_loss:
        #     wandb.log(
        #         {
        #             "loss e frac val": loss_E_frac,
        #             "loss e frac true val": loss_E_frac_true,
        #         }
        #     )
        # ks = sorted(list(all_val_losses[0][9].keys()))
        # concatenated = {}
        # for key in ks:
        #     concatenated[key] = np.concatenate([x[9][key] for x in all_val_losses])
        # tables = {}
        # for key in ks:
        #     tables[key] = concatenated[
        #         key
        #     ]  # wandb.Table(data=[[x] for x in concatenated[key]], columns=[key])
        # wandb.log(
        #     {
        #         "val " + key: wandb.Histogram(clip_list(tables[key]), num_bins=100)
        #         for key in ks
        #     }
        # )  # , step=step)

    time_diff = time.time() - start_time
    _logger.info(
        "Processed %d entries in total (avg. speed %.1f entries/s)"
        % (count, count / time_diff)
    )

    # scores = np.concatenate(scores)
    # labels = {k: _concat(v) for k, v in labels.items()}
    # metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    # _logger.info('Evaluation metrics: \n%s', '\n'.join(
    #    ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))
    if count > 0:
        if for_training:
            return total_loss / count
        else:
            # convert 2D labels/scores
            observers = {k: _concat(v) for k, v in observers.items()}
            return total_loss / count, scores, labels, observers


def _check_scales_centers(iterator):
    regress_items = ["part_theta", "part_phi"]
    centers = np.zeros(2)
    scales = np.zeros(2)
    for ii, item in enumerate(regress_items):
        centers[ii] = iterator._data_config.preprocess_params[item]["center"]
        scales[ii] = iterator._data_config.preprocess_params[item]["scale"]
    return centers, scales


def upd_dict(d, small_dict):
    for k in small_dict:
        if k not in d:
            d[k] = []
        d[k] += small_dict[k]
    return d


def plot_regression_resolution(model, test_loader, dev, **kwargs):
    model.eval()
    results = []  # resolution results
    pid_classification_results = []
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for batch_g, y in tq:
                batch_g = batch_g.to(dev)
                if args.loss_regularization:
                    model_output, loss_regularizing_neig = model(batch_g)
                else:
                    model_output = model(batch_g)
                resolutions, pid_true, pid_pred = model.mod.object_condensation_loss2(
                    batch_g,
                    model_output,
                    y,
                    return_resolution=True,
                    q_min=args.qmin,
                    frac_clustering_loss=0,
                    use_average_cc_pos=args.use_average_cc_pos,
                    hgcalloss=args.hgcalloss,
                )
                results.append(resolutions)
                pid_classification_results.append((pid_true, pid_pred))
    result_dict = {}
    for key in results[0]:
        result_dict[key] = np.concatenate([r[key] for r in results])
    result_dict["event_by_event_accuracy"] = [
        accuracy_score(pid_true.argmax(dim=0), pid_pred.argmax(dim=0))
        for pid_true, pid_pred in pid_classification_results
    ]
    # just plot all for now
    result = {}
    for key in results[0]:
        data = result_dict[key]
        fig, ax = plt.subplots()
        ax.hist(data, bins=100, range=(-1.5, 1.5), histtype="step", label=key)
        ax.set_xlabel("resolution")
        ax.set_ylabel("count")
        ax.legend()
        result[key] = fig
    conf_mat = confusion_matrix(pid_true.argmax(dim=0), pid_pred.argmax(dim=0))
    # confusion matrix
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # add onehot_particle_arr as class names
    class_names = onehot_particles_arr
    im = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(len(class_names)), class_names, rotation=45)
    ax.set_yticks(np.arange(len(class_names)), class_names)
    result["PID_confusion_matrix"] = fig

    return result
