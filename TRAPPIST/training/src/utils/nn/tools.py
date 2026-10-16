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


def log_losses_wandb(
    logwandb, num_batches, local_rank, losses, loss, loss_ll, loss_ec=0.0, val=False
):
    if val:
        val_ = " val"
    else:
        val_ = ""
    if logwandb and ((num_batches - 1) % 10) == 0 and local_rank == 0:
        wandb.log(
            {
                "loss" + val_ + " regression": loss,
                "loss" + val_ + " lv": losses[0],
                "loss" + val_ + " beta": losses[1],
                "loss" + val_ + " beta sig": losses[2],
                "loss" + val_ + " beta noise": losses[3],
                # "loss" + val_ + " PID": losses[4],
                # "loss" + val_ + " momentum": losses[5],
                # "loss" + val_ + " mass (not us. for opt.)": losses[6],
                # "inter-clustering loss" + val_ + "": losses[10],
                # "filling loss" + val_ + "": losses[11],
                "loss" + val_ + " attractive": losses[12],
                "loss" + val_ + " repulsive": losses[13],
                "loss" + val_ + " repulsive 2": losses[18],
                #"loss" + val_ + " alpha coord": losses[14],
                #"loss" + val_ + " beta zeros": losses[15],

                "loss regularization gravnet" + val_: loss_ll,
                "loss energy correction": loss_ec,
            }
        )


def log_losses_wandb_tracking(
    logwandb, num_batches, local_rank, losses, loss, val=False
):
    if val:
        val_ = " val"
    else:
        val_ = ""
    if logwandb and ((num_batches - 1) % 10) == 0 and local_rank == 0:
        wandb.log(
            {
                "loss" + val_ + " regression": loss,
                "loss" + val_ + " lv": losses[0],
                "loss" + val_ + " beta": losses[1],
                "loss" + val_ + " beta sig": losses[4],
                "loss" + val_ + " beta noise": losses[5],
                "loss" + val_ + " attractive": losses[2],
                "loss" + val_ + " repulsive": losses[3],
            }
        )


def update_and_log_scheduler(scheduler, args, loss, logwandb, local_rank, opt):
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


def lst_nonzero(x):
    return x[x != 0.0]


def clip_list(l, clip_val=4.0):
    result = []
    for item in l:
        if abs(item) > clip_val:
            if item > 0:
                result.append(clip_val)
            else:
                result.append(-clip_val)
        elif np.isnan(item):
            result.append(0.0)  # i don't know why the hell we need this
        else:
            result.append(item)
    return result


def turn_grads_off(model):
    for name, param in model.named_parameters():
        if name == "module.mod.pred_energy.0.weight":
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def log_step_time(
    logwandb, num_batches, local_rank, load_end_time, prev_time, step_end_time
):
    if logwandb and (num_batches % 100) == 0 and local_rank == 0:
        wandb.log(
            {
                "load_time": load_end_time - prev_time,
                "step_time": step_end_time - load_end_time,
            }
        )  # , step=step_count)


def log_betas_hist(logwandb, local_rank, num_batches, betas, args):
    if logwandb and local_rank == 0 and (num_batches % 100) == 0:
        wandb.log(
            {
                "betas": wandb.Histogram(torch.nan_to_num(torch.tensor(betas), 0.0)),
                "qs": wandb.Histogram(
                    np.arctanh(betas.clip(0.0, 1 - 1e-4) / 1.002) ** 2 + args.qmin
                ),
            }
        )  # , step=step_count)


def _flatten_label(label, mask=None):
    if label.ndim > 1:
        label = label.view(-1)
        if mask is not None:
            label = label[mask.view(-1)]
    # print('label', label.shape, label)
    return label


def _flatten_preds(preds, mask=None, label_axis=1):
    if preds.ndim > 2:
        # assuming axis=1 corresponds to the classes
        preds = preds.transpose(label_axis, -1).contiguous()
        preds = preds.view((-1, preds.shape[-1]))
        if mask is not None:
            preds = preds[mask.view(-1)]
    # print('preds', preds.shape, preds)
    return preds


def _check_scales_centers(iterator):
    regress_items = ["part_theta", "part_phi"]
    centers = np.zeros(2)
    scales = np.zeros(2)
    for ii, item in enumerate(regress_items):
        centers[ii] = iterator._data_config.preprocess_params[item]["center"]
        scales[ii] = iterator._data_config.preprocess_params[item]["scale"]
    return centers, scales


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
):
    model.train()

    data_config = train_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_abs_err = 0
    sum_sqr_err = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for batch_g, y in tq:
            label = y
            num_examples = label.shape[0]
            label = label.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                batch_g = batch_g.to(dev)
                model_output = model(batch_g)
                preds = model_output.squeeze()
                loss = loss_func(preds, label)
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

            num_batches += 1
            count += num_examples
            total_loss += loss
            e = preds - label
            abs_err = e.abs().sum().item()
            sum_abs_err += abs_err
            sqr_err = e.square().sum().item()
            sum_sqr_err += sqr_err

            tq.set_postfix(
                {
                    "lr": "%.2e" % scheduler.get_last_lr()[0]
                    if scheduler
                    else opt.defaults["lr"],
                    "Loss": "%.5f" % loss,
                    "AvgLoss": "%.5f" % (total_loss / num_batches),
                    "MSE": "%.5f" % (sqr_err / num_examples),
                    "AvgMSE": "%.5f" % (sum_sqr_err / count),
                    "MAE": "%.5f" % (abs_err / num_examples),
                    "AvgMAE": "%.5f" % (sum_abs_err / count),
                }
            )

            if tb_helper:
                tb_helper.write_scalars(
                    [
                        ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                        (
                            "MSE/train",
                            sqr_err / num_examples,
                            tb_helper.batch_train_count + num_batches,
                        ),
                        (
                            "MAE/train",
                            abs_err / num_examples,
                            tb_helper.batch_train_count + num_batches,
                        ),
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

            if logwandb and (num_batches % 50):
                import wandb

                wandb.log({"loss regression": loss})

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info(
        "Processed %d entries in total (avg. speed %.1f entries/s)"
        % (count, count / time_diff)
    )
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

    if scheduler and not getattr(scheduler, "_update_per_step", False):
        scheduler.step()


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
):
    model.eval()

    data_config = test_loader.dataset.config

    center, scales = _check_scales_centers(iter(test_loader.dataset))
    total_loss = 0
    num_batches = 0
    sum_sqr_err = 0
    sum_abs_err = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for batch_g, y in tq:
                batch_g = batch_g.to(dev)
                label = y
                num_examples = label.shape[0]
                label = label.to(dev)
                model_output = model(batch_g)
                print(label.shape, model_output.shape)
                preds = model_output.squeeze().float()

                loss = 0 if loss_func is None else loss_func(preds, label).item()

                num_batches += 1
                count += num_examples
                total_loss += loss * num_examples
                e = preds - label
                abs_err = e.abs().sum().item()
                sum_abs_err += abs_err
                sqr_err = e.square().sum().item()
                sum_sqr_err += sqr_err

                tq.set_postfix(
                    {
                        "Loss": "%.5f" % loss,
                        "AvgLoss": "%.5f" % (total_loss / count),
                        "MSE": "%.5f" % (sqr_err / num_examples),
                        "AvgMSE": "%.5f" % (sum_sqr_err / count),
                        "MAE": "%.5f" % (abs_err / num_examples),
                        "AvgMAE": "%.5f" % (sum_abs_err / count),
                    }
                )

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(
                                model_output=model_output,
                                model=model,
                                epoch=epoch,
                                i_batch=num_batches,
                                mode="eval" if for_training else "test",
                            )

                if logwandb and (num_batches % 50):
                    wandb.log({"loss val regression": loss})
                    e_nn0 = torch.sum(torch.abs((preds[:, 0] - label[:, 0]))) / len(
                        preds
                    )  # /scales[0]))
                    e_nn1 = torch.sum(torch.abs((preds[:, 1] - label[:, 1]))) / len(
                        preds
                    )  # /scales[1]))
                    e_nn2 = torch.sum(torch.abs((preds[:, 2] - label[:, 2]))) // len(
                        preds
                    )
                    # e_nn3 = torch.sum(torch.abs((preds[:,3] - label[:,3])/scales[3]))
                    # e_nn4 = (preds[:,4] - label[:,4])/scales[4]))
                    # wandb.log({"part_p error ": e_nn0})
                    wandb.log({"part_x error": e_nn0})
                    wandb.log({"part_y error ": e_nn1})
                    wandb.log({"part_z error ": e_nn2})
                    # wandb.log({"part_m error": e_nn3})
                    # wandb.log({"part_pid error": e_nn4})

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info(
        "Processed %d entries in total (avg. speed %.1f entries/s)"
        % (count, count / time_diff)
    )

    if tb_helper:
        tb_mode = "eval" if for_training else "test"
        tb_helper.write_scalars(
            [
                ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
                ("MSE/%s (epoch)" % tb_mode, sum_sqr_err / count, epoch),
                ("MAE/%s (epoch)" % tb_mode, sum_abs_err / count, epoch),
            ]
        )
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(
                    model_output=model_output,
                    model=model,
                    epoch=epoch,
                    i_batch=-1,
                    mode=tb_mode,
                )

    # scores = np.concatenate(scores)
    # labels = {k: _concat(v) for k, v in labels.items()}
    # metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    # _logger.info('Evaluation metrics: \n%s', '\n'.join(
    #    ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        return total_loss / count
    else:
        # convert 2D labels/scores
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_loss / count, scores, labels, observers
