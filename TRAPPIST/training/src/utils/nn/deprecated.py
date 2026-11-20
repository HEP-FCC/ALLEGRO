if alternate_steps is not None:
    if not hasattr(model.mod, "current_state_alternate_steps"):
        model.mod.current_state_alternate_steps = 0
if alternate_steps is not None and step_count % alternate_steps == 0:
    print("Flipping steps")
    state = model.mod.current_state_alternate_steps
    state = 1 - state
    model.mod.current_state_alternate_steps = state
    wandb.log(
        {"current_state_alternate_steps": model.mod.current_state_alternate_steps}
    )
    if state == 0:
        print("Switched to beta loss")
        model.mod.beta_weight = (
            1.0  # set this to zero for no beta loss (when it's frozen)
        )
        model.mod.beta_exp_weight = 1.0
        model.mod.attr_rep_weight = 0.0
    else:
        print("Switched to clustering loss")
        model.mod.beta_weight = (
            0.0  # set this to zero for no beta loss (when it's frozen)
        )
        model.mod.beta_exp_weight = 0.0
        model.mod.attr_rep_weight = 1.0


# if clust_loss_only and calc_e_frac_loss and logwandb and local_rank == 0:
#     wandb.log(
#         {
#             "loss e frac": loss_E_frac,
#             "loss e frac true": loss_E_frac_true,
#         }
#     )

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

# fig, ax = plt.subplots()
# repulsive, attractive = (
#     lst_nonzero(losses[16].detach().cpu().flatten()),
#     lst_nonzero(losses[17].detach().cpu().flatten()),
# )
# ax.hist(
#     repulsive.view(-1),
#     bins=100,
#     alpha=0.5,
#     label="repulsive",
#     color="r",
# )
# ax.hist(
#     attractive.view(-1),
#     bins=100,
#     alpha=0.5,
#     label="attractive",
#     color="b",
# )
# ax.set_yscale("log")
# ax.legend()
# wandb.log({"rep. and att. norms": wandb.Image(fig)})
# plt.close(fig)


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


def inference_statistics(
    model,
    train_loader,
    dev,
    grad_scaler=None,
    loss_terms=[],
    args=None,
    radius=0.7,
    total_num_batches=10,
    save_ckpt_to_folder=None,
):
    model.eval()
    clust_loss_only = loss_terms[0]
    add_energy_loss = loss_terms[1]
    num_batches = 0
    loss_E_fracs = []
    loss_E_fracs_true = []
    loss_E_fracs_true_nopart = []
    loss_E_fracs_nopart = []
    part_E_true = []
    part_PID_true = []
    betas_list = []
    figs = []
    reco_counts, non_reco_counts = {}, {}
    total_counts = {}
    with tqdm.tqdm(train_loader) as tq:
        for batch_g, y in tq:
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                batch_g = batch_g.to(dev)
                if args.loss_regularization:
                    model_output, loss_regularizing_neig, loss_ll = model(batch_g)
                else:
                    model_output = model(batch_g, 1)
                preds = model_output.squeeze()
                (
                    loss,
                    losses,
                    loss_E_frac,
                    loss_E_frac_true,
                    loss_E_frac_nopart,
                    loss_E_frac_true_nopart,
                ) = object_condensation_loss2(
                    batch_g,
                    model_output,
                    y,
                    clust_loss_only=clust_loss_only,
                    add_energy_loss=add_energy_loss,
                    calc_e_frac_loss=True,
                    e_frac_loss_return_particles=True,
                    q_min=args.qmin,
                    frac_clustering_loss=args.frac_cluster_loss,
                    attr_weight=args.L_attractive_weight,
                    repul_weight=args.L_repulsive_weight,
                    fill_loss_weight=args.fill_loss_weight,
                    use_average_cc_pos=args.use_average_cc_pos,
                    hgcalloss=args.hgcalloss,
                    e_frac_loss_radius=radius,
                )
                (
                    loss_E_frac_true,
                    particle_ids_all,
                    reco_count,
                    non_reco_count,
                    total_count,
                ) = loss_E_frac_true
                (
                    loss_E_frac_true_nopart,
                    particle_ids_all_nopart,
                    reco_count_nopart,
                    non_reco_count_nopart,
                    total_count_nopart,
                ) = loss_E_frac_true_nopart
                update_dict(reco_counts, reco_count_nopart)
                update_dict(total_counts, total_count_nopart)
                if len(reco_count):
                    assert len(reco_counts) >= len(reco_count_nopart)
                update_dict(non_reco_counts, non_reco_count_nopart)
                loss_E_fracs.append([x.cpu() for x in loss_E_frac])
                loss_E_fracs_true.append([x.cpu() for x in loss_E_frac_true])
                loss_E_fracs_true_nopart.append(
                    [x.cpu() for x in loss_E_frac_true_nopart]
                )
                loss_E_fracs_nopart.append([x.cpu() for x in loss_E_frac_nopart])
                part_PID_true.append(
                    [
                        y[torch.tensor(pidall) - 1, 6].long()
                        for pidall in particle_ids_all
                    ]
                )
                part_E_true.append(
                    [y[torch.tensor(pidall) - 1, 3] for pidall in particle_ids_all]
                )
                if clust_loss_only:
                    clust_space_dim = 3
                else:
                    clust_space_dim = model.mod.output_dim - 28
                xj = model_output[:, 0:clust_space_dim]
                # if model.mod.clust_space_norm == "twonorm":
                #     xj = torch.nn.functional.normalize(xj, dim=1)
                # elif model.mod.clust_space_norm == "tanh":
                #     xj = torch.tanh(xj)
                # elif model.mod.clust_space_norm == "none":
                #     pass
                bj = torch.sigmoid(
                    torch.reshape(model_output[:, clust_space_dim], [-1, 1])
                )  # 3: betas
                bj = bj.clip(0.0, 1 - 1e-4)
                q = bj.arctanh() ** 2 + args.qmin
                fig, ax = plot_clust(
                    batch_g,
                    q,
                    xj,
                    y=y,
                    radius=radius,
                    loss_e_frac=loss_E_fracs[-1],
                    betas=bj,
                )
                betas = (
                    torch.sigmoid(
                        torch.reshape(preds[:, args.clustering_space_dim], [-1, 1])
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                # figs.append(fig)
                betas_list.append(betas)
            num_batches += 1
            if num_batches % 5 == 0 and save_ckpt_to_folder is not None:
                Path(save_ckpt_to_folder).mkdir(parents=True, exist_ok=True)
                loss_E_fracs_fold = [
                    item for sublist in loss_E_fracs for item in sublist
                ]
                loss_E_fracs_fold = torch.concat(loss_E_fracs_fold).flatten()
                loss_E_fracs_true_fold = [
                    item for sublist in loss_E_fracs_true for item in sublist
                ]
                loss_E_fracs_true_fold = torch.concat(loss_E_fracs_true_fold).flatten()
                part_E_true_fold = [item for sublist in part_E_true for item in sublist]
                part_E_true_fold = torch.concat(part_E_true_fold).flatten()
                part_PID_true_fold = [
                    item for sublist in part_PID_true for item in sublist
                ]
                part_PID_true_fold = torch.concat(part_PID_true_fold).flatten()
                loss_E_fracs_nopart_fold = [
                    item for sublist in loss_E_fracs_nopart for item in sublist
                ]
                loss_E_fracs_true_nopart_fold = [
                    item for sublist in loss_E_fracs_true_nopart for item in sublist
                ]
                obj = {
                    "loss_e_fracs_nopart": loss_E_fracs_nopart_fold,
                    "loss_e_fracs_true_nopart": loss_E_fracs_true_nopart_fold,
                    "loss_e_fracs": loss_E_fracs_fold,
                    "loss_e_fracs_true": loss_E_fracs_true_fold,
                    "part_E_true": part_E_true_fold,
                    "part_PID_true": part_PID_true_fold,
                    "reco_counts": reco_counts,
                    "non_reco_counts": non_reco_counts,
                    "total_counts": total_counts,
                }
                file_to_save = os.path.join(save_ckpt_to_folder, "temp_ckpt" + ".pkl")
                with open(file_to_save, "wb") as f:
                    pickle.dump(obj, f)
            if num_batches >= total_num_batches:
                break
            # flatten the lists
        if save_ckpt_to_folder is not None:
            return
        loss_E_fracs = [item for sublist in loss_E_fracs for item in sublist]
        loss_E_fracs = torch.concat(loss_E_fracs).flatten()
        loss_E_fracs_true = [item for sublist in loss_E_fracs_true for item in sublist]
        loss_E_fracs_true = torch.concat(loss_E_fracs_true).flatten()
        part_E_true = [item for sublist in part_E_true for item in sublist]
        part_E_true = torch.concat(part_E_true).flatten()
        part_PID_true = [item for sublist in part_PID_true for item in sublist]
        part_PID_true = torch.concat(part_PID_true).flatten()
        loss_E_fracs_nopart = [
            item for sublist in loss_E_fracs_nopart for item in sublist
        ]
        loss_E_fracs_true_nopart = [
            item for sublist in loss_E_fracs_true_nopart for item in sublist
        ]

    return {
        "loss_e_fracs": loss_E_fracs,
        "loss_e_fracs_true": loss_E_fracs_true,
        "loss_e_fracs_nopart": loss_E_fracs_nopart,
        "loss_e_fracs_true_nopart": loss_E_fracs_true_nopart,
        "betas": betas_list,
        "part_E_true": part_E_true,
        "part_PID_true": part_PID_true,
        "reco_counts": reco_counts,
        "non_reco_counts": non_reco_counts,
        "total_counts": total_counts,
    }


def inference(model, test_loader, dev):
    """
    Similar to evaluate_regression, but without the ground truth labels.
    """
    model.eval()
    num_batches = 0
    count = 0
    results = []
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for batch_g, _ in tq:
                batch_g = batch_g.to(dev)
                model_output = model(batch_g)
                # preds = model_output.squeeze().float()
                preds = model.mod.object_condensation_inference(batch_g, model_output)
                num_batches += 1
                results.append(preds)
    time_diff = time.time() - start_time
    _logger.info(
        "Processed %d entries in total (avg. speed %.1f entries/s)"
        % (count, count / time_diff)
    )
    return results

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

    # losses_cpu = [
    #     x.detach().to("cpu") if isinstance(x, torch.Tensor) else x
    #     for x in losses
    # ]
    # all_val_losses.append(losses_cpu)
    # all_val_loss.append(loss.detach().to("cpu").item())

    # pid_true, pid_pred = torch.cat(
    #     [torch.tensor(x[7]) for x in all_val_losses]
    # ), torch.cat([torch.tensor(x[8]) for x in all_val_losses])
    # pid_true, pid_pred = pid_true.tolist(), pid_pred.tolist()


# , step=step)
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

# scores = np.concatenate(scores)
# labels = {k: _concat(v) for k, v in labels.items()}
# metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
# _logger.info('Evaluation metrics: \n%s', '\n'.join(
#    ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))


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


# if args.loss_regularization:
#     wandb.log({"loss regul neigh": loss_regularizing_neig})
#     wandb.log({"loss ll": loss_ll})

# if (num_batches - 1) % 100 == 0:
#     if clust_loss_only:
#         clust_space_dim = 3  # model.mod.output_dim - 1
#     else:
#         clust_space_dim = model.mod.output_dim - 28
#     bj = torch.sigmoid(
#         torch.reshape(model_output[:, clust_space_dim], [-1, 1])
#     )  # 3: betas
#     xj = model_output[:, 0:clust_space_dim]  # xj: cluster space coords
#     # assert len(bj) == len(xj)
#     # if model.mod.clust_space_norm == "twonorm":
#     #     xj = torch.nn.functional.normalize(
#     #         xj, dim=1
#     #     )  # 0, 1, 2: cluster space coords
#     # elif model.mod.clust_space_norm == "tanh":
#     #     xj = torch.tanh(xj)
#     # elif model.mod.clust_space_norm == "none":
#     #     pass

#     bj = bj.clip(0.0, 1 - 1e-4)
#     q = bj.arctanh() ** 2 + args.qmin
#     assert q.shape[0] == xj.shape[0]
#     assert batch_g.ndata["h"].shape[0] == xj.shape[0]
#     fig, ax = plot_clust(
#         batch_g,
#         q,
#         xj,
#         title_prefix="train ep. {}, batch {}".format(
#             epoch, num_batches
#         ),
#         y=y,
#         betas=bj,
#     )
#     wandb.log({"clust": wandb.Image(fig)})
#     fig.clf()
#     # if (num_batches - 1) % 500 == 0:
#     #     wandb.log(
#     #         {
#     #             "conf_mat_train": wandb.plot.confusion_matrix(
#     #                 y_true=pid_true,
#     #                 preds=pid_pred,
#     #                 class_names=class_names,
#     #             )
#     #         }
#     #     )
# ks = sorted(list(losses[9].keys()))
# losses_cpu = [
# x.detach().to("cpu") if isinstance(x, torch.Tensor) else x
# for x in losses
# ]
# tables = {}
# for key in ks:
# tables[key] = losses[9][
# key
# ]  # wandb.Table(data=[[x] for x in losses[9][key]], columns=[key])
# if local_rank == 0:
# wandb.log(
# {
#     key: wandb.Histogram(clip_list(tables[key]), num_bins=100)
#     for key, val in losses_cpu[9].items()
# }
# )  # , step=step_count)
# return loss_epoch_total, losses_epoch_total
