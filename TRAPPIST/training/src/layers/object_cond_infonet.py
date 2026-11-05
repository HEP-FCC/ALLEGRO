from typing import Tuple, Union
import numpy as np
import torch
from torch_scatter import scatter_max, scatter_add, scatter_mean
from src.layers.loss_fill_space_torch import LLFillSpace
import dgl


def infonet_updated(g, qmin, xj, bj):
    list_graphs = dgl.unbatch(g)
    loss_total = 0
    Loss_beta_zero = 0
    number_particles_accounted_for = 0
    node_counter = 0
    Loss_beta = 0
    for i in range(0, len(list_graphs)):
        graph_eval = list_graphs[i]
        non = graph_eval.number_of_nodes()
        xj_graph = xj[node_counter : non + node_counter]
        bj_graph = bj[node_counter : non + node_counter]
        q_graph = bj_graph.arctanh() ** 2 + qmin
        q = q_graph.detach().cpu().numpy()
        part_num = graph_eval.ndata["particle_number"].view(-1).to(torch.long)
        q_alpha, index_alpha = scatter_max(q_graph.view(-1), part_num - 1)
        x_alpha = xj_graph[index_alpha]
        number_of_particles = torch.unique(graph_eval.ndata["particle_number"])
        indx = torch.zeros((len(number_of_particles), 50)).to(x_alpha.device)
        b_alpha = bj_graph[index_alpha]
        # beta_zero_loss = torch.sum(torch.exp(10 * bj_graph)) / non
        beta_zero_loss = torch.sum(bj_graph) / non
        if len(number_of_particles) > 1:
            for nn in range(0, len(number_of_particles)):
                idx_part = number_of_particles[nn]
                positives_of_class = graph_eval.ndata["particle_number"] == idx_part
                pos_indx = torch.where(positives_of_class == True)[0]
                if len(pos_indx) > 50:
                    indx[nn, :] = pos_indx[0:50]
                else:
                    indx[nn, 0 : len(pos_indx)] = pos_indx
                    indx[nn, len(pos_indx) :] = pos_indx[0]
            for nn in range(0, len(number_of_particles)):
                idx_part = number_of_particles[nn]
                positives_of_class = graph_eval.ndata["particle_number"] == idx_part
                xj_ = xj_graph[positives_of_class]
                x_alpha_ = x_alpha[nn]
                dot_products = torch.mul(
                    xj_, x_alpha_.unsqueeze(0).tile((xj_.shape[0], 1))
                ).sum(dim=1)
                dot_products_exp = torch.exp(dot_products)
                indx_copy = indx.clone()
                indx_copy[nn] = -1
                indx_copy = indx_copy.view(-1)
                indx_copy = indx_copy[indx_copy > -1]
                neg_indx = indx_copy
                xj_neg = xj_graph[neg_indx.long()]
                dot_neg = torch.sum(
                    torch.exp(torch.tensordot(xj_, xj_neg, dims=([1], [1]))), dim=1
                )
                loss = torch.mean(-torch.log(dot_products_exp / dot_neg))
                if torch.sum(torch.isnan(loss)) > 0:
                    print(dot_products)
                    print(dot_neg)
                    print(dot_products / dot_neg)
                    print(torch.log(dot_products / dot_neg))
                loss_total = loss_total + loss
                number_particles_accounted_for = number_particles_accounted_for + 1
        Loss_beta = Loss_beta + torch.sum((1 - b_alpha)) / len(b_alpha)
        Loss_beta_zero = Loss_beta_zero + beta_zero_loss

    loss_total = loss_total / number_particles_accounted_for
    Loss_beta = Loss_beta / len(list_graphs)
    Loss_beta_zero = Loss_beta_zero / len(list_graphs)
    loss_total_ = loss_total + Loss_beta + Loss_beta_zero

    return loss_total_, Loss_beta, Loss_beta_zero, loss_total
