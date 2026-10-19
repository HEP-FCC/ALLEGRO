
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from wpca import PCA, WPCA, EMPCA  # The sklearn PCA doesn't support weights so we're using Weighted PCA here
import torch
from xformers.ops.fmha import BlockDiagonalMask
import numpy as np
from src.models.thrust_axis import Thrust, hits_xyz_to_momenta, LR, weighted_least_squares_line
from torch_scatter import scatter_mean, scatter_sum

def pick_lowest_chi_squared(pxpypz, chi_s, batch_idx, xyz_nodes):
    unique_batch = torch.unique(batch_idx)
    p_direction = []
    track_xyz = []
    for i in range(0, len(unique_batch)):
        mask = batch_idx == unique_batch[i]
        if torch.sum(mask) > 1:
            chis = chi_s[mask]
            ind_min = torch.argmin(chis)
            p_direction.append(pxpypz[mask][ind_min].view(-1, 3))
            track_xyz.append(xyz_nodes[mask][ind_min].view(-1, 3))

        else:
            p_direction.append(pxpypz[mask].view(-1, 3))
            track_xyz.append(xyz_nodes[mask].view(-1, 3))
    return torch.concat(p_direction, dim=0), torch.stack(track_xyz)[:, 0]



class ThrustAxis(torch.nn.Module):
    #  Same layout of the module as the GNN one, but just computes the direction of the shower by finding the Thrust Axis.
    def __init__(self):
        super(ThrustAxis, self).__init__()
    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        assert graphs_new is not None
        batch_num_nodes = graphs_new.batch_num_nodes()  # Num. of hits in each graph
        batch_idx = []
        batch_bounds = []
        p_directions = []
        barycenters = []
        p_dir_avg = []  # For debugging
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
            batch_bounds.append(n)
        for i in np.unique(batch_idx):
            hits_xyz = graphs_new.ndata["h"][np.array(batch_idx) == i, :3].detach().cpu().numpy()
            hits_E = graphs_new.ndata["h"][np.array(batch_idx) == i, 7].detach().cpu().numpy()
            momenta = hits_xyz_to_momenta(hits_xyz, hits_E)
            #thrust_axis = Thrust.calculate_thrust(momenta)
            #thrust_axis = LR.calculate_thrust(hits_xyz, hits_E)
            thrust_axis = weighted_least_squares_line(hits_xyz, np.ones_like(hits_E))[1]
            thrust_axis /= np.linalg.norm(thrust_axis)
            barycenter = np.average(hits_xyz, weights=hits_E, axis=0)
            dot_prod = np.dot(thrust_axis, barycenter)
            if dot_prod < 0:
                thrust_axis = -thrust_axis
            p_directions.append(torch.tensor(thrust_axis))
            barycenters.append(torch.tensor(barycenter)*3300)
        p_direction = torch.stack(p_directions)
        p_tracks = torch.norm(p_direction, dim=1)
        p_direction = p_direction / torch.norm(p_direction, dim=1).unsqueeze(1)
        barycenters = torch.stack(barycenters)
        return p_tracks, p_direction, barycenters # ref pt



class NeutralPCA(torch.nn.Module):
    # Same layout of the module as the GNN one, but just computes the direction of the shower.
    def __init__(self):
        super(NeutralPCA, self).__init__()

    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        assert graphs_new is not None
        batch_num_nodes = graphs_new.batch_num_nodes()  # Num. of hits in each graph
        batch_idx = []
        batch_bounds = []
        p_directions = []
        barycenters = []
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
            batch_bounds.append(n)
        for i in np.unique(batch_idx):
            #w = WPCA(n_components=1)
            w = PCA(n_components=1) # Try unweighted PCA for debugging
            weights = graphs_new.ndata["h"][np.array(batch_idx) == i, 7].detach().cpu().reshape(-1, 1)
            weights = weights / torch.sum(weights)
            # repeat weights 3 times
            weights = np.repeat(weights, 3, axis=1)
            hits_xyz = graphs_new.ndata["h"][np.array(batch_idx) == i, :3].detach().cpu()
            w.fit(hits_xyz)   #, weights=weights)
            k = torch.tensor(w.components_[0])
            # mask = dist_from_first_pca < 50
            # only keep the 90% closest hits
            mean = torch.tensor(w.mean_)
            #norm1 =  torch.norm(mean + k)
            #norm2 = torch.norm(mean - k)
            #if norm1 < norm2:
            #    k *= -1
            a = hits_xyz - mean
            dist_from_first_pca = np.sqrt(np.linalg.norm(a, axis=1) ** 2 - np.dot(a, k) ** 2)
            mask = dist_from_first_pca < np.quantile(dist_from_first_pca, 0.9)
            if mask.sum() == 0:
                #mask = dist_from_first_pca < np.quantile(dist_from_first_pca, 0.95)
                mask = np.ones_like(mask)
            hits_filtered = hits_xyz[mask]
            hits_E_filtered = graphs_new.ndata["h"][np.array(batch_idx) == i, 7][mask].detach().cpu().numpy()
            k = weighted_least_squares_line(hits_filtered, hits_E_filtered)[1]
            k = torch.tensor(k)
            k /= torch.norm(k)
            if np.dot(k, mean) < 0:
                k *= -1
            # Figure out the direction
            p_directions.append(k)
            barycenters.append(mean)
            #print(graphs_new.ndata["h"][batch_idx == i, :3])
            #print(w.components_)
            #print("-------------------")
        p_direction = torch.stack(p_directions)
        #batch_idx = torch.tensor(batch_idx).to(graphs_new.device)
        #xyz_hits = graphs_new.ndata["h"][:, :3]
        #E_hits = graphs_new.ndata["h"][:, 7]
        #weighted_avg_hits = scatter_sum(xyz_hits * E_hits.unsqueeze(1), batch_idx, dim=0)
        # get the principal axis
        #E_total = scatter_sum(E_hits, batch_idx, dim=0)
        #p_direction = weighted_avg_hits / E_total.unsqueeze(1)
        p_tracks = torch.norm(p_direction, dim=1)
        p_direction = p_direction  / torch.norm(p_direction, dim=1).unsqueeze(1)
        # if self.pos_regression:
        return p_tracks, p_direction, torch.stack(barycenters)*3300# reference point
        # return p_tracks


class AverageHitsP(torch.nn.Module):
    # Same layout of the module as the GNN one, but just computes the average of the hits. Try to compare this + ML clustering with Pandora
    def __init__(self, ecal_only=False):
        super(AverageHitsP, self).__init__()
        self.ecal_only = ecal_only
    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        assert graphs_new is not None
        batch_num_nodes = graphs_new.batch_num_nodes()  # Num. of hits in each graph
        batch_idx = []
        batch_bounds = []
        if self.ecal_only:
            mask_ecal_only = [] # whether to consider only ECAL or ECAL+HCAL
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
            batch_bounds.append(n)
        batch_idx = np.array(batch_idx)
        for i in range(len(np.unique(batch_idx))):
            if self.ecal_only:
                n_ecal_hits = (graphs_new.ndata["h"][batch_idx == i, 5] > 0).sum()
                n_hcal_hits = (graphs_new.ndata["h"][batch_idx == i, 6] > 0).sum()
                if self.ecal_only:
                    for _ in range(batch_num_nodes[i]):
                        mask_ecal_only.append((n_ecal_hits / (n_hcal_hits + n_ecal_hits)).item())
        batch_idx = torch.tensor(batch_idx).to(graphs_new.device)
        if self.ecal_only:
            mask_ecal_only = torch.tensor(mask_ecal_only)  # round().int().bool().to(graphs_new.device)
            mask_ecal_only = (mask_ecal_only > 0.05).int().bool().to(graphs_new.device)
            #mask_ecal_only=torch.zeros(len(mask_ecal_only)).bool().to(graphs_new.device)
        xyz_hits = graphs_new.ndata["h"][:, :3]
        E_hits = graphs_new.ndata["h"][:, 8]
        if self.ecal_only:
            hcal_hits = graphs_new.ndata["h"][:, 6] > 0
            E_hits[mask_ecal_only & (hcal_hits)] = 0
        weighted_avg_hits = scatter_sum(xyz_hits * E_hits.unsqueeze(1), batch_idx, dim=0)
        E_total = scatter_sum(E_hits, batch_idx, dim=0)
        p_direction = weighted_avg_hits / E_total.unsqueeze(1)
        p_tracks = torch.norm(p_direction, dim=1)
        p_direction = p_direction / torch.norm(p_direction, dim=1).unsqueeze(1)
        # if self.pos_regression:
        return p_tracks, p_direction,  weighted_avg_hits / E_total.unsqueeze(1) * 3300 # Reference point
        # return p_tracks



class PickPAtDCA(torch.nn.Module):
    # Same layout of the module as the GNN one, but just picks the track
    def __init__(self):
        super(PickPAtDCA, self).__init__()

    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        assert graphs_new is not None
        batch_num_nodes = graphs_new.batch_num_nodes()  # Num. hits in each graph
        batch_idx = []
        batch_bounds = []
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
            batch_bounds.append(n)
        batch_idx = torch.tensor(batch_idx).to(graphs_new.device)
        # ht = graphs_new.ndata["hit_type"]
        ht = graphs_new.ndata["h"][:, 3:7].argmax(dim=1)
        filt = ht == 1  # track
        filt_hits = ((ht == 2) + (ht == 3)).bool()
        # if "pos_pxpypz_at_vertex" in graphs_new.ndata.keys():
        #    key = "pos_pxpypz_at_vertex"
        # else:
        #    key = "pos_pxpypz"
        # p_direction = scatter_mean(
        #     graphs_new.ndata["pos_pxpypz_at_vertex"][filt], batch_idx[filt], dim=0
        # )
        # take the min chi squared track if there are multiple
        p_direction, p_xyz = pick_lowest_chi_squared(
            graphs_new.ndata["pos_pxpypz_at_vertex"][filt],
            graphs_new.ndata["chi_squared_tracks"][filt],
            batch_idx[filt],
            graphs_new.ndata["h"][filt, :3]
        )
        # Barycenters of clusters of hits
        xyz_hits = graphs_new.ndata["h"][:, :3]
        E_hits = graphs_new.ndata["h"][:, 8]
        weighted_avg_hits = scatter_sum(xyz_hits * E_hits.unsqueeze(1), batch_idx, dim=0)
        E_total = scatter_sum(E_hits, batch_idx, dim=0)
        barycenters = weighted_avg_hits / E_total.unsqueeze(1)
        p_tracks = torch.norm(p_direction, dim=1)
        p_direction = p_direction  # / torch.norm(p_direction, dim=1).unsqueeze(1)
        return p_tracks, p_direction, barycenters - p_xyz   # torch.concat([barycenters, p_xyz], dim =1) # Reference point



class ECNetWrapperAvg(torch.nn.Module):
    # use the GNN+NN model for energy correction
    # This one concatenates GNN features to the global features
    def __init__(self):
        super(ECNetWrapperAvg, self).__init__()
        self.AvgHits = AverageHitsP(ecal_only=True)

    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        _, p_pred, _ = self.AvgHits.predict(x_global_features, graphs_new)
        p_pred = (p_pred / torch.norm(p_pred, dim=1).unsqueeze(1)).clone()
        return None, p_pred, None, None
    @staticmethod
    def obtain_batch_numbers(g):
        graphs_eval = dgl.unbatch(g)
        number_graphs = len(graphs_eval)
        batch_numbers = []
        for index in range(0, number_graphs):
            gj = graphs_eval[index]
            num_nodes = gj.number_of_nodes()
            batch_numbers.append(index * torch.ones(num_nodes))
            num_nodes = gj.number_of_nodes()
        batch = torch.cat(batch_numbers, dim=0)
        return batch

    def build_attention_mask(self, g):
        batch_numbers = self.obtain_batch_numbers(g)
        return BlockDiagonalMask.from_seqlens(
            torch.bincount(batch_numbers.long()).tolist()
        )
