import torch



class LLFillSpace(torch.nn.Module):
    def __init__(self,
                 maxhits: int = 1000,
                 runevery: int = -1):
        #print('INFO: LLFillSpace: this is actually a regulariser: move to right file soon.')
        assert maxhits > 0
        self.maxhits = maxhits
        self.runevery = runevery
        self.counter = -1
        if runevery < 0:
            self.counter = -2
        super(LLFillSpace, self).__init__()

    def get_config(self):
        config = {'maxhits': self.maxhits,
                  'runevery': self.runevery}
        base_config = super(LLFillSpace, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _rs_loop(self, coords):
        # only select a few hits to keep memory managable
        nhits = coords.shape[0]
        maxhits = self.maxhits
        sel = None
        if nhits > maxhits:
            sel = torch.randint(low=0, high=coords.shape[0] - 1, size=(maxhits,), dtype=torch.int32)
        else:
            sel = torch.arange(coords.shape[0], dtype=torch.int32)
        sel = sel.to(coords.device)
        sel = torch.unsqueeze(sel, dim=1).flatten()
        coords_selected = torch.index_select(coords, 0, sel).clone()  # V' x C
        # print('coords',coords.shape)
        means = torch.mean(coords_selected, axis=0)  # 1 x C
        coords_selected = coords_selected - means  # V' x C
        # build covariance
        cov = torch.unsqueeze(coords_selected, dim=1) * torch.unsqueeze(coords_selected, dim=2)
        cov = torch.mean(cov, dim=0, keepdim=False)  # 1 x C x C
        # print('cov',cov)
        # get eigenvals
        eigenvals, _ = torch.linalg.eig(cov)  # cheap because just once, no need for approx
        eigenvals = eigenvals.to(torch.float32)
        # penalise one small EV (e.g. when building a surface)
        pen = torch.log((torch.mean(eigenvals) / (torch.min(eigenvals) + 1e-6) - 1.) ** 2 + 1.)
        return pen

    def _raw_loss(self, coords, batch_idx):
        loss = torch.tensor(0).float().to(coords.device)
        for i in batch_idx.unique():
            idx = batch_idx == i
            loss += self._rs_loop(coords[idx, :])
        return loss

    def forward(self, clust_space, batch_idx):
        if self.counter >= 0:  # completely optimise away increment
            if self.counter < self.runevery:
                self.counter += 1
                return torch.tensor(0).to(clust_space.device)
            self.counter = 0
        lossval = self._raw_loss(clust_space, batch_idx)

        if self.counter == -1:
            self.counter += 1
        return lossval


