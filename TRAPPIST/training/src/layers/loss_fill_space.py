# copy of the original from https://github.com/cms-pepr/HGCalML/blob/master/modules/LossLayers.py#L916-L943

class LLFillSpace(LossLayerBase):
    def __init__(self,
                 maxhits: int = 1000,
                 runevery: int = -1,
                 **kwargs):
        '''
        calculated a PCA of all points in coordinate space and
        penalises very asymmetric PCs.
        Reduces the risk of falling back to a (hyper)surface

        Inputs:
         - coordinates, row splits, (truth index - optional. then only applied to non-noise)
        Outputs:
         - coordinates (unchanged)
        '''
        #print('INFO: LLFillSpace: this is actually a regulariser: move to right file soon.')
        assert maxhits > 0
        self.maxhits = maxhits
        self.runevery = runevery
        self.counter = -1
        if runevery < 0:
            self.counter = -2
        if 'dynamic' in kwargs:
            super(LLFillSpace, self).__init__(**kwargs)
        else:
            super(LLFillSpace, self).__init__(dynamic=True, **kwargs)

    def get_config(self):
        config = {'maxhits': self.maxhits,
                  'runevery': self.runevery}
        base_config = super(LLFillSpace, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def _rs_loop(coords, tidx, maxhits=1000):
        # only select a few hits to keep memory managable
        nhits = coords.shape[0]
        sel = None
        if nhits > maxhits:
            sel = tf.random.uniform(shape=(maxhits,), minval=0, maxval=coords.shape[0] - 1, dtype=tf.int32)
        else:
            sel = tf.range(coords.shape[0], dtype=tf.int32)
        sel = tf.expand_dims(sel, axis=1)
        coords = tf.gather_nd(coords, sel)  # V' x C
        if tidx is not None:
            tidx = tf.gather_nd(tidx, sel)  # V' x C
            coords = coords[tidx[:, 0] >= 0]
        # print('coords',coords.shape)
        means = tf.reduce_mean(coords, axis=0, keepdims=True)  # 1 x C
        coords -= means  # V' x C
        # build covariance
        cov = tf.expand_dims(coords, axis=1) * tf.expand_dims(coords, axis=2)  # V' x C x C
        cov = tf.reduce_mean(cov, axis=0, keepdims=False)  # 1 x C x C
        # print('cov',cov)
        # get eigenvals
        eigenvals, _ = tf.linalg.eig(cov)  # cheap because just once, no need for approx
        eigenvals = tf.cast(eigenvals, dtype='float32')
        # penalise one small EV (e.g. when building a surface)
        pen = tf.math.log((tf.math.divide_no_nan(tf.reduce_mean(eigenvals),
                                                 tf.reduce_min(eigenvals) + 1e-6) - 1.) ** 2 + 1.)
        return pen

    @staticmethod
    def raw_loss(coords, rs, tidx, maxhits=1000):
        loss = tf.zeros([], dtype='float32')
        for i in range(len(rs) - 1):
            rscoords = coords[rs[i]:rs[i + 1]]
            loss += LLFillSpace._rs_loop(rscoords, tidx, maxhits)
        return tf.math.divide_no_nan(loss, tf.cast(rs.shape[0], dtype='float32'))

    def loss(self, inputs):
        assert len(inputs) == 2 or len(inputs) == 3  # coords, rs
        tidx = None
        if len(inputs) == 3:
            coords, rs, tidx = inputs
        else:
            coords, rs = inputs
        if self.counter >= 0:  # completely optimise away increment
            if self.counter < self.runevery:
                self.counter += 1
                return tf.zeros_like(coords[0, 0])
            self.counter = 0
        lossval = LLFillSpace.raw_loss(coords, rs, tidx, self.maxhits)

        if self.counter == -1:
            self.counter += 1
        return lossval


