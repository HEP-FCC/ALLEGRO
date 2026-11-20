import tensorflow as tf
class Basic_OC_per_sample(object):
    def __init__(self, 
                 q_min,
                 s_b,
                 use_mean_x,
                 spect_supp=None, #None means same as noise
                 global_weight=False
                 ):
        
        self.q_min = q_min
        self.s_b = s_b
        self.use_mean_x = use_mean_x
        self.global_weight = global_weight
        if spect_supp is None:
            spect_supp = s_b
        self.spect_supp = spect_supp
        
        self.valid=False #constants not created
        
        
    #helper

    def create_Ms(self, truth_idx):
        self.Msel, self.Mnot, _ = CreateMidx(truth_idx, calc_m_not=True)
    
    def set_input(self, 
                         beta,
                         x,
                         d,
                         pll,
                         truth_idx,
                         object_weight,
                         is_spectator_weight,
                         calc_Ms=True,
                         ):
        self.valid=True
        #used for pll and q
        self.tanhsqbeta = tf.math.atanh(beta/(1.01))**2
        
        self.beta_v = tf.debugging.check_numerics(beta,"OC: beta input")
        self.d_v = tf.debugging.check_numerics(d,"OC: d input")
        self.x_v = tf.debugging.check_numerics(x,"OC: x input")
        self.pll_v = tf.debugging.check_numerics(pll,"OC: pll input")
        self.sw_v = tf.debugging.check_numerics(is_spectator_weight,"OC: is_spectator_weight input")

        object_weight = tf.debugging.check_numerics(object_weight,"OC: object_weight input")
        
        self.isn_v = tf.where(truth_idx<0, tf.zeros_like(truth_idx,dtype='float32')+1., 0.)
        
        #spectators do not participate in the potential losses
        self.q_v = (self.tanhsqbeta + self.q_min)*tf.clip_by_value(1.-is_spectator_weight, 0., 1.)
        
        if calc_Ms:
            self.create_Ms(truth_idx)
        if self.Msel is None:
            self.valid=False
            return
        #if self.Msel.shape[0] < 2:#less than two objects - can be dangerous
        #    self.valid=False
        #    return
        
        self.mask_k_m = SelectWithDefault(self.Msel, tf.zeros_like(beta)+1., 0.) #K x V-obj x 1
        self.beta_k_m = SelectWithDefault(self.Msel, self.beta_v, 0.) #K x V-obj x 1
        self.x_k_m = SelectWithDefault(self.Msel, self.x_v, 0.) #K x V-obj x C
        self.q_k_m = SelectWithDefault(self.Msel, self.q_v, 0.)#K x V-obj x 1
        self.d_k_m = SelectWithDefault(self.Msel, self.d_v, 0.)
        
        self.alpha_k = tf.argmax(self.q_k_m, axis=1)# high beta and not spectator -> large q
        
        self.beta_k = tf.gather_nd(self.beta_k_m, self.alpha_k, batch_dims=1) # K x 1
        self.x_k = self._create_x_alpha_k() #K x C
        self.q_k = tf.gather_nd(self.q_k_m, self.alpha_k, batch_dims=1) # K x 1
        self.d_k = tf.gather_nd(self.d_k_m, self.alpha_k, batch_dims=1) # K x 1
        
        #just a temp
        ow_k_m = SelectWithDefault(self.Msel, object_weight, 0.)
        self.ow_k = tf.gather_nd(ow_k_m, self.alpha_k, batch_dims=1) # K x 1
        
    
    ### the following functions should not modify any of the constants and must only depend on them
    
    #for override through inheriting
    def att_func(self,dsq_k_m):
        return tf.math.log(tf.math.exp(1.)*dsq_k_m/2. + 1.)
    
    def V_att_k(self):
        '''
        '''
        K = tf.reduce_sum(tf.ones_like(self.q_k))
        N_k =  tf.reduce_sum(self.mask_k_m, axis=1)
        dsq_k_m = self.calc_dsq_att() #K x V-obj x 1
        sigma = self.weighted_d_k_m(dsq_k_m) #create gradients for all
        dsq_k_m = tf.math.divide_no_nan(dsq_k_m, sigma + 1e-4)
        V_att = self.att_func(dsq_k_m) * self.q_k_m * self.mask_k_m  #K x V-obj x 1
        V_att = self.q_k * tf.reduce_sum( V_att ,axis=1)  #K x 1
        # if self.global_weight:
        #     N_full = tf.reduce_sum(tf.ones_like(self.beta_v))
        #     V_att = K * tf.math.divide_no_nan(V_att, N_full+1e-3)  #K x 1
        # else:
        V_att = tf.math.divide_no_nan(V_att, N_k+1e-3)  #K x 1
        
        #print(tf.reduce_mean(self.d_v),tf.reduce_max(self.d_v))
        
        return V_att
    
    def rep_func(self,dsq_k_v):
        return tf.math.exp(-dsq_k_v/2.)
    
    def weighted_d_k_m(self, dsq): # dsq K x V x 1
        return tf.expand_dims(self.d_k, axis=1) # K x 1 x 1
    
    def calc_dsq_att(self):
        x_k_e = tf.expand_dims(self.x_k,axis=1)
        dsq_k_m = tf.reduce_sum((self.x_k_m - x_k_e)**2, axis=-1, keepdims=True) #K x V-obj x 1
        return dsq_k_m
    
    def calc_dsq_rep(self):
        dsq = tf.expand_dims(self.x_k, axis=1) - tf.expand_dims(self.x_v, axis=0) #K x V x C
        dsq = tf.reduce_sum(dsq**2, axis=-1, keepdims=True)  #K x V x 1
        return dsq
        
    def V_rep_k(self):
        
        
        K = tf.reduce_sum(tf.ones_like(self.q_k))
        N_notk = tf.reduce_sum(self.Mnot, axis=1)
        #future remark: if this gets too large, one could use a kNN here
        
        dsq = self.calc_dsq_rep()
        
        # nogradbeta = tf.stop_gradient(self.beta_k_m)
        #weight. tf.reduce_sum( tf.exp(-dsq) * d_v_e, , axis=1) / tf.reduce_sum( tf.exp(-dsq) )
        sigma = self.weighted_d_k_m(dsq) #create gradients for all, but prefer k vertex
        
        dsq = tf.math.divide_no_nan(dsq, sigma + 1e-4) #K x V x 1

        V_rep = self.rep_func(dsq) * self.Mnot * tf.expand_dims(self.q_v,axis=0)  #K x V x 1

        V_rep = self.q_k * tf.reduce_sum(V_rep, axis=1) #K x 1
        if self.global_weight:
            N_full = tf.reduce_sum(tf.ones_like(self.beta_v))
            V_rep = K * tf.math.divide_no_nan(V_rep, N_full+1e-3)  #K x 1
        else:
            V_rep = tf.math.divide_no_nan(V_rep, N_notk+1e-3)  #K x 1

        return V_rep


    def Pll_k(self):
        
        tanhsqbeta = self.beta_v**2 #softer here
        tanhsqbeta = tf.debugging.check_numerics(tanhsqbeta, "OC: pw b**2")
        pw = tanhsqbeta * tf.clip_by_value((1.-tf.clip_by_value(self.isn_v+self.sw_v,0.,1.)),0.,1.) + 1e-6
        
        pw = tf.debugging.check_numerics(pw, "OC: pw")
        
        pll_k_m = SelectWithDefault(self.Msel, self.pll_v, 0.) #K x V_perobj x P
        pw_k_m = SelectWithDefault(self.Msel, pw, 0.) #K x V-obj x P
        pw_k_sum = tf.reduce_sum(pw_k_m, axis=1)
        pw_k_sum = tf.where(pw_k_sum <= 0., 1e-2, pw_k_sum)
        
        pll_k = tf.math.divide_no_nan(tf.reduce_sum(pll_k_m * pw_k_m, axis=1), 
                                             pw_k_sum  )#K x P
        return pll_k
    
    def Beta_pen_k(self):
        #use continuous max approximation through LSE
        eps = 1e-3
        beta_pen = 1. - eps * tf.reduce_logsumexp(self.beta_k_m/eps, axis=1)#sum over m
        #for faster convergence  
        beta_pen += 1. - tf.clip_by_value(tf.reduce_sum(self.beta_k_m, axis=1), 0., 1)
        beta_pen = tf.debugging.check_numerics(beta_pen, "OC: beta pen")
        return beta_pen
        
    def Noise_pen(self):
        
        nsupp_v = self.beta_v * self.isn_v
        nsupp = tf.math.divide_no_nan(tf.reduce_sum(nsupp_v), 
                                      tf.reduce_sum(self.isn_v)+1e-3) # nodim
        
        specsupp_v = self.beta_v * self.sw_v
        specsupp = tf.math.divide_no_nan(tf.reduce_sum(specsupp_v), 
                                      tf.reduce_sum(self.sw_v)+1e-3) # nodim
        
        return self.s_b * nsupp + self.spect_supp * specsupp
        
    
    # doesn't do anything in this implementation
    def high_B_pen_k(self):
        return 0.* self.beta_k
    
    # override with more complex through inheritance
    def pll_weight_k(self, ow_k, vatt_k, vrep_k):
        return ow_k
    
    
        
    def add_to_terms(self,
                     V_att, 
                     V_rep,
                     Noise_pen, 
                     B_pen, 
                     pll,
                     high_B_pen
                     ):
        
        zero_tensor = tf.zeros_like(tf.reduce_mean(self.q_v,axis=0))
        
        if not self.valid: # no objects
            zero_payload = tf.zeros_like(tf.reduce_mean(self.pll_v,axis=0))
            print('WARNING: no objects in sample, continue to next')
            return zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_payload, zero_tensor
    
        K = tf.reduce_sum(tf.ones_like(self.q_k)) # > 0
        
        V_att_k = self.V_att_k()
        V_rep_k = self.V_rep_k()
        
        V_att += tf.reduce_sum(self.ow_k * V_att_k)/K
        V_rep += tf.reduce_sum(self.ow_k * V_rep_k)/K
        Noise_pen += self.Noise_pen()
        B_pen += tf.reduce_sum(self.ow_k * self.Beta_pen_k())/K
        
        pl_ow_k = self.pll_weight_k(self.ow_k, V_att_k, V_rep_k)
        pll += tf.reduce_sum(pl_ow_k * self.Pll_k(),axis=0)/K 
        
        high_B_pen += tf.reduce_sum(self.ow_k *self.high_B_pen_k())/K
        
        return V_att, V_rep, Noise_pen, B_pen, pll, high_B_pen
        