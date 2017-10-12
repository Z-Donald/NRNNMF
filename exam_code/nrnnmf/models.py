from __future__ import absolute_import, print_function
"""Defines NRNNMF models."""
# Third party modules
import tensorflow as tf
import numpy as np
# Local modules
from .utils import build_mlp, mlp_out

class _NRNNMFBase(object):
    def __init__(self, num_users, num_items, drugMat, geneMat, D=5, Dprime=30, hidden_units_per_layer=25, cfix=3, alpha=0.5, beta=0.5, lambda_d=0.5, lambda_t=0.5, K1=5, K2=5,
                 latent_normal_init_params={'mean': 0.0, 'stddev': 0.1}):
        self.num_users = num_users
        self.num_items = num_items
        self.K1 = int(K1)
        self.K2 = int(K2)
        self.drugMat = drugMat
        self.geneMat = geneMat
        self.construct_neighborhood(drugMat, geneMat)
        self.D = D
        self.Dprime = Dprime
        self.hidden_units_per_layer = hidden_units_per_layer
        self.cfix = int(cfix)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lambda_d = float(lambda_d)
        self.lambda_t = float(lambda_t)
        self.latent_normal_init_params = latent_normal_init_params

        # Internal counter to keep track of current iteration
        self._iters = 0

        # Input
        self.user_index = tf.placeholder(tf.int32, [None])
        self.item_index = tf.placeholder(tf.int32, [None])
        self.r_target = tf.placeholder(tf.float32, [None])

        # Call methods to initialize variables and operations (to be implemented by children)
        self._init_vars()
        self._init_ops()

        # RMSE
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.r, self.r_target))))

    def _init_vars(self):
        raise NotImplementedError

    def _init_ops(self):
        raise NotImplementedError

    def init_sess(self, sess):
        self.sess = sess
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _train_iteration(self, data, additional_feed=None):
        user_ids = data['drug']
        item_ids = data['gene']
        ratings = data['interaction']

        feed_dict = {self.user_index: user_ids, self.item_index: item_ids, self.r_target: ratings}

        if additional_feed:
            feed_dict.update(additional_feed)

        for step in self.optimize_steps:
            self.sess.run(step, feed_dict=feed_dict)

        self._iters += 1

    def train_iteration(self, data):
        self._train_iteration(data)

    def eval_loss(self, data):
        raise NotImplementedError

    def predict(self, test_data, train_data):
        raise NotImplementedError

    def eval_rmse(self, data):
        user_ids = data['drug']
        item_ids = data['gene']
        ratings = data['interaction']

        feed_dict = {self.user_index: user_ids, self.item_index: item_ids, self.r_target: ratings}
        return self.sess.run(self.rmse, feed_dict=feed_dict)

class NRNNMF(_NRNNMFBase):
    def __init__(self, *args, **kwargs):
        if 'lam' in kwargs:
            self.lam = float(kwargs['lam'])
            del kwargs['lam']
        else:
            self.lam = 0.5

        super(NRNNMF, self).__init__(*args, **kwargs)

    def _init_vars(self):
        # Latents
        self.U = tf.Variable(tf.truncated_normal([self.num_users, self.D],   **self.latent_normal_init_params))
        self.Uprime = tf.Variable(tf.truncated_normal([self.num_users, self.Dprime], **self.latent_normal_init_params))
        self.V = tf.Variable(tf.truncated_normal([self.num_items, self.D], **self.latent_normal_init_params))
        self.Vprime = tf.Variable(tf.truncated_normal([self.num_items, self.Dprime], **self.latent_normal_init_params))

        # Lookups
        self.U_lu = tf.nn.embedding_lookup(self.U, self.user_index)
        self.Uprime_lu = tf.nn.embedding_lookup(self.Uprime, self.user_index)
        self.V_lu = tf.nn.embedding_lookup(self.V, self.item_index)
        self.Vprime_lu = tf.nn.embedding_lookup(self.Vprime, self.item_index)

        # MLP ("f")
        f_input_layer = tf.concat(values=[self.U_lu, self.V_lu, tf.multiply(self.Uprime_lu, self.Vprime_lu)], axis=1)

        _r, self.mlp_weights = build_mlp(f_input_layer, hidden_units_per_layer=self.hidden_units_per_layer)
        self.r = tf.squeeze(_r, squeeze_dims=[1])

    def _init_ops(self):
        # Loss
        reconstruction_loss = tf.reduce_sum(self.cfix * self.r_target * tf.log(self.r) + (1 - self.r_target) * tf.log(1 - self.r), axis=0)
        reg = tf.add_n([1/2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.U), self.lambda_d * tf.eye(self.num_users) + self.alpha * self.DL), self.U)),  
                        1/2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.V), self.lambda_t * tf.eye(self.num_items) + self.beta * self.TL), self.V)),
                        1/2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.Uprime), self.lambda_d * tf.eye(self.num_users) + self.alpha * self.DL), self.Uprime)),  
                        1/2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(self.Vprime), self.lambda_t * tf.eye(self.num_items) + self.beta * self.TL), self.Vprime))])
        self.loss = - reconstruction_loss + reg

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer()
        # Optimize the MLP weights
        f_train_step = self.optimizer.minimize(self.loss, var_list=self.mlp_weights.values())
        # Then optimize the latents
        latent_train_step = self.optimizer.minimize(self.loss, var_list=[self.U, self.Uprime, self.V, self.Vprime])

        self.optimize_steps = [f_train_step, latent_train_step]

    def eval_loss(self, data):
        user_ids = data['drug']
        item_ids = data['gene']
        ratings = data['interaction']

        feed_dict = {self.user_index: user_ids, self.item_index: item_ids, self.r_target: ratings}
        return self.sess.run(self.loss, feed_dict=feed_dict)

    def predict(self, test_data, train_data = None):
        if self.K2 == 0 or train_data is None:
            rating = self.sess.run(self.r, feed_dict={self.user_index: test_data.ix[:, 0], self.item_index: test_data.ix[:, 1]})
            return rating
        else:
            user_ids = test_data.ix[:, 0]
            item_ids = test_data.ix[:, 1]
            train_user_ids, train_item_ids = set(train_data.ix[:, 0].tolist()), set(train_data.ix[:, 1].tolist())
            dinx = np.array(list(train_user_ids))
            DS = self.drugMat.ix[:, dinx]
            tinx = np.array(list(train_item_ids))
            TS = self.geneMat.ix[:, tinx]
            tilde_U = self.U
            tilde_V = self.V
            tilde_Uprime = self.Uprime
            tilde_Vprime = self.Vprime
            for d, t in np.array([list(user_ids), list(item_ids)]).T:
                if d not in train_user_ids:
                    ii = np.argsort(DS[d, :])[::-1][:self.K2]
                    tilde_U[d, :] = tf.multiply(DS[d, ii], self.U[dinx[ii], :])/tf.reduce_sum(DS[d, ii], axis = 0)
                    tilde_Uprime[d, :] = tf.multiply(DS[d, ii], self.Uprime[dinx[ii], :])/tf.reduce_sum(DS[d, ii], axis = 0)
                elif t not in train_item_ids:
                    jj = np.argsort(TS[t, :])[::-1][:self.K2]
                    tilde_V[t, :] = tf.multiply(TS[t, jj], self.V[tinx[jj], :])/tf.reduce_sum(TS[t, jj], axis = 0)
                    tilde_Vprime[t, :] = tf.multiply(TS[t, jj], self.Vprime[tinx[jj], :])/tf.reduce_sum(TS[t, jj], axis = 0)
            tilde_U_lu = tf.nn.embedding_lookup(tilde_U, self.user_index)
            tilde_Uprime_lu = tf.nn.embedding_lookup(tilde_Uprime, self.user_index)
            tilde_V_lu = tf.nn.embedding_lookup(tilde_V, self.item_index)
            tilde_Vprime_lu = tf.nn.embedding_lookup(tilde_Vprime, self.item_index)
            f_input_layer = tf.concat(values=[tilde_U_lu, tilde_V_lu, tf.multiply(tilde_Uprime_lu, tilde_Vprime_lu)], axis=1)
            _rating = mlp_out(f_input_layer, self.mlp_weights)
            rating = tf.squeeze(_rating, squeeze_dims=[1])
            return self.sess.run(rating, feed_dict={self.user_index: test_data.ix[:, 0], self.item_index: test_data.ix[:, 1]})

    def construct_neighborhood(self, drugMat, targetMat):
        self.dsMat = drugMat - np.diag(np.diag(drugMat))
        self.tsMat = targetMat - np.diag(np.diag(targetMat))
        if self.K1 > 0:
            S1 = self.get_nearest_neighbors(self.dsMat, self.K1)
            self.DL = self.laplacian_matrix(S1)
            S2 = self.get_nearest_neighbors(self.tsMat, self.K1)
            self.TL = self.laplacian_matrix(S2)
        else:
            self.DL = self.laplacian_matrix(self.dsMat)
            self.TL = self.laplacian_matrix(self.tsMat)

    def laplacian_matrix(self, S):
        x = np.sum(S, axis=0)
        y = np.sum(S, axis=1)
        L = 0.5*(np.diag(x+y) - (S+S.T))  # neighborhood regularization matrix
        return L


    def get_nearest_neighbors(self, S, size=5):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(S.ix[i, :])[::-1][:min(size, n)]
            X[i, ii] = S.ix[i, ii]
        return X
