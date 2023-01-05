
import os, warnings, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import tensorflow as tf 
tf.executing_eagerly()
# print 'vae _base_ cond-py tf-executing_eagerly: ', tf.executing_eagerly0))
tf.config. run_functions_eagerly(True)


from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import joblib 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import Mean
from tensorflow.keras.backend import random_normal

import torch

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class BaseVariationalAutoencoder(Model, ABC):
    def __init__(self,  
            seq_len, 
            feat_dim,  
            latent_dim,
            reconstruction_wt = 3.0,
            **kwargs  ):
        super(BaseVariationalAutoencoder, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean( name="reconstruction_loss" )
        self.kl_loss_tracker = Mean(name="kl_loss")
        self.acf_loss_tracker = Mean(name="acf_loss")

        self.encoder = None
        self.decoder = None


    def call(self, X):
        z_mean, _, _ = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        if len(x_decoded.shape) == 1: x_decoded = x_decoded.reshape((1, -1))
        return x_decoded


    def get_num_trainable_variables(self):
        trainableParams = int(np.sum([np.prod(v.get_shape()) for v in self.trainable_weights]))
        nonTrainableParams = int(np.sum([np.prod(v.get_shape()) for v in self.non_trainable_weights]))
        totalParams = trainableParams + nonTrainableParams
        return trainableParams, nonTrainableParams, totalParams


    def get_prior_samples(self, num_samples):
        Z = np.random.randn(num_samples, self.latent_dim)
        samples = self.decoder.predict(Z)
        return samples
    

    def get_prior_samples_given_Z(self, Z):
        samples = self.decoder.predict(Z)
        return samples


    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    
    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError


    def summary(self):
        self.encoder.summary()
        self.decoder.summary()



    def _get_reconstruction_loss(self, X, X_recons): 

        def get_reconst_loss_by_axis(X, X_c, axis): 
            x_r = tf.reduce_mean(X, axis = axis)
            x_c_r = tf.reduce_mean(X_recons, axis = axis)
            err = tf.math.squared_difference(x_r, x_c_r)
            loss = tf.reduce_sum(err)
            return loss

        # overall    
        err = tf.math.squared_difference(X, X_recons)
        reconst_loss = tf.reduce_sum(err)
      
        reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[2])     # by time axis        
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[1])    # by feature axis
        return reconst_loss
    
    def compare_acf2(self, x_real, x_fake, ops='original', max_lag=5, dim=(0, 1)):
        # Computes ACF of historical and (mean)-ACF of generated and plots those. ops: 'original','abs', 'square'

        def cacf_torch(x, max_lag, dim=(0, 1)):
            def get_lower_triangular_indices(n):
                return [list(x) for x in torch.tril_indices(n, n)]

            ind = get_lower_triangular_indices(x.shape[2])
            x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
            x_l = x[..., ind[0]]
            x_r = x[..., ind[1]]
            cacf_list = list()
            for i in range(max_lag):
                y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
                cacf_i = torch.mean(y, (1))
                cacf_list.append(cacf_i)
            cacf = torch.cat(cacf_list, 1)
            return cacf.reshape(cacf.shape[0], -1, len(ind[0]))

        # convert × real and x fake from TF to torch
        xr0 = x_real.numpy()
        xf0 = x_fake.numpy( )
        if ops == 'abs':
            xr = np.absolute(xr0)
            xf = np.absolute(xf0)
        elif ops =='square':
            xr = np.square (xr0)
            xf = np.square (xf0)
        else:
            xr = xr0
            xf = xf0

        # print('compare_acf2 tf.executing_eagerly:', tf.executing_eagerly())
        xr = torch.from_numpy (xr)
        xf = torch.from_numpy (xf)
        acf_real_list = cacf_torch(xr, max_lag=max_lag, dim=dim)
        acf_fake_list = cacf_torch(xf, max_lag=max_lag, dim=dim)
        err = tf.math.squared_difference(acf_real_list, acf_fake_list) 
        acf_dif = tf.reduce_mean(err)
        return acf_dif


    def train_step(self, X):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(X)

            reconstruction = self.decoder(z)

            reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
            # kl_loss = kl_loss / self.latent_dim

            # acf_loss = self.compare_acf2 (X, reconstruction, ops='original')
            acf_loss_original = self.compare_acf2 (X, reconstruction, ops='original')
            acf_loss_abs = self.compare_acf2 (X, reconstruction, ops='abs')
            acf_loss_square = self.compare_acf2 (X, reconstruction, ops='square')
            acf_loss = (acf_loss_original + acf_loss_abs + acf_loss_square) / 3

            total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss + acf_loss 

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.acf_loss_tracker.update_state(acf_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "acf_loss": self.acf_loss_tracker.result(),
        }

    
    def test_step(self, X): 
        z_mean, z_log_var, z = self.encoder(X)
        reconstruction = self.decoder(z)
        reconstruction_loss = self._get_reconstruction_loss(X, reconstruction)

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
        # kl_loss = kl_loss / self.latent_dim

        acf_loss_original = self.compare_acf2 (X, reconstruction, ops='original')
        acf_loss_abs = self.compare_acf2 (X, reconstruction, ops='abs')
        acf_loss_square = self.compare_acf2 (X, reconstruction, ops='square')
        acf_loss = (acf_loss_original + acf_loss_abs + acf_loss_square) / 3

        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss + acf_loss 

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.acf_loss_tracker.update_state(acf_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "acf_loss": self.acf_loss_tracker.result(),
        }


    def save_weights(self, model_dir, file_pref): 
        encoder_wts = self.encoder.get_weights()
        decoder_wts = self.decoder.get_weights()
        joblib.dump(encoder_wts, os.path.join(model_dir, f'{file_pref}encoder_wts.h5'))
        joblib.dump(decoder_wts, os.path.join(model_dir, f'{file_pref}decoder_wts.h5'))

    
    def load_weights(self, model_dir, file_pref):
        encoder_wts = joblib.load(os.path.join(model_dir, f'{file_pref}encoder_wts.h5'))
        decoder_wts = joblib.load(os.path.join(model_dir, f'{file_pref}decoder_wts.h5'))

        self.encoder.set_weights(encoder_wts)
        self.decoder.set_weights(decoder_wts)


    def save(self, model_dir, file_pref): 

        self.save_weights(model_dir, file_pref)
        dict_params = {

            'seq_len': self.seq_len,
            'feat_dim': self.feat_dim,
            'latent_dim': self.latent_dim,
            'reconstruction_wt': self.reconstruction_wt,
            'hidden_layer_sizes': self.hidden_layer_sizes,
        }
        params_file = os.path.join(model_dir, f'{file_pref}parameters.pkl') 
        joblib.dump(dict_params, params_file)


#####################################################################################################
#####################################################################################################


if __name__ == '__main__':

    pass