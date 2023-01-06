from os.path import join
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from lib.test_metrics import *
from lib.utils import to_numpy


def compare_acf2(x_real, x_fake, max_lag=5, dim=(0, 1)):
    """ Computes ACF of historical and (mean)-ACF of generated and plots those. """
    # if ax is None:
    #     _, ax = plt.subplots(1, 1)
    acf_real_list = cacf_torch(x_real, max_lag=max_lag, dim=dim).cpu().numpy()
    acf_real = np.mean(acf_real_list, axis=0)

    acf_fake_list = cacf_torch(x_fake, max_lag=max_lag, dim=dim).cpu().numpy()
    acf_fake = np.mean(acf_fake_list, axis=0)

    err = tf.math.squared_difference(acf_real, acf_fake)
    acf_dif = tf.reduce_sum(err)

    # ax.plot(acf_real[drop_first_n_lags:], label='Historical')
    # ax.plot(acf_fake[drop_first_n_lags:], label='Generated', alpha=0.8)

    # if CI:
    #     acf_fake_std = np.std(acf_fake_list, axis=0)
    #     ub = acf_fake + acf_fake_std
    #     lb = acf_fake - acf_fake_std

    #     for i in range(acf_real.shape[-1]):
    #         ax.fill_between(
    #             range(acf_fake[:, i].shape[0]),
    #             ub[:, i], lb[:, i],
    #             color='orange',
    #             alpha=.3
    #         )
    # set_style(ax)
    # ax.set_xlabel('Lags')
    # ax.set_ylabel('ACF')
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.grid(True)
    # ax.legend()

    return acf_dif



def compare_cross_corr(x_real, x_fake):
    """ Computes cross correlation matrices of x_real and x_fake and plots them. """
    x_real = x_real.reshape(-1, x_real.shape[2])
    x_fake = x_fake.reshape(-1, x_fake.shape[2])
    cc_real = np.corrcoef(to_numpy(x_real).T)
    cc_fake = np.corrcoef(to_numpy(x_fake).T)

    vmin = min(cc_fake.min(), cc_real.min())
    vmax = max(cc_fake.max(), cc_real.max())

    fig, axes = plt.subplots(1, 2)
    axes[0].matshow(cc_real, vmin=vmin, vmax=vmax)
    im = axes[1].matshow(cc_fake, vmin=vmin, vmax=vmax)

    axes[0].set_title('Real')
    axes[1].set_title('Generated')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


