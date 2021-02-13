import tensorflow as tf
import numpy as np
import pdb
import json
from mod_core_rnn_cell_impl import LSTMCell  # modified to allow initializing bias in lstm

import data_utils
import plotting
import model
import mmd
import utils
import eval
import DR_discriminator

from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
from differential_privacy.dp_sgd.dp_optimizer import sanitizer
from differential_privacy.privacy_accountant.tf import accountant

"""
Here, both the discriminator and generator were used to do the anomaly detection
"""

# --- get settings --- #
# parse command line arguments, or use defaults
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())
# if a settings file is specified, it overrides command line arguments/defaults
if settings['settings_file']: settings = utils.load_settings_from_file(settings)

# --- get data, split --- #
data_path = './experiments/data/' + settings['data_load_from'] + '.data.npy'
print('Loading data from', data_path)
samples, labels, index = data_utils.get_data(settings["data"], settings["seq_length"], settings["seq_step"],
                                             settings["num_signals"], settings["sub_id"], settings["eval_single"],
                                             settings["eval_an"], data_path)
# --- save settings, data --- #
# no need
print('Ready to run with settings:')
for (k, v) in settings.items(): print(v, '\t', k)
# add the settings to local environment
# WARNING: at this point a lot of variables appear
locals().update(settings)
json.dump(settings, open('./experiments/settings/' + identifier + '.txt', 'w'), indent=0)

class myADclass():
    def __init__(self, epoch, settings=settings, samples=samples, labels=labels, index=index):
        self.epoch = epoch
        self.settings = settings
        self.samples = samples
        self.labels = labels
        self.index = index
    def ADfunc(self):
        num_samples_t = self.samples.shape[0]
        t_size = 500
        T_index = np.random.choice(num_samples_t, size=t_size, replace=False)
        print('sample_shape:', self.samples.shape[0])
        print('num_samples_t', num_samples_t)

        # -- only discriminate one batch for one time -- #
        D_test = np.empty([t_size, self.settings['seq_length'], 1])
        DL_test = np.empty([t_size, self.settings['seq_length'], 1])
        GG = np.empty([t_size, self.settings['seq_length'], self.settings['num_signals']])
        T_samples = np.empty([t_size, self.settings['seq_length'], self.settings['num_signals']])
        L_mb = np.empty([t_size, self.settings['seq_length'], 1])
        I_mb = np.empty([t_size, self.settings['seq_length'], 1])
        for batch_idx in range(0, t_size):
            # print('epoch:{}'.format(self.epoch))
            # print('batch_idx:{}'.format(batch_idx))
            # display batch progress
            model.display_batch_progression(batch_idx, t_size)
            T_mb = self.samples[T_index[batch_idx], :, :]
            L_mmb = self.labels[T_index[batch_idx], :, :]
            I_mmb = self.index[T_index[batch_idx], :, :]
            para_path = './experiments/parameters/' + self.settings['sub_id'] + '_' + str(
                self.settings['seq_length']) + '_' + str(self.epoch) + '.npy'
            D_t, L_t = DR_discriminator.dis_D_model(self.settings, T_mb, para_path)
            Gs, Zs, error_per_sample, heuristic_sigma = DR_discriminator.invert(self.settings, T_mb, para_path,
                                                                                g_tolerance=None,
                                                                                e_tolerance=0.1, n_iter=None,
                                                                                max_iter=1000,
                                                                                heuristic_sigma=None)
            GG[batch_idx, :, :] = Gs
            T_samples[batch_idx, :, :] = T_mb
            D_test[batch_idx, :, :] = D_t
            DL_test[batch_idx, :, :] = L_t
            L_mb[batch_idx, :, :] = L_mmb
            I_mb[batch_idx, :, :] = I_mmb

        # -- use self-defined evaluation functions -- #
        # -- test different tao values for the detection function -- #
        results = np.empty([5, 5])
        # for i in range(2, 8):
        #     tao = 0.1 * i
        tao = 0.5
        lam = 0.8
        Accu1, Pre1, Rec1, F11, FPR1, D_L1 = DR_discriminator.detection_D_I(DL_test, L_mb, I_mb, self.settings['seq_step'], tao)
        print('seq_length:', self.settings['seq_length'])
        print('D:Comb-logits-based-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
              .format(self.epoch, tao, Accu1, Pre1, Rec1, F11, FPR1))
        results[0, :] = [Accu1, Pre1, Rec1, F11, FPR1]

        Accu2, Pre2, Rec2, F12, FPR2, D_L2 = DR_discriminator.detection_D_I(D_test, L_mb, I_mb, self.settings['seq_step'], tao)
        print('seq_length:', self.settings['seq_length'])
        print('D:Comb-statistic-based-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
              .format(self.epoch, tao, Accu2, Pre2, Rec2, F12, FPR2))
        results[1, :] = [Accu2, Pre2, Rec2, F12, FPR2]

        Accu3, Pre3, Rec3, F13, FPR3, D_L3 = DR_discriminator.detection_R_D_I(DL_test, GG, T_samples, L_mb, self.settings['seq_step'], tao, lam)
        print('seq_length:', self.settings['seq_length'])
        print('RD:Comb-logits_based-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
            .format(self.epoch, tao, Accu3, Pre3, Rec3, F13, FPR3))
        results[2, :] = [Accu3, Pre3, Rec3, F13, FPR3]

        Accu4, Pre4, Rec4, F14, FPR4, D_L4 = DR_discriminator.detection_R_D_I(D_test, GG, T_samples, L_mb, self.settings['seq_step'], tao, lam)
        print('seq_length:', self.settings['seq_length'])
        print('RD:Comb-statistic-based-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
              .format(self.epoch, tao, Accu4, Pre4, Rec4, F14, FPR4))
        results[3, :] = [Accu4, Pre4, Rec4, F14, FPR4]

        Accu5, Pre5, Rec5, F15, FPR5, D_L5 = DR_discriminator.detection_R_I(GG, T_samples, L_mb, self.settings['seq_step'],tao)
        print('seq_length:', self.settings['seq_length'])
        print('G:Comb-sample-based-Epoch: {}; tao={:.1}; Accu: {:.4}; Pre: {:.4}; Rec: {:.4}; F1: {:.4}; FPR: {:.4}'
              .format(self.epoch, tao, Accu5, Pre5, Rec5, F15, FPR5))
        results[4, :] = [Accu5, Pre5, Rec5, F15, FPR5]

        return results, GG, D_test, DL_test



if __name__ == "__main__":
    print('Main Starting...')

    Results = np.empty([settings['num_epochs'], 5, 5])

    t_size = 500
    D_test = np.empty([settings['num_epochs'], t_size, settings['seq_length'], 1])
    DL_test = np.empty([settings['num_epochs'], t_size, settings['seq_length'], 1])
    GG = np.empty([settings['num_epochs'], t_size, settings['seq_length'], settings['num_signals']])

    for epoch in range(settings['num_epochs']):
    # for epoch in range(1):
        ob = myADclass(epoch)
        Results[epoch, :, :], GG[epoch, :, :, :], D_test[epoch, :, :, :], DL_test[epoch, :, :, :] = ob.ADfunc()

    res_path = './experiments/plots/Results_Invert' + '_' + settings['sub_id'] + '_' + str(
        settings['seq_length']) + '.npy'
    np.save(res_path, Results)

    dg_path = './experiments/plots/DG_Invert' + '_' + settings['sub_id'] + '_' + str(
        settings['seq_length']) + '_'
    np.save(dg_path + 'D_test.npy', D_test)
    np.save(dg_path + 'DL_test.npy', DL_test)
    np.save(dg_path + 'GG.npy', DL_test)

    print('Main Terminating...')