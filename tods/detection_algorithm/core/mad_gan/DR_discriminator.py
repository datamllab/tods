import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from tods.detection_algorithm.core.mad_gan import model
from tods.detection_algorithm.core.mad_gan import mmd
from tods.detection_algorithm.core.mad_gan.mod_core_rnn_cell_impl import LSTMCell
from sklearn.metrics import precision_recall_fscore_support

def anomaly_detection_plot(D_test, T_mb, L_mb, D_L, epoch, identifier):

    aa = D_test.shape[0]
    bb = D_test.shape[1]
    D_L = D_L.reshape([aa, bb, -1])

    x_points = np.arange(bb)

    fig, ax = plt.subplots(4, 4, sharex=True)
    for m in range(4):
        for n in range(4):
            D = D_test[n * 4 + m, :, :]
            T = T_mb[n * 4 + m, :, :]
            L = L_mb[n * 4 + m, :, :]
            DL = D_L[n * 4 + m, :, :]
            ax[m, n].plot(x_points, D, '--g', label='Pro')
            ax[m, n].plot(x_points, T, 'b', label='Data')
            ax[m, n].plot(x_points, L, 'k', label='Label')
            ax[m, n].plot(x_points, DL, 'r', label='Label')
            ax[m, n].set_ylim(-1, 1)
    for n in range(4):
        ax[-1, n].xaxis.set_ticks(range(0, bb, int(bb/6)))
    fig.suptitle(epoch)
    fig.subplots_adjust(hspace=0.15)
    fig.savefig("./experiments/plots/DR_dis/" + identifier + "_epoch" + str(epoch).zfill(4) + ".png")
    plt.clf()
    plt.close()

    return True

def detection_Comb(Label_test, L_mb, I_mb, seq_step, tao):
    aa = Label_test.shape[0]
    bb = Label_test.shape[1]

    LL = (aa-1)*seq_step+bb

    Label_test = abs(Label_test.reshape([aa, bb]))
    L_mb = L_mb .reshape([aa, bb])
    I_mb = I_mb .reshape([aa, bb])
    D_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])
    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i*10+j)
            D_L[i*seq_step+j] += Label_test[i, j]
            L_L[i * seq_step + j] += L_mb[i, j]
            Count[i * seq_step + j] += 1

    D_L /= Count
    L_L /= Count

    TP, TN, FP, FN = 0, 0, 0, 0

    for i in range(LL):
        if D_L[i] > tao:
            # true/negative
            D_L[i] = 0
        else:
            # false/positive
            D_L[i] = 1

    cc = (D_L == L_L)
    # print('D_L:', D_L)
    # print('L_L:', L_L)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)

    print('N:', N)

    Accu = float((N / LL) * 100)

    precision, recall, f1, _ = precision_recall_fscore_support(L_L, D_L, average='binary')

    return Accu, precision, recall, f1,


def detection_logits_I(DL_test, L_mb, I_mb, seq_step, tao):
    aa = DL_test.shape[0]
    bb = DL_test.shape[1]

    LL = (aa-1)*seq_step+bb

    DL_test = abs(DL_test.reshape([aa, bb]))
    L_mb = L_mb .reshape([aa, bb])
    I_mb = I_mb .reshape([aa, bb])
    D_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])
    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i*10+j)
            D_L[i*seq_step+j] += DL_test[i, j]
            L_L[i * seq_step + j] += L_mb[i, j]
            Count[i * seq_step + j] += 1

    D_L /= Count
    L_L /= Count

    TP, TN, FP, FN = 0, 0, 0, 0

    for i in range(LL):
        if D_L[i] > tao:
            # true/negative
            D_L[i] = 0
        else:
            # false/positive
            D_L[i] = 1

        A = D_L[i]
        B = L_L[i]
        if A == 1 and B == 1:
            TP += 1
        elif A == 1 and B == 0:
            FP += 1
        elif A == 0 and B == 0:
            TN += 1
        elif A == 0 and B == 1:
            FN += 1


    cc = (D_L == L_L)
    # print('D_L:', D_L)
    # print('L_L:', L_L)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)

    print('N:', N)

    Accu = float((N / LL) * 100)

    precision, recall, f1, _ = precision_recall_fscore_support(L_L, D_L, average='binary')

    # true positive among all the detected positive
    # Pre = (100 * TP) / (TP + FP + 1)
    # # true positive among all the real positive
    # Rec = (100 * TP) / (TP + FN + 1)
    # # The F1 score is the harmonic average of the precision and recall,
    # # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    # F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate--false alarm rate
    FPR = (100 * FP) / (FP + TN+1)

    return Accu, precision, recall, f1, FPR, D_L

def detection_statistic_I(D_test, L_mb, I_mb, seq_step, tao):
    # point-wise detection for one dimension

    aa = D_test.shape[0]
    bb = D_test.shape[1]

    LL = (aa-1) * seq_step + bb
    # print('aa:', aa)
    # print('bb:', bb)
    # print('LL:', LL)

    D_test = D_test.reshape([aa, bb])
    L_mb = L_mb.reshape([aa, bb])
    I_mb = I_mb.reshape([aa, bb])
    D_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])
    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i * 10 + j)
            D_L[i * seq_step + j] += D_test[i, j]
            L_L[i * seq_step + j] += L_mb[i, j]
            Count[i * seq_step + j] += 1

    D_L /= Count
    L_L /= Count

    TP, TN, FP, FN = 0, 0, 0, 0

    for i in range(LL):
        if D_L[i] > tao:
            # true/negative
            D_L[i] = 0
        else:
            # false/positive
            D_L[i] = 1

        A = D_L[i]
        B = L_L[i]
        if A == 1 and B == 1:
            TP += 1
        elif A == 1 and B == 0:
            FP += 1
        elif A == 0 and B == 0:
            TN += 1
        elif A == 0 and B == 1:
            FN += 1

    cc = (D_L == L_L)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)
    Accu = float((N / LL) * 100)

    precision, recall, f1, _ = precision_recall_fscore_support(L_L, D_L, average='binary')

    # true positive among all the detected positive
    # Pre = (100 * TP) / (TP + FP + 1)
    # # true positive among all the real positive
    # Rec = (100 * TP) / (TP + FN + 1)
    # # The F1 score is the harmonic average of the precision and recall,
    # # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    # F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate--false alarm rate
    FPR = (100 * FP) / (FP + TN)

    return Accu, precision, recall, f1, FPR, D_L

def detection_D_I(DD, L_mb, I_mb, seq_step, tao):
    # point-wise detection for one dimension

    aa = DD.shape[0]
    bb = DD.shape[1]

    LL = (aa-1)*seq_step+bb

    DD = abs(DD.reshape([aa, bb]))
    L_mb = L_mb .reshape([aa, bb])
    I_mb = I_mb .reshape([aa, bb])
    D_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])
    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i*10+j)
            D_L[i*10+j] += DD[i, j]
            L_L[i * 10 + j] += L_mb[i, j]
            Count[i * 10 + j] += 1

    D_L /= Count
    L_L /= Count

    TP, TN, FP, FN = 0, 0, 0, 0

    for i in range(LL):
        if D_L[i] > tao:
            # true/negative
            D_L[i] = 0
        else:
            # false/positive
            D_L[i] = 1

        A = D_L[i]
        B = L_L[i]
        if A == 1 and B == 1:
            TP += 1
        elif A == 1 and B == 0:
            FP += 1
        elif A == 0 and B == 0:
            TN += 1
        elif A == 0 and B == 1:
            FN += 1


    cc = (D_L == L_L)
    # print('D_L:', D_L)
    # print('L_L:', L_L)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)

    print('N:', N)

    Accu = float((N / LL) * 100)

    # true positive among all the detected positive
    Pre = (100 * TP) / (TP + FP + 1)
    # true positive among all the real positive
    Rec = (100 * TP) / (TP + FN + 1)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate--false alarm rate
    FPR = (100 * FP) / (FP + TN+1)

    return Accu, Pre, Rec, F1, FPR, D_L

def detection_R_D_I(DD, Gs, T_mb, L_mb, seq_step, tao, lam):
    # point-wise detection for one dimension
    # (1-lambda)*R(x)+lambda*D(x)
    # lambda=0.5?
    # D_test, Gs, T_mb, L_mb  are of same size

    R = np.absolute(Gs - T_mb)
    R = np.mean(R, axis=2)
    aa = DD.shape[0]
    bb = DD.shape[1]

    LL = (aa - 1) * seq_step + bb

    DD = abs(DD.reshape([aa, bb]))
    DD = 1-DD
    L_mb = L_mb.reshape([aa, bb])
    R = R.reshape([aa, bb])

    D_L = np.zeros([LL, 1])
    R_L = np.zeros([LL, 1])
    L_L = np.zeros([LL, 1])
    L_pre = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])
    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i*10+j)
            D_L[i * 10 + j] += DD[i, j]
            L_L[i * 10 + j] += L_mb[i, j]
            R_L[i * 10 + j] += R[i, j]
            Count[i * 10 + j] += 1
    D_L /= Count
    L_L /= Count
    R_L /= Count

    TP, TN, FP, FN = 0, 0, 0, 0

    for i in range(LL):
        if (1-lam)*R_L[i] + lam*D_L[i] > tao:
            # false
            L_pre[i] = 1
        else:
            # true
            L_pre[i] = 0

        A = L_pre[i]
        # print('A:', A)
        B = L_L[i]
        # print('B:', B)
        if A == 1 and B == 1:
            TP += 1
        elif A == 1 and B == 0:
            FP += 1
        elif A == 0 and B == 0:
            TN += 1
        elif A == 0 and B == 1:
            FN += 1

    cc = (L_pre == L_L)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)
    Accu = float((N / (aa*bb)) * 100)

    # true positive among all the detected positive
    Pre = (100 * TP) / (TP + FP + 1)
    # true positive among all the real positive
    Rec = (100 * TP) / (TP + FN + 1)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate
    FPR = (100 * FP) / (FP + TN+1)

    return Accu, Pre, Rec, F1, FPR, L_pre

def detection_R_I(Gs, T_mb, L_mb, seq_step, tao):
    # point-wise detection for one dimension
    # (1-lambda)*R(x)+lambda*D(x)
    # lambda=0.5?
    # D_test, Gs, T_mb, L_mb  are of same size

    R = np.absolute(Gs - T_mb)
    R = np.mean(R, axis=2)
    aa = R.shape[0]
    bb = R.shape[1]

    LL = (aa - 1) * seq_step + bb

    L_mb = L_mb.reshape([aa, bb])
    R = R.reshape([aa, bb])

    L_L = np.zeros([LL, 1])
    R_L = np.zeros([LL, 1])
    L_pre = np.zeros([LL, 1])
    Count = np.zeros([LL, 1])
    for i in range(0, aa):
        for j in range(0, bb):
            # print('index:', i*10+j)
            L_L[i * 10 + j] += L_mb[i, j]
            R_L[i * 10 + j] += R[i, j]
            Count[i * 10 + j] += 1
    L_L /= Count
    R_L /= Count

    TP, TN, FP, FN = 0, 0, 0, 0

    for i in range(LL):
        if R_L[i] > tao:
            # false
            L_pre[i] = 1
        else:
            # true
            L_pre[i] = 0

        A = L_pre[i]
        B = L_L[i]
        if A == 1 and B == 1:
            TP += 1
        elif A == 1 and B == 0:
            FP += 1
        elif A == 0 and B == 0:
            TN += 1
        elif A == 0 and B == 1:
            FN += 1

    cc = (L_pre == L_L)
    cc = list(cc.reshape([-1]))
    N = cc.count(True)
    Accu = float((N / (aa*bb)) * 100)

    # true positive among all the detected positive
    Pre = (100 * TP) / (TP + FP + 1)
    # true positive among all the real positive
    Rec = (100 * TP) / (TP + FN + 1)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate
    FPR = (100 * FP) / (FP + TN+1)

    return Accu, Pre, Rec, F1, FPR, L_pre


def sample_detection(D_test, L_mb, tao):
    # sample-wise detection for one dimension

    aa = D_test.shape[0]
    bb = D_test.shape[1]

    D_test = D_test.reshape([aa, bb])
    L_mb = L_mb.reshape([aa, bb])
    L = np.sum(L_mb, 1)
    # NN = 0-10
    L[L > 0] = 1

    D_L = np.empty([aa, ])

    for i in range(aa):
        if np.mean(D_test[i, :]) > tao:
            # true/negative
            D_L[i] = 0
        else:
            # false/positive
            D_L[i] = 1

    cc = (D_L == L)
    # cc = list(cc)
    N = list(cc).count(True)
    Accu = float((N / (aa)) * 100)

    precision, recall, f1, _ = precision_recall_fscore_support(L, D_L, average='binary')

    return Accu, precision, recall, f1


def CUSUM_det(spe_n, spe_a, labels):

    mu = np.mean(spe_n)
    sigma = np.std(spe_n)

    kk = 3*sigma
    H = 15*sigma
    print('H:', H)

    tar = np.mean(spe_a)

    mm = spe_a.shape[0]

    SH = np.empty([mm, ])
    SL = np.empty([mm, ])

    for i in range(mm):
        SH[-1] = 0
        SL[-1] = 0
        SH[i] = max(0, SH[i-1]+spe_a[i]-(tar+kk))
        SL[i] = min(0, SL[i-1]+spe_a[i]-(tar-kk))


    count = np.empty([mm, ])
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(mm):
        A = SH[i]
        B = SL[i]
        AA = H
        BB = -H
        if A <= AA and B >= BB:
            count[i] = 0
        else:
            count[i] = 1

        C = count[i]
        D = labels[i]
        if C == 1 and D == 1:
            TP += 1
        elif C == 1 and D == 0:
            FP += 1
        elif C == 0 and D == 0:
            TN += 1
        elif C == 0 and D == 1:
            FN += 1

    cc = (count == labels)
    # cc = list(cc)
    N = list(cc).count(True)
    Accu = float((N / (mm)) * 100)

    # true positive among all the detected positive
    Pre = (100 * TP) / (TP + FP + 1)
    # true positive among all the real positive
    Rec = (100 * TP) / (TP + FN)
    # The F1 score is the harmonic average of the precision and recall,
    # where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
    F1 = (2 * Pre * Rec) / (100 * (Pre + Rec + 1))
    # False positive rate
    FPR = (100 * FP) / (FP + TN)

    return Accu, Pre, Rec, F1, FPR


def SPE(X, pc):
    a = X.shape[0]
    b = X.shape[1]

    spe = np.empty([a])
    # Square Prediction Error (square of residual distance)
    #  spe = X'(I-PP')X
    I = np.identity(b, float) - np.matmul(pc.transpose(1, 0), pc)
    # I = np.matmul(I, I)
    for i in range(a):
        x = X[i, :].reshape([b, 1])
        y = np.matmul(x.transpose(1, 0), I)
        spe[i] = np.matmul(y, x)

    return spe



def generator_o(z, hidden_units_g, seq_length, batch_size, num_generated_features, reuse=False, parameters=None, learn_scale=True):
    """
    If parameters are supplied, initialise as such
    """
    # It is important to specify different variable scopes for the LSTM cells.
    with tf.variable_scope("generator_o") as scope:

        W_out_G_initializer = tf.constant_initializer(value=parameters['generator/W_out_G:0'])
        b_out_G_initializer = tf.constant_initializer(value=parameters['generator/b_out_G:0'])
        try:
            scale_out_G_initializer = tf.constant_initializer(value=parameters['generator/scale_out_G:0'])
        except KeyError:
            scale_out_G_initializer = tf.constant_initializer(value=1)
            assert learn_scale
        lstm_initializer = tf.constant_initializer(value=parameters['generator/rnn/lstm_cell/weights:0'])
        bias_start = parameters['generator/rnn/lstm_cell/biases:0']

        W_out_G = tf.get_variable(name='W_out_G', shape=[hidden_units_g, num_generated_features], initializer=W_out_G_initializer)
        b_out_G = tf.get_variable(name='b_out_G', shape=num_generated_features, initializer=b_out_G_initializer)
        scale_out_G = tf.get_variable(name='scale_out_G', shape=1, initializer=scale_out_G_initializer, trainable=False)

        inputs = z

        cell = LSTMCell(num_units=hidden_units_g,
                        state_is_tuple=True,
                        initializer=lstm_initializer,
                        bias_start=bias_start,
                        reuse=reuse)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=[seq_length] * batch_size,
            inputs=inputs)
        rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
        logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G #out put weighted sum
        output_2d = tf.nn.tanh(logits_2d) # logits operation [-1, 1]
        output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
    return output_3d


def discriminator_o(x, hidden_units_d, reuse=False, parameters=None):

    with tf.variable_scope("discriminator_0") as scope:

        W_out_D_initializer = tf.constant_initializer(value=parameters['discriminator/W_out_D:0'])
        b_out_D_initializer = tf.constant_initializer(value=parameters['discriminator/b_out_D:0'])

        W_out_D = tf.get_variable(name='W_out_D', shape=[hidden_units_d, 1],  initializer=W_out_D_initializer)
        b_out_D = tf.get_variable(name='b_out_D', shape=1, initializer=b_out_D_initializer)


        inputs = x

        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_d, state_is_tuple=True, reuse=reuse)

        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float32, inputs=inputs)


        logits = tf.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D # output weighted sum

        output = tf.nn.sigmoid(logits) # y = 1 / (1 + exp(-x)). output activation [0, 1]. Probability??
        # sigmoid output ([0,1]), Probability?

    return output, logits


def invert(settings, samples, para_path, g_tolerance=None, e_tolerance=0.1,
           n_iter=None, max_iter=10000, heuristic_sigma=None):
    """
    Return the latent space points corresponding to a set of a samples
    ( from gradient descent )
    Note: this function is designed for ONE sample generation
    """
    # num_samples = samples.shape[0]
    # cast samples to float32

    samples = np.float32(samples)

    # get the model
    # if settings is a string, assume it's an identifier and load
    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))



    # print('Inverting', 1, 'samples using model', settings['identifier'], 'at epoch', epoch,)
    # if not g_tolerance is None:
    #     print('until gradient norm is below', g_tolerance)
    # else:
    #     print('until error is below', e_tolerance)


    # get parameters
    parameters = model.load_parameters(para_path)
    # # assertions
    # assert samples.shape[2] == settings['num_generated_features']
    # create VARIABLE Z
    Z = tf.get_variable(name='Z', shape=[1, settings['seq_length'],
                                         settings['latent_dim']],
                        initializer=tf.random_normal_initializer())
    # create outputs

    G_samples = generator_o(Z, settings['hidden_units_g'], settings['seq_length'],
                          1, settings['num_generated_features'],
                          reuse=False, parameters=parameters)
    # generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 'num_generated_features', 'cond_dim', 'learn_scale']
    # generator_settings = dict((k, settings[k]) for k in generator_vars)
    # G_samples = model.generator(Z, **generator_settings, reuse=True)

    fd = None

    # define loss mmd-based loss
    if heuristic_sigma is None:
        heuristic_sigma = mmd.median_pairwise_distance_o(samples)  # this is noisy
        print('heuristic_sigma:', heuristic_sigma)
    samples = tf.reshape(samples, [1, settings['seq_length'], settings['num_generated_features']])
    Kxx, Kxy, Kyy, wts = mmd._mix_rbf_kernel(G_samples, samples, sigmas=tf.constant(value=heuristic_sigma, shape=(1, 1)))
    similarity_per_sample = tf.diag_part(Kxy)
    reconstruction_error_per_sample = 1 - similarity_per_sample
    # reconstruction_error_per_sample = tf.reduce_sum((tf.nn.l2_normalize(G_samples, dim=1) - tf.nn.l2_normalize(samples, dim=1))**2, axis=[1,2])
    similarity = tf.reduce_mean(similarity_per_sample)
    reconstruction_error = 1 - similarity
    # updater
    #    solver = tf.train.AdamOptimizer().minimize(reconstruction_error_per_sample, var_list=[Z])
    # solver = tf.train.RMSPropOptimizer(learning_rate=500).minimize(reconstruction_error, var_list=[Z])
    solver = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(reconstruction_error_per_sample, var_list=[Z])
    # solver = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(reconstruction_error_per_sample, var_list=[Z])

    grad_Z = tf.gradients(reconstruction_error_per_sample, Z)[0]
    grad_per_Z = tf.norm(grad_Z, axis=(1, 2))
    grad_norm = tf.reduce_mean(grad_per_Z)
    # solver = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(reconstruction_error, var_list=[Z])
    print('Finding latent state corresponding to samples...')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        error = sess.run(reconstruction_error, feed_dict=fd)
        g_n = sess.run(grad_norm, feed_dict=fd)
        # print(g_n)
        i = 0
        if not n_iter is None:
            while i < n_iter:
                _ = sess.run(solver, feed_dict=fd)
                error = sess.run(reconstruction_error, feed_dict=fd)
                i += 1
        else:
            if not g_tolerance is None:
                while g_n > g_tolerance:
                    _ = sess.run(solver, feed_dict=fd)
                    error, g_n = sess.run([reconstruction_error, grad_norm], feed_dict=fd)
                    i += 1
                    print(error, g_n)
                    if i > max_iter:
                        break
            else:
                while np.abs(error) > e_tolerance:
                    _ = sess.run(solver, feed_dict=fd)
                    error = sess.run(reconstruction_error, feed_dict=fd)
                    i += 1
                    # print(error)
                    if i > max_iter:
                        break
        Zs = sess.run(Z, feed_dict=fd)
        Gs = sess.run(G_samples, feed_dict={Z: Zs})
        error_per_sample = sess.run(reconstruction_error_per_sample, feed_dict=fd)
        print('Z found in', i, 'iterations with final reconstruction error of', error)
    tf.reset_default_graph()

    return Gs, Zs, error_per_sample, heuristic_sigma


def dis_trained_model(settings, samples, para_path):
    """
    Return the discrimination results of  num_samples testing samples from a trained model described by settings dict
    Note: this function is designed for ONE sample discrimination
    """

    # if settings is a string, assume it's an identifier and load
    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))

    num_samples = samples.shape[0]
    samples = np.float32(samples)
    num_variables = samples.shape[2]
    # samples = np.reshape(samples, [1, settings['seq_length'], settings['num_generated_features']])

    # get the parameters, get other variables
    # parameters = model.load_parameters(settings['sub_id'] + '_' + str(settings['seq_length']) + '_' + str(epoch))
    parameters = model.load_parameters(para_path)
    # settings['sub_id'] + '_' + str(settings['seq_length']) + '_' + str(epoch)

    # create placeholder, T samples
    # T = tf.placeholder(tf.float32, [settings['batch_size'], settings['seq_length'], settings['num_generated_features']])

    T = tf.placeholder(tf.float32, [num_samples, settings['seq_length'], num_variables])

    # create the discriminator (GAN)
    # normal GAN
    D_t, L_t = discriminator_o(T, settings['hidden_units_d'], reuse=False, parameters=parameters)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # with tf.device('/gpu:1'):
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        D_t, L_t = sess.run([D_t, L_t], feed_dict={T: samples})

    tf.reset_default_graph()
    return D_t, L_t

def dis_D_model(settings, samples, para_path):
    """
    Return the discrimination results of  num_samples testing samples from a trained model described by settings dict
    Note: this function is designed for ONE sample discrimination
    """

    # if settings is a string, assume it's an identifier and load
    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))

    # num_samples = samples.shape[0]
    samples = np.float32(samples)
    samples = np.reshape(samples, [1, settings['seq_length'], settings['num_generated_features']])

    # get the parameters, get other variables
    parameters = model.load_parameters(para_path)
    # create placeholder, T samples

    T = tf.placeholder(tf.float32, [1, settings['seq_length'], settings['num_generated_features']])

    # create the discriminator (GAN or CGAN)
    # normal GAN
    D_t, L_t = discriminator_o(T, settings['hidden_units_d'], reuse=False, parameters=parameters)
    # D_t, L_t = model.discriminator(T, settings['hidden_units_d'], settings['seq_length'], num_samples, reuse=False,
    #               parameters=parameters, cond_dim=0, c=None, batch_mean=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        D_t, L_t = sess.run([D_t, L_t], feed_dict={T: samples})

    tf.reset_default_graph()
    return D_t, L_t