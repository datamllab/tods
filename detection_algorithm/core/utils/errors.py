import numpy as np
import pandas as pd
import more_itertools as mit
import os
import logging

logger = logging.getLogger('telemanom')


class Errors:
    def __init__(self, channel, window_size,batch_size, smoothing_perc,n_predictions,l_s,error_buffer,p):
        """
        Batch processing of errors between actual and predicted values
        for a channel.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
            config (obj): Config object containing parameters for processing
            run_id (str): Datetime referencing set of predictions in use

        Attributes:
            config (obj): see Args
            window_size (int): number of trailing batches to use in error
                calculation
            n_windows (int): number of windows in test values for channel
            i_anom (arr): indices of anomalies in channel test values
            E_seq (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in test values
            anom_scores (arr): score indicating relative severity of each
                anomaly sequence in E_seq
            e (arr): errors in prediction (predicted - actual)
            e_s (arr): exponentially-smoothed errors in prediction
            normalized (arr): prediction errors as a percentage of the range
                of the channel values
        """

        # self.config = config


        self._window_size =window_size
        self._batch_size = batch_size
        self._smoothing_perc = smoothing_perc
        self._n_predictions = n_predictions
        self._l_s = l_s
        self._error_buffer = error_buffer
        self._p = p


        self.window_size = self._window_size
        self.n_windows = int((channel.y_test.shape[0] -
                              (self._batch_size * self.window_size))
                             / self._batch_size)
        self.i_anom = np.array([])
        self.E_seq = []
        self.anom_scores = []
        channel.y_test = np.reshape(channel.y_test,(channel.X_test.shape[0],self._n_predictions,channel.X_test.shape[2]))
        # print("*****************************")
        # print("y_hat shape",channel.y_hat.shape)
        # print("y_test shape",channel.y_test.shape)

        channel.y_hat = np.reshape(channel.y_hat, (channel.y_hat.shape[0]*channel.y_hat.shape[1]*channel.y_hat.shape[2]))
        channel.y_test = np.reshape(channel.y_test, (channel.y_test.shape[0]*channel.y_test.shape[1]*channel.y_test.shape[2]))

        # print("after y_hat shape",channel.y_hat.shape)
        # print(" after y_test shape",channel.y_test.shape)
        
        
        # raw prediction error
        self.e = [abs(y_h-y_t) for y_h, y_t in
                  zip(channel.y_hat, channel.y_test)]

        self.e = np.reshape(self.e,(channel.X_test.shape[0],self._n_predictions,channel.X_test.shape[2]))
        # print("raw shape",self.e.shape)

        n_pred = self._n_predictions
        n_feature = channel.X_test.shape[2]

        # Aggregation for point wise
        # aggregated_error = np.zeros(n_feature*(len(self.e)+n_pred-1))
        # aggregated_error = np.reshape(aggregated_error,((len(self.e)+n_pred-1),n_feature))
        # # print(aggregated_error)
        # for i in range(0,len(self.e)):
        #     for j in range(len(self.e[i])):
        #         aggregated_error[i+j]+= self.e[i][j]
                
        # for i in range(1, len(aggregated_error)+1):
        #     if i < n_pred:
        #         aggregated_error[i-1] /=i 
        #     elif len(aggregated_error) - i+1 < n_pred:
        #         aggregated_error[i-1]/= len(aggregated_error) - i+1
        #     else:
        #         aggregated_error[i-1] /=n_pred 

        # Aggregation sequence wise
        aggregated_error = []
        for i in range(0,len(self.e)):
            aggregated_error.append(np.sum(self.e[i],axis=0))

        aggregated_error = np.asarray(aggregated_error)
        # print(aggregated_error.shape)
        
        smoothing_window = int(self._batch_size * self._window_size
                               * self._smoothing_perc)
        if not len(channel.y_hat) == len(channel.y_test):
            raise ValueError('len(y_hat) != len(y_test): {}, {}'
                             .format(len(channel.y_hat), len(channel.y_test)))

        # smoothed prediction error
        self.e_s = pd.DataFrame(aggregated_error).ewm(span=smoothing_window)\
            .mean().values.flatten()

        # print("ES",self.e_s)
        # print("ES",self.e_s.shape)
        # for values at beginning < sequence length, just use avg
        # if not channel.id == 'C-2':  # anomaly occurs early in window
            
            # print("LHS",self.e_s[:self.config.l_s])
            # print("RHS",[np.mean(self.e_s[:self.config.l_s * 2])] * self.config.l_s)
            # b = [np.mean(self.e_s[:self.config.l_s * 2])] * self.config.l_s
            # print("RHS shape",len(b))
            # self.e_s[:self._l_s] = [np.mean(self.e_s[:self._l_s * 2])] * self._l_s

        # np.save(os.path.join('data', run_id, 'smoothed_errors', '{}.npy'
        #                      .format(channel.id)),
        #         np.array(self.e_s))

        self.normalized = np.mean(self.e / np.ptp(channel.y_test))
        # logger.info("normalized prediction error: {0:.2f}"
        #             .format(self.normalized))

    def adjust_window_size(self, channel):
        """
        Decrease the historical error window size (h) if number of test
        values is limited.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        while self.n_windows < 0:
            self.window_size -= 1
            self.n_windows = int((channel.y_test.shape[0]
                                 - (self._batch_size * self.window_size))
                                 / self._batch_size)
            if self.window_size == 1 and self.n_windows < 0:
                raise ValueError('Batch_size ({}) larger than y_test (len={}). '
                                 'Adjust in config.yaml.'
                                 .format(self._batch_size,
                                         channel.y_test.shape[0]))

    def merge_scores(self):
        """
        If anomalous sequences from subsequent batches are adjacent they
        will automatically be combined. This combines the scores for these
        initial adjacent sequences (scores are calculated as each batch is
        processed) where applicable.
        """

        merged_scores = []
        score_end_indices = []

        for i, score in enumerate(self.anom_scores):
            if not score['start_idx']-1 in score_end_indices:
                merged_scores.append(score['score'])
                score_end_indices.append(score['end_idx'])

    def process_batches(self, channel):
        """
        Top-level function for the Error class that loops through batches
        of values for a channel.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        self.adjust_window_size(channel)

        for i in range(0, self.n_windows+1):
            prior_idx = i * self._batch_size
            idx = (self._window_size * self._batch_size) \
                  + (i * self._batch_size)
            if i == self.n_windows:
                idx = channel.y_test.shape[0]

            window = ErrorWindow(channel, prior_idx, idx, self, i,self._l_s,self._error_buffer,self._batch_size,self._p)

            window.find_epsilon()
            window.find_epsilon(inverse=True)

            window.compare_to_epsilon(self)
            window.compare_to_epsilon(self, inverse=True)

            if len(window.i_anom) == 0 and len(window.i_anom_inv) == 0:
                continue

            window.prune_anoms()
            window.prune_anoms(inverse=True)

            if len(window.i_anom) == 0 and len(window.i_anom_inv) == 0:
                continue

            window.i_anom = np.sort(np.unique(
                np.append(window.i_anom, window.i_anom_inv))).astype('int')
            window.score_anomalies(prior_idx)
            # print("window anom scores", window.anom_scores)

            # update indices to reflect true indices in full set of values
            self.i_anom = np.append(self.i_anom, window.i_anom + prior_idx)
            self.anom_scores = self.anom_scores + window.anom_scores

        if len(self.i_anom) > 0:
            # group anomalous indices into continuous sequences
            groups = [list(group) for group in
                      mit.consecutive_groups(self.i_anom)]
            self.E_seq = [(int(g[0]), int(g[-1])) for g in groups
                          if not g[0] == g[-1]]

            # additional shift is applied to indices so that they represent the
            # position in the original data array, obtained from the .npy files,
            # and not the position on y_test (See PR #27).
            self.E_seq = [(e_seq[0] + self._l_s,
                           e_seq[1] + self._l_s) for e_seq in self.E_seq]

            self.merge_scores()


class ErrorWindow:
    def __init__(self, channel,start_idx, end_idx, errors, window_num,l_s,error_buffer,batch_size,p):
        """
        Data and calculations for a specific window of prediction errors.
        Includes finding thresholds, pruning, and scoring anomalous sequences
        for errors and inverted errors (flipped around mean) - significant drops
        in values can also be anomalous.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
            config (obj): Config object containing parameters for processing
            start_idx (int): Starting index for window within full set of
                channel test values
            end_idx (int): Ending index for window within full set of channel
                test values
            errors (arr): Errors class object
            window_num (int): Current window number within channel test values

        Attributes:
            i_anom (arr): indices of anomalies in window
            i_anom_inv (arr): indices of anomalies in window of inverted
                telemetry values
            E_seq (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in window
            E_seq_inv (arr of tuples): array of (start, end) indices for each
                continuous anomaly sequence in window of inverted telemetry
                values
            non_anom_max (float): highest smoothed error value below epsilon
            non_anom_max_inv (float): highest smoothed error value below
                epsilon_inv
            config (obj): see Args
            anom_scores (arr): score indicating relative severity of each
                anomaly sequence in E_seq within a window
            window_num (int): see Args
            sd_lim (int): default number of standard deviations to use for
                threshold if no winner or too many anomalous ranges when scoring
                candidate thresholds
            sd_threshold (float): number of standard deviations for calculation
                of best anomaly threshold
            sd_threshold_inv (float): same as above for inverted channel values
            e_s (arr): exponentially-smoothed prediction errors in window
            e_s_inv (arr): inverted e_s
            sd_e_s (float): standard deviation of e_s
            mean_e_s (float): mean of e_s
            epsilon (float): threshold for e_s above which an error is
                considered anomalous
            epsilon_inv (float): threshold for inverted e_s above which an error
                is considered anomalous
            y_test (arr): Actual telemetry values for window
            sd_values (float): st dev of y_test
            perc_high (float): the 95th percentile of y_test values
            perc_low (float): the 5th percentile of y_test values
            inter_range (float): the range between perc_high - perc_low
            num_to_ignore (int): number of values to ignore initially when
                looking for anomalies
        """

        self._l_s = l_s
        self._error_buffer = error_buffer
        self._batch_size = batch_size
        self._p = p


        self.i_anom = np.array([])
        self.E_seq = np.array([])
        self.non_anom_max = -1000000
        self.i_anom_inv = np.array([])
        self.E_seq_inv = np.array([])
        self.non_anom_max_inv = -1000000

        # self.config = config
        self.anom_scores = []

        self.window_num = window_num

        self.sd_lim = 12.0
        self.sd_threshold = self.sd_lim
        self.sd_threshold_inv = self.sd_lim

        self.e_s = errors.e_s[start_idx:end_idx]

        self.mean_e_s = np.mean(self.e_s)
        self.sd_e_s = np.std(self.e_s)
        self.e_s_inv = np.array([self.mean_e_s + (self.mean_e_s - e)
                                 for e in self.e_s])

        self.epsilon = self.mean_e_s + self.sd_lim * self.sd_e_s
        self.epsilon_inv = self.mean_e_s + self.sd_lim * self.sd_e_s

        self.y_test = channel.y_test[start_idx:end_idx]
        self.sd_values = np.std(self.y_test)

        self.perc_high, self.perc_low = np.percentile(self.y_test, [95, 5])
        self.inter_range = self.perc_high - self.perc_low

        # ignore initial error values until enough history for processing
        self.num_to_ignore = self._l_s * 2
        # if y_test is small, ignore fewer
        if len(channel.y_test) < 2500:
            self.num_to_ignore = self._l_s
        if len(channel.y_test) < 1800:
            self.num_to_ignore = 0

    def find_epsilon(self, inverse=False):
        """
        Find the anomaly threshold that maximizes function representing
        tradeoff between:
            a) number of anomalies and anomalous ranges
            b) the reduction in mean and st dev if anomalous points are removed
            from errors
        (see https://arxiv.org/pdf/1802.04431.pdf)

        Args:
            inverse (bool): If true, epsilon is calculated for inverted errors
        """
        e_s = self.e_s if not inverse else self.e_s_inv

        max_score = -10000000

        for z in np.arange(2.5, self.sd_lim, 0.5):
            epsilon = self.mean_e_s + (self.sd_e_s * z)

            pruned_e_s = e_s[e_s < epsilon]

            i_anom = np.argwhere(e_s >= epsilon).reshape(-1,)
            buffer = np.arange(1, self._error_buffer)
            i_anom = np.sort(np.concatenate((i_anom,
                                            np.array([i+buffer for i in i_anom])
                                             .flatten(),
                                            np.array([i-buffer for i in i_anom])
                                             .flatten())))
            i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
            i_anom = np.sort(np.unique(i_anom))

            if len(i_anom) > 0:
                # group anomalous indices into continuous sequences
                groups = [list(group) for group
                          in mit.consecutive_groups(i_anom)]
                E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

                mean_perc_decrease = (self.mean_e_s - np.mean(pruned_e_s)) \
                                     / self.mean_e_s
                sd_perc_decrease = (self.sd_e_s - np.std(pruned_e_s)) \
                                   / self.sd_e_s
                score = (mean_perc_decrease + sd_perc_decrease) \
                        / (len(E_seq) ** 2 + len(i_anom))

                # sanity checks / guardrails
                if score >= max_score and len(E_seq) <= 5 and \
                        len(i_anom) < (len(e_s) * 0.5):
                    max_score = score
                    if not inverse:
                        self.sd_threshold = z
                        self.epsilon = self.mean_e_s + z * self.sd_e_s
                    else:
                        self.sd_threshold_inv = z
                        self.epsilon_inv = self.mean_e_s + z * self.sd_e_s

    def compare_to_epsilon(self, errors_all, inverse=False):
        """
        Compare smoothed error values to epsilon (error threshold) and group
        consecutive errors together into sequences.

        Args:
            errors_all (obj): Errors class object containing list of all
            previously identified anomalies in test set
        """

        e_s = self.e_s if not inverse else self.e_s_inv
        epsilon = self.epsilon if not inverse else self.epsilon_inv

        # Check: scale of errors compared to values too small?
        if not (self.sd_e_s > (.05 * self.sd_values) or max(self.e_s)
                > (.05 * self.inter_range)) or not max(self.e_s) > 0.05:
            return

        i_anom = np.argwhere((e_s >= epsilon) &
                             (e_s > 0.05 * self.inter_range)).reshape(-1,)

        if len(i_anom) == 0:
            return
        buffer = np.arange(1, self._error_buffer+1)
        i_anom = np.sort(np.concatenate((i_anom,
                                         np.array([i + buffer for i in i_anom])
                                         .flatten(),
                                         np.array([i - buffer for i in i_anom])
                                         .flatten())))
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]

        # if it is first window, ignore initial errors (need some history)
        if self.window_num == 0:
            i_anom = i_anom[i_anom >= self.num_to_ignore]
        else:
            i_anom = i_anom[i_anom >= len(e_s) - self._batch_size]

        i_anom = np.sort(np.unique(i_anom))

        # capture max of non-anomalous values below the threshold
        # (used in filtering process)
        batch_position = self.window_num * self._batch_size
        window_indices = np.arange(0, len(e_s)) + batch_position
        adj_i_anom = i_anom + batch_position
        window_indices = np.setdiff1d(window_indices,
                                      np.append(errors_all.i_anom, adj_i_anom))
        candidate_indices = np.unique(window_indices - batch_position)
        non_anom_max = np.max(np.take(e_s, candidate_indices))

        # group anomalous indices into continuous sequences
        groups = [list(group) for group in mit.consecutive_groups(i_anom)]
        E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

        if inverse:
            self.i_anom_inv = i_anom
            self.E_seq_inv = E_seq
            self.non_anom_max_inv = non_anom_max
        else:
            self.i_anom = i_anom
            self.E_seq = E_seq
            self.non_anom_max = non_anom_max

    def prune_anoms(self, inverse=False):
        """
        Remove anomalies that don't meet minimum separation from the next
        closest anomaly or error value

        Args:
            inverse (bool): If true, epsilon is calculated for inverted errors
        """

        E_seq = self.E_seq if not inverse else self.E_seq_inv
        e_s = self.e_s if not inverse else self.e_s_inv
        non_anom_max = self.non_anom_max if not inverse \
            else self.non_anom_max_inv

        if len(E_seq) == 0:
            return

        E_seq_max = np.array([max(e_s[e[0]:e[1]+1]) for e in E_seq])
        E_seq_max_sorted = np.sort(E_seq_max)[::-1]
        E_seq_max_sorted = np.append(E_seq_max_sorted, [non_anom_max])

        i_to_remove = np.array([])
        for i in range(0, len(E_seq_max_sorted)-1):
            if (E_seq_max_sorted[i] - E_seq_max_sorted[i+1]) \
                    / E_seq_max_sorted[i] < self._p:
                i_to_remove = np.append(i_to_remove, np.argwhere(
                    E_seq_max == E_seq_max_sorted[i]))
            else:
                i_to_remove = np.array([])
        i_to_remove[::-1].sort()

        if len(i_to_remove) > 0:
            E_seq = np.delete(E_seq, i_to_remove, axis=0)

        if len(E_seq) == 0 and inverse:
            self.i_anom_inv = np.array([])
            return
        elif len(E_seq) == 0 and not inverse:
            self.i_anom = np.array([])
            return

        indices_to_keep = np.concatenate([range(e_seq[0], e_seq[-1]+1)
                                          for e_seq in E_seq])

        if not inverse:
            mask = np.isin(self.i_anom, indices_to_keep)
            self.i_anom = self.i_anom[mask]
        else:
            mask_inv = np.isin(self.i_anom_inv, indices_to_keep)
            self.i_anom_inv = self.i_anom_inv[mask_inv]

    def score_anomalies(self, prior_idx):
        """
        Calculate anomaly scores based on max distance from epsilon
        for each anomalous sequence.

        Args:
            prior_idx (int): starting index of window within full set of test
                values for channel
        """

        groups = [list(group) for group in mit.consecutive_groups(self.i_anom)]

        for e_seq in groups:

            score_dict = {
                "start_idx": e_seq[0] + prior_idx,
                "end_idx": e_seq[-1] + prior_idx,
                "score": 0
            }

            score = max([abs(self.e_s[i] - self.epsilon)
                         / (self.mean_e_s + self.sd_e_s) for i in
                         range(e_seq[0], e_seq[-1] + 1)])
            inv_score = max([abs(self.e_s_inv[i] - self.epsilon_inv)
                             / (self.mean_e_s + self.sd_e_s) for i in
                             range(e_seq[0], e_seq[-1] + 1)])

            # the max score indicates whether anomaly was from regular
            # or inverted errors
            score_dict['score'] = max([score, inv_score])
            self.anom_scores.append(score_dict)