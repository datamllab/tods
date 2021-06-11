import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def series_segmentation(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def sine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    # timestamp = np.linspace(0, 10, length)
    timestamp = np.arange(length)
    value = np.sin(2 * np.pi * freq * timestamp)
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp * noise
    value = coef * value + offset
    return value

def cosine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    # timestamp = np.linspace(0, 10, length)
    timestamp = np.arange(length)
    value = np.cos(2 * np.pi * freq * timestamp)
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp * noise
    value = coef * value + offset
    return value


def square_sine(level=5, length=500, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    value = np.zeros(length)
    for i in range(level):
        value += 1 / (2 * i + 1) * sine(length=length, freq=freq * (2 * i + 1), coef=coef, offset=offset, noise_amp=noise_amp)
    return value


def collective_global_synthetic(length, base, coef=1.5, noise_amp=0.005):
    value = []
    norm = np.linalg.norm(base)
    base = base / norm
    num = int(length / len(base))
    for i in range(num):
        value.extend(base)
    residual = length - len(value)
    value.extend(base[:residual])
    value = np.array(value)
    noise = np.random.normal(0, 1, length)
    value = coef * value + noise_amp * noise
    return value


class MultivariateDataGenerator:
    def __init__(self, dim, stream_length, behavior, behavior_config=None):

        self.dim = dim
        self.STREAM_LENGTH = stream_length
        self.behavior = behavior if behavior is not None else [sine] * dim
        self.behavior_config = behavior_config if behavior_config is not None else [{}] * dim

        self.data = np.empty(shape=[0, stream_length],dtype=float)
        self.label = None
        self.data_origin = None
        self.timestamp = np.arange(self.STREAM_LENGTH)

        self.generate_timeseries()

    def generate_timeseries(self):
        for i in range(self.dim):
            self.behavior_config[i]['length'] = self.STREAM_LENGTH
            self.data = np.append(self.data, [self.behavior[i](**self.behavior_config[i])], axis=0)
        self.data_origin = self.data.copy()
        self.label = np.zeros(self.STREAM_LENGTH, dtype=int)

    def point_global_outliers(self, dim_no, ratio, factor, radius):
        """
        Add point global outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: the larger, the outliers are farther from inliers
            radius: the radius of collective outliers range
        """
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio)) * self.STREAM_LENGTH).astype(int)
        maximum, minimum = max(self.data[dim_no]), min(self.data[dim_no])
        for i in position:
            local_std = self.data_origin[dim_no][max(0, i - radius):min(i + radius, self.STREAM_LENGTH)].std()
            self.data[dim_no][i] = self.data_origin[dim_no][i] * factor * local_std
            if 0 <= self.data[dim_no][i] < maximum: self.data[dim_no][i] = maximum
            if 0 > self.data[dim_no][i] > minimum: self.data[dim_no][i] = minimum
            self.label[i] = 1

    def point_contextual_outliers(self,dim_no, ratio, factor, radius):
        """
        Add point contextual outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: the larger, the outliers are farther from inliers
                    Notice: point contextual outliers will not exceed the range of [min, max] of original data
            radius: the radius of collective outliers range
        """
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio)) * self.STREAM_LENGTH).astype(int)
        maximum, minimum = max(self.data[dim_no]), min(self.data[dim_no])
        for i in position:
            local_std = self.data_origin[dim_no][max(0, i - radius):min(i + radius, self.STREAM_LENGTH)].std()
            self.data[dim_no][i] = self.data_origin[dim_no][i] * factor * local_std
            if self.data[dim_no][i] > maximum: self.data[dim_no][i]= maximum * min(0.95, abs(np.random.normal(0, 1)))
            if self.data[dim_no][i] < minimum: self.data[dim_no][i] = minimum * min(0.95, abs(np.random.normal(0, 1)))

            self.label[i] = 1

    def collective_global_outliers(self,dim_no, ratio, radius, option='square', coef=3., noise_amp=0.0,
                                    level=5, freq=0.04, offset=0.0, # only used when option=='square'
                                    base=[0.,]): # only used when option=='other'
        """
        Add collective global outliers to original data
        Args:
            ratio: what ratio outliers will be added
            radius: the radius of collective outliers range
            option: if 'square': 'level' 'freq' and 'offset' are used to generate square sine wave
                    if 'other': 'base' is used to generate outlier shape
            level: how many sine waves will square_wave synthesis
            base: a list of values that we want to substitute inliers when we generate outliers
        """
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio / (2 * radius))) * self.STREAM_LENGTH).astype(int)

        valid_option = {'square', 'other'}
        if option not in valid_option:
            raise ValueError("'option' must be one of %r." % valid_option)

        if option == 'square':
            sub_data = square_sine(level=level, length=self.STREAM_LENGTH, freq=freq,
                                   coef=coef, offset=offset, noise_amp=noise_amp)
        else:
            sub_data = collective_global_synthetic(length=self.STREAM_LENGTH, base=base,
                                                   coef=coef, noise_amp=noise_amp)
        for i in position:
            start, end = max(0, i - radius), min(self.STREAM_LENGTH, i + radius)
            self.data[dim_no][start:end] = sub_data[start:end]
            self.label[start:end] = 1

    def collective_trend_outliers(self,dim_no, ratio, factor, radius):
        """
        Add collective trend outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: how dramatic will the trend be
            radius: the radius of collective outliers range
        """
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio / (2 * radius))) * self.STREAM_LENGTH).astype(int)
        for i in position:
            start, end = max(0, i - radius), min(self.STREAM_LENGTH, i + radius)
            slope = np.random.choice([-1, 1]) * factor * np.arange(end - start)
            self.data[dim_no][start:end] = self.data_origin[dim_no][start:end] + slope
            self.data[dim_no][end:] = self.data[dim_no][end:] + slope[-1]
            self.label[start:end] = 1

    def collective_seasonal_outliers(self,dim_no, ratio, factor, radius):
        """
        Add collective seasonal outliers to original data
        Args:
            ratio: what ratio outliers will be added
            factor: how many times will frequency multiple
            radius: the radius of collective outliers range
        """
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio / (2 * radius))) * self.STREAM_LENGTH).astype(int)
        seasonal_config = self.behavior_config[dim_no]
        seasonal_config['freq'] = factor * self.behavior_config[dim_no]['freq']
        for i in position:
            start, end = max(0, i - radius), min(self.STREAM_LENGTH, i + radius)
            self.data[dim_no][start:end] = self.behavior[dim_no](**seasonal_config)[start:end]
            self.label[start:end] = 1



if __name__ == '__main__':
    np.random.seed(100)

    BEHAVIOR = [sine, cosine, sine, cosine, sine]
    BEHAVIOR_CONFIG = [{'freq': 0.04, 'coef': 1.5, "offset": 0.0, 'noise_amp': 0.05},
                       {'freq': 0.04, 'coef': 2.5, "offset": 0.0, 'noise_amp': 0.05},
                       {'freq': 0.04, 'coef': 1.5, "offset": 0.0, 'noise_amp': 0.05},
                       {'freq': 0.04, 'coef': 2.5, "offset": 2.0, 'noise_amp': 0.05},
                       {'freq': 0.04, 'coef': 1.5, "offset": -2.0, 'noise_amp': 0.05},]


    multivariate_data = MultivariateDataGenerator(dim=5, stream_length=400, behavior=BEHAVIOR,
                                                behavior_config=BEHAVIOR_CONFIG)


    multivariate_data.point_global_outliers(dim_no=0, ratio=0.05, factor=3.5, radius=5)
    multivariate_data.point_contextual_outliers(dim_no=1, ratio=0.05, factor=2.5, radius=5)
    multivariate_data.collective_global_outliers(dim_no=2, ratio=0.05, radius=5, option='square', coef=1.5, noise_amp=0.03, level=20, freq=0.04, offset=0.0)
    multivariate_data.collective_seasonal_outliers(dim_no=3, ratio=0.05, factor=3, radius=5)
    multivariate_data.collective_trend_outliers(dim_no=4, ratio=0.05, factor=0.5, radius=5)

    df = pd.DataFrame({'col_0': multivariate_data.data[0],
                       'col_1': multivariate_data.data[1],
                       'col_2': multivariate_data.data[2],
                       'col_3': multivariate_data.data[3],
                       'col_4': multivariate_data.data[4],
                       'anomaly': multivariate_data.label})
    df.to_csv("/Users/apple/Desktop/tods/benchmark/multi_dataset/01234.csv", index=False)

    plt.figure(figsize=(10, 15))
    plt.subplot(511)
    plt.plot(multivariate_data.timestamp, multivariate_data.data[0])
    plt.subplot(512)
    plt.plot(multivariate_data.timestamp, multivariate_data.data[1])
    plt.subplot(513)
    plt.plot(multivariate_data.timestamp, multivariate_data.data[2])
    plt.subplot(514)
    plt.plot(multivariate_data.timestamp, multivariate_data.data[3])
    plt.subplot(515)
    plt.plot(multivariate_data.timestamp, multivariate_data.data[4])

    plt.savefig("/Users/apple/Desktop/tods/benchmark/multi_dataset/01234.jpg")
    plt.show()
