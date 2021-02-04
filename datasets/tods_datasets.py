import os
import pandas as pd

from tods_dataset_base import TODS_dataset
from shutil import copyfile

class kpi_dataset(TODS_dataset):
    resources = [
        # ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        # ("https://github.com/datamllab/tods/blob/master/datasets/anomaly/kpi/TRAIN/dataset_TRAIN/tables/learningData.csv", None),
        # ("https://github.com/NetManAIOps/KPI-Anomaly-Detection/blob/master/Preliminary_dataset/train.csv", None),
        ("https://hegsns.github.io/tods_datasets/kpi/TRAIN/dataset_TRAIN/tables/learningData.csv", None), # it needs md5 to check if local learningData.csv is the same with online.
        ("https://hegsns.github.io/tods_datasets/kpi/TRAIN/dataset_TRAIN/datasetDoc.json", None),
        # needs a server to store the dataset.
        # ("https://raw.githubusercontent.com/datamllab/tods/master/datasets/anomaly/kpi/TRAIN/dataset_TRAIN/tables/learningData.csv", None), # it needs md5 to check if local learningData.csv is the same with online.
    ]

    training_file = 'learningData.csv'
    testing_file = 'testingData.csv'
    ground_truth_index = 3
    _repr_indent = 4

    # def __init__(self, root, train, transform=None, target_transform=None, download=True):
    #     super().__init__(root, train, transform=None, target_transform=None, download=True)

    def process(self) -> None:

        print('Processing...')

        os.makedirs(self.processed_folder, exist_ok=True)
        os.makedirs(os.path.join(self.processed_folder, 'tables'), exist_ok=True)

        training_set_fname = os.path.join(self.raw_folder, 'learningData.csv')
        self.training_set_dataframe = pd.read_csv(training_set_fname)
        testing_set_fname = os.path.join(self.raw_folder, 'learningData.csv')  # temperarily same with training set
        self.testing_set_dataframe = pd.read_csv(testing_set_fname)

        self.process_dataframe()
        self.training_set_dataframe.to_csv(os.path.join(self.processed_folder, 'tables', self.training_file))
        self.testing_set_dataframe.to_csv(os.path.join(self.processed_folder, 'tables', self.testing_file))
        copyfile(os.path.join(self.raw_folder, 'datasetDoc.json'), os.path.join(self.processed_folder, 'datasetDoc.json'))

        print('Done!')


class yahoo_dataset(TODS_dataset):
    resources = [
        # ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        # ("https://github.com/datamllab/tods/blob/master/datasets/anomaly/kpi/TRAIN/dataset_TRAIN/tables/learningData.csv", None),
        # ("https://github.com/NetManAIOps/KPI-Anomaly-Detection/blob/master/Preliminary_dataset/train.csv", None),
        ("https://hegsns.github.io/tods_datasets/yahoo_sub_5/TRAIN/dataset_TRAIN/tables/learningData.csv", None), # it needs md5 to check if local learningData.csv is the same with online.
        ("https://hegsns.github.io/tods_datasets/yahoo_sub_5/TRAIN/dataset_TRAIN/datasetDoc.json", None),
        # needs a server to store the dataset.
        # ("https://raw.githubusercontent.com/datamllab/tods/master/datasets/anomaly/kpi/TRAIN/dataset_TRAIN/tables/learningData.csv", None), # it needs md5 to check if local learningData.csv is the same with online.
    ]

    training_file = 'learningData.csv'
    testing_file = 'testingData.csv'
    ground_truth_index = 7
    _repr_indent = 4

    def process(self) -> None:

        print('Processing...')

        os.makedirs(self.processed_folder, exist_ok=True)
        os.makedirs(os.path.join(self.processed_folder, 'tables'), exist_ok=True)

        training_set_fname = os.path.join(self.raw_folder, 'learningData.csv')
        self.training_set_dataframe = pd.read_csv(training_set_fname)
        testing_set_fname = os.path.join(self.raw_folder, 'learningData.csv')  # temperarily same with training set
        self.testing_set_dataframe = pd.read_csv(testing_set_fname)

        self.process_dataframe()
        self.training_set_dataframe.to_csv(os.path.join(self.processed_folder, 'tables', self.training_file))
        self.testing_set_dataframe.to_csv(os.path.join(self.processed_folder, 'tables', self.testing_file))
        copyfile(os.path.join(self.raw_folder, 'datasetDoc.json'), os.path.join(self.processed_folder, 'datasetDoc.json'))

        print('Done!')


class NAB_dataset(TODS_dataset):
    resources = [
        ("https://hegsns.github.io/tods_datasets/NAB/realTweets/labeled_Twitter_volume_AMZN.csv", None),
        # it needs md5 to check if local learningData.csv is the same with online.
        ("https://hegsns.github.io/tods_datasets/NAB/realTweets/labeled_Twitter_volume_AMZN.json", None),
        # needs a server to store the dataset.
    ]

    training_file = 'learningData.csv'
    testing_file = 'testingData.csv'
    ground_truth_index = 2
    _repr_indent = 4

    def process(self) -> None:
        print('Processing...')

        os.makedirs(self.processed_folder, exist_ok=True)
        os.makedirs(os.path.join(self.processed_folder, 'tables'), exist_ok=True)

        training_set_fname = os.path.join(self.raw_folder, 'labeled_Twitter_volume_AMZN.csv')
        self.training_set_dataframe = pd.read_csv(training_set_fname)
        testing_set_fname = os.path.join(self.raw_folder, 'labeled_Twitter_volume_AMZN.csv')  # temperarily same with training set
        self.testing_set_dataframe = pd.read_csv(testing_set_fname)

        self.process_dataframe()
        self.training_set_dataframe.to_csv(os.path.join(self.processed_folder, 'tables', self.training_file))
        self.testing_set_dataframe.to_csv(os.path.join(self.processed_folder, 'tables', self.testing_file))
        copyfile(os.path.join(self.raw_folder, 'labeled_Twitter_volume_AMZN.json'),
                 os.path.join(self.processed_folder, 'datasetDoc.json'))

        print('Done!')

# kpi_dataset(root='./datasets', train=True, transform='binarize')
# yahoo_dataset(root='./datasets', train=True, transform='binarize')
# NAB_dataset(root='./datasets', train=True, transform='binarize')
