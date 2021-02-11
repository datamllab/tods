import warnings
import os
import os.path
import numpy as np
import codecs
import string
import gzip
import lzma
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from dataset_utils import download_url, download_and_extract_archive, extract_archive, verify_str_arg

# tqdm >= 4.31.1

from tods import generate_dataset
from sklearn import preprocessing
import pandas as pd

class TODS_dataset:
    resources = []
    training_file = None
    testing_file = None
    ground_truth_index = None
    _repr_indent = None

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def __init__(self, root, train, transform=None, download=True):

        self.root = root
        self.train = train
        self.transform = self.transform_init(transform)

        if download:
            self.download()
        pass

        self.process()


    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.testing_file)))


    def download(self) -> None:

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)


    def process(self) -> None:

        pass


    def process_dataframe(self) -> None:

        if self.transform is None:
            pass

        else:
            self.transform.fit(self.training_set_dataframe)
            self.training_set_array = self.transform.transform(self.training_set_dataframe.values)
            self.testing_set_array = self.transform.transform(self.testing_set_dataframe.values)
            self.training_set_dataframe = pd.DataFrame(self.training_set_array)
            self.testing_set_dataframe = pd.DataFrame(self.testing_set_array)


    def transform_init(self, transform_str):

        if transform_str is None:
            return None
        elif transform_str == 'standardscale':
            return preprocessing.StandardScaler()
        elif transform_str == 'normalize':
            return preprocessing.Normalizer()
        elif transform_str == 'minmaxscale':
            return preprocessing.MinMaxScaler()
        elif transform_str == 'maxabsscale':
            return preprocessing.MaxAbsScaler()
        elif transform_str == 'binarize':
            return preprocessing.Binarizer()
        else:
            raise ValueError("Input parameter transform must take value of 'standardscale', 'normalize', " +
                             "'minmaxscale', 'maxabsscale' or 'binarize'."
                             )


    def to_axolotl_dataset(self):
        if self.train:
            return generate_dataset(self.training_set_dataframe, self.ground_truth_index)
        else:
            return generate_dataset(self.testing_set_dataframe, self.ground_truth_index)


    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]

        print(self.training_set_dataframe)

        return '\n'.join(lines)


    def __len__(self) -> int:
        return len(self.training_set_dataframe)


    def extra_repr(self) -> str:
        return ""


# kpi(root='./datasets', train=True)

# class yahoo5:
#
#     def __init__(self):
#         pass