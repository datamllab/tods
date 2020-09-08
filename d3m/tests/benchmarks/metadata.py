import os
import tempfile

import numpy
import pandas

from d3m import container, utils

oneformat_dataset_json = """
{
  "about": {
    "datasetID": "benchmark_dataset",
    "datasetName": "benchmark_dataset_name",
    "license": "Unknown",
    "datasetSchemaVersion": "3.2.0",
    "redacted": false    
  },
  "dataResources": [
    {
      "resID": "0",
      "resPath": "media/",
      "resType": "image",
      "resFormat": [
        "image/png"
      ],
      "isCollection": true
    }
  ]
}
"""

twoformats_dataset_json = """
{
  "about": {
    "datasetID": "benchmark_dataset",
    "datasetName": "benchmark_dataset_name",
    "license": "Unknown",
    "datasetSchemaVersion": "3.2.0",
    "redacted": false    
  },
  "dataResources": [
    {
      "resID": "0",
      "resPath": "media/",
      "resType": "image",
      "resFormat": [
        "image/png",
        "image/jpeg"
      ],
      "isCollection": true
    }
  ]
}
"""


def create_oneformat_dataset(dataset_folder, n):
    media_folder = os.path.join(dataset_folder, 'media')
    os.makedirs(media_folder, mode=0o777, exist_ok=False)

    dataset_doc_file_path = os.path.join(dataset_folder, 'datasetDoc.json')
    dataset_doc_file_uri = "file://" + os.path.abspath(dataset_doc_file_path)

    with open(dataset_doc_file_path, 'w') as f:
        f.write(oneformat_dataset_json)

    filenames = ["image_{x}.png".format(x=x) for x in range(n)]

    for filename in filenames:
        with open(os.path.join(media_folder, filename), 'w') as f:
            pass

    return dataset_doc_file_uri


def create_twoformats_dataset(dataset_folder, n):
    media_folder = os.path.join(dataset_folder, 'media')
    os.makedirs(media_folder, mode=0o777, exist_ok=False)

    dataset_doc_file_path = os.path.join(dataset_folder, 'datasetDoc.json')
    dataset_doc_file_uri = "file://" + os.path.abspath(dataset_doc_file_path)

    with open(dataset_doc_file_path, 'w') as f:
        f.write(twoformats_dataset_json)

    filenames = ["image_{x}.{ext}".format(x=x, ext='png' if x % 2 else 'jpeg') for x in range(n)]

    for filename in filenames:
        with open(os.path.join(media_folder, filename), 'w') as f:
            pass

    return dataset_doc_file_uri


class OneFormatDataset:
    params = [[True, False], [10000, 30000, 50000]]
    param_names = ['compute_digest', 'dataset_files']

    def setup(self, compute_digest, dataset_files):
        self.temp_directory = tempfile.TemporaryDirectory()

        self.dataset_doc_file_uri = create_oneformat_dataset(self.temp_directory.name, dataset_files)

    def teardown(self, compute_digest, dataset_files):
        self.temp_directory.cleanup()

    def time_dataset_load(self, compute_digest, dataset_files):
        container.dataset.Dataset.load(self.dataset_doc_file_uri, compute_digest=container.ComputeDigest.ALWAYS if compute_digest else container.ComputeDigest.NEVER)


class TwoFormatsDataset:
    params = [[True, False], [10000, 30000, 50000]]
    param_names = ['compute_digest', 'dataset_files']

    def setup(self, compute_digest, dataset_files):
        self.temp_directory = tempfile.TemporaryDirectory()

        self.dataset_doc_file_uri = create_twoformats_dataset(self.temp_directory.name, dataset_files)

    def teardown(self, compute_digest, dataset_files):
        self.temp_directory.cleanup()

    def time_dataset_load(self, compute_digest, dataset_files):
        container.dataset.Dataset.load(self.dataset_doc_file_uri, compute_digest=container.ComputeDigest.ALWAYS if compute_digest else container.ComputeDigest.NEVER)


class DatasetToJsonStructure:
    params = [[10000, 30000, 50000]]
    param_names = ['dataset_files']

    def setup(self, dataset_files):
        self.temp_directory = tempfile.TemporaryDirectory()

        dataset_doc_file_uri = create_twoformats_dataset(self.temp_directory.name, dataset_files)

        self.dataset_metadata = container.dataset.Dataset.load(dataset_doc_file_uri).metadata

    def teardown(self, dataset_files):
        self.temp_directory.cleanup()

    def time_to_json_structure(self, dataset_files):
        self.dataset_metadata.to_internal_json_structure()

    def time_to_simple_structure_without_json(self, dataset_files):
        self.dataset_metadata.to_internal_simple_structure()

    def time_to_simple_structure_with_json(self, dataset_files):
        utils.to_json_structure(self.dataset_metadata.to_internal_simple_structure())


class MetadataGeneration:
    params = [[True, False]]
    param_names = ['compact']

    def setup(self, compact):
        self.large_dataframe_with_objects = pandas.DataFrame({str(i): [str(j) for j in range(10000)] for i in range(50)}, columns=[str(i) for i in range(50)])
        self.large_list_with_objects = [container.List([str(j) for i in range(50)]) for j in range(10000)]
        self.large_ndarray_with_objects = numpy.array([[[str(k) for k in range(5)] for i in range(10)] for j in range(10000)], dtype=object)
        self.large_dict_with_objects = {str(i): {str(j): j for j in range(10000)} for i in range(50)}

    def time_large_dataframe_with_objects(self, compact):
        df = container.DataFrame(self.large_dataframe_with_objects, generate_metadata=False)
        df.metadata.generate(df, compact=compact)

    def time_large_list_with_objects(self, compact):
        l = container.List(self.large_list_with_objects, generate_metadata=False)
        l.metadata.generate(l, compact=compact)

    def time_large_ndarray_with_objects(self, compact):
        a = container.ndarray(self.large_ndarray_with_objects, generate_metadata=False)
        a.metadata.generate(a, compact=compact)

    def time_large_dict_with_objects(self, compact):
        l = container.List([self.large_dict_with_objects], generate_metadata=False)
        l.metadata.generate(l, compact=compact)


class MetadataToJsonStructure:
    def setup(self):
        self.large_dataframe = container.DataFrame(pandas.DataFrame({str(i): [str(j) for j in range(10000)] for i in range(50)}, columns=[str(i) for i in range(50)]), generate_metadata=True)
        self.large_list = container.List([container.List([str(j) for i in range(50)]) for j in range(10000)], generate_metadata=True)
        self.large_ndarray = container.ndarray(numpy.array([[[str(k) for k in range(5)] for i in range(10)] for j in range(10000)], dtype=object), generate_metadata=True)
        self.large_dict_list = container.List({str(i): {str(j): j for j in range(10000)} for i in range(50)}, generate_metadata=True)

    def time_large_dataframe(self):
        self.large_dataframe.metadata.to_internal_json_structure()

    def time_large_list(self):
        self.large_list.metadata.to_internal_json_structure()

    def time_large_ndarray(self):
        self.large_ndarray.metadata.to_internal_json_structure()

    def time_large_dict_list(self):
        self.large_dict_list.metadata.to_internal_json_structure()
