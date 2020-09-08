import abc
import typing
import weakref

import frozendict  # type: ignore
import numpy  # type: ignore
import pandas  # type: ignore

from d3m import container, exceptions, types
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, generator, transformer

__all__ = (
    'FileReaderPrimitiveBase',
    'DatasetSplitPrimitiveBase',
    'TabularSplitPrimitiveBase',
)

FileReaderInputs = container.DataFrame
FileReaderOutputs = container.DataFrame


class FileReaderHyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column does not contain filenames for supported media types, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='append',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should columns with read files be appended, should they replace original columns, or should only columns with read files be returned?",
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )


class FileReaderPrimitiveBase(transformer.TransformerPrimitiveBase[FileReaderInputs, FileReaderOutputs, FileReaderHyperparams]):
    """
    A primitive base class for reading files referenced in columns.

    Primitives using this base class must implement:

     * ``_supported_media_types``: A sequence of supported media types such as ``audio/mpeg``, ``image/jpeg``, etc.
     * ``_file_structural_type``: Structural type of the file contents after being read such as ``container.ndarray``, ``container.DataFrame``, etc.
     * ``_file_semantic_types``: A sequence of semantic types to be applied to the produced column.
     * ``metadata``: Primitive Metadata.
     * ``_read_fileuri``: The function which describes how to load each file. This function must load one file at the time.
    """

    _supported_media_types: typing.Sequence[str] = ()
    _file_structural_type: type = None
    # If any of these semantic types already exists on a column, then nothing is done.
    # If all are missing, the first one is set.
    _file_semantic_types: typing.Sequence[str] = ()

    def __init__(self, *, hyperparams: FileReaderHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

        # Because same file can be referenced multiple times in multiple rows, we maintain
        # a cache of read files so that we do not have to read same files again and again.
        self._cache: weakref.WeakValueDictionary[typing.Tuple[int, str], typing.Any] = weakref.WeakValueDictionary()

    def _can_use_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        if column_metadata['structural_type'] != str:
            return False

        semantic_types = column_metadata.get('semantic_types', [])
        media_types = set(column_metadata.get('media_types', []))

        if 'https://metadata.datadrivendiscovery.org/types/FileName' in semantic_types and media_types <= set(self._supported_media_types):
            return True

        return False

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata) -> typing.List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)

        # We are OK if no columns ended up being read.
        # "base_utils.combine_columns" will throw an error if it cannot work with this.

        if self.hyperparams['use_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified columns contain filenames for supported media types. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    def produce(self, *, inputs: FileReaderInputs, timeout: float = None, iterations: int = None) -> base.CallResult[FileReaderOutputs]:
        columns_to_use = self._get_columns(inputs.metadata)

        output_columns = [self._produce_column(inputs, column_index) for column_index in columns_to_use]

        outputs = base_utils.combine_columns(inputs, columns_to_use, output_columns, return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'])

        if self.hyperparams['return_result'] == 'append':
            outputs.metadata = self._reassign_boundaries(outputs.metadata, columns_to_use)

        return base.CallResult(outputs)

    @abc.abstractmethod
    def _read_fileuri(self, metadata: frozendict.FrozenOrderedDict, fileuri: str) -> typing.Any:
        pass

    def _read_filename(self, column_index: int, metadata: frozendict.FrozenOrderedDict, filename: str) -> typing.Any:
        # TODO: Support handling multiple "location_base_uris".
        # "location_base_uris" should be made so that we can just concat with the filename
        # ("location_base_uris" end with "/").
        fileuri = metadata['location_base_uris'][0] + filename

        # We do not use the structure where we check if the key exists in the cache and if not set it and then
        # return from the cache outside if clause because we are not sure garbage collection might not remove it
        # before we get to return. So we directly ask for a reference and return it, or we obtain the file
        # and populate the cache.
        file = self._cache.get((column_index, fileuri), None)
        if file is not None:
            return file

        file = self._read_fileuri(metadata, fileuri)

        # We cache the file based on column index as well, because it could be that file is read differently
        # based on column metadata, or that resulting metadata is different for a different column.
        # We cache only if we can make a weakref. Many Python built-in types like "str" do not support them.
        if type(file).__weakrefoffset__:
            self._cache[(column_index, fileuri)] = file

        return file

    def _produce_column(self, inputs: FileReaderInputs, column_index: int) -> FileReaderOutputs:
        read_files = [self._read_filename(column_index, inputs.metadata.query((row_index, column_index)), value) for row_index, value in enumerate(inputs.iloc[:, column_index])]

        column = container.DataFrame({inputs.columns[column_index]: read_files}, generate_metadata=False)

        column.metadata = self._produce_column_metadata(inputs.metadata, column_index, read_files)
        column.metadata = column.metadata.generate(column, compact=True)

        return column

    def _produce_column_metadata(
        self, inputs_metadata: metadata_base.DataMetadata, column_index: int, read_files: typing.Sequence[typing.Any],
    ) -> metadata_base.DataMetadata:
        column_metadata = inputs_metadata.select_columns([column_index])
        column_metadata = column_metadata.update_column(0, {
            'structural_type': self._file_structural_type,
            # Clear metadata useful for filename columns.
            'location_base_uris': metadata_base.NO_VALUE,
            'media_types': metadata_base.NO_VALUE,
        })

        # It is not a filename anymore.
        column_metadata = column_metadata.remove_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/FileName')

        # At least one semantic type from listed semantic types should be set.
        semantic_types = column_metadata.query_column(0).get('semantic_types', [])
        if not set(semantic_types) & set(self._file_semantic_types):
            # Add the first one.
            column_metadata = column_metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), self._file_semantic_types[0])

        for row_index, file in enumerate(read_files):
            # Copy metadata only if we have a container type.
            if isinstance(file, types.Container):
                column_metadata = file.metadata.copy_to(column_metadata, (), (row_index, 0))

        column_metadata = column_metadata.compact(['name', 'structural_type', 'media_types', 'location_base_uris', 'semantic_types'])

        return column_metadata

    def _reassign_boundaries(self, inputs_metadata: metadata_base.DataMetadata, columns: typing.List[int]) -> metadata_base.DataMetadata:
        """
        Moves metadata about boundaries from the filename column to image object column.
        """

        outputs_metadata = inputs_metadata
        columns_length = inputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        for column_index in range(columns_length):
            column_metadata = outputs_metadata.query_column(column_index)

            if 'boundary_for' not in column_metadata:
                continue

            # TODO: Support also "column_name" boundary metadata.
            if 'column_index' not in column_metadata['boundary_for']:
                continue

            try:
                i = columns.index(column_metadata['boundary_for']['column_index'])
            except ValueError:
                continue

            outputs_metadata = outputs_metadata.update_column(column_index, {
                'boundary_for': {
                    # We know that "columns" were appended at the end.
                    'column_index': columns_length - len(columns) + i,
                }
            })

        return outputs_metadata


DatasetSplitInputs = container.List
DatasetSplitOutputs = container.List


class DatasetSplitPrimitiveBase(generator.GeneratorPrimitiveBase[DatasetSplitOutputs, base.Params, base.Hyperparams]):
    """
    A base class for primitives which fit on a ``Dataset`` object to produce splits of that
    ``Dataset`` when producing. There are two produce methods: `produce` and `produce_score_data`.
    They take as an input a list of non-negative integers which identify which ``Dataset``
    splits to return.

    This class is parameterized using only by two type variables,
    ``Params`` and ``Hyperparams``.
    """

    @abc.abstractmethod
    def produce(self, *, inputs: DatasetSplitInputs, timeout: float = None, iterations: int = None) -> base.CallResult[DatasetSplitOutputs]:
        """
        For each input integer creates a ``Dataset`` split and produces the training ``Dataset`` object.
        This ``Dataset`` object should then be used to fit (train) the pipeline.
        """

    @abc.abstractmethod
    def produce_score_data(self, *, inputs: DatasetSplitInputs, timeout: float = None, iterations: int = None) -> base.CallResult[DatasetSplitOutputs]:
        """
        For each input integer creates a ``Dataset`` split and produces the scoring ``Dataset`` object.
        This ``Dataset`` object should then be used to test the pipeline and score the results.

        Output ``Dataset`` objects do not have targets redacted and are not directly suitable for testing.
        """

    @abc.abstractmethod
    def set_training_data(self, *, dataset: container.Dataset) -> None:  # type: ignore
        """
        Sets training data of this primitive, the ``Dataset`` to split.

        Parameters
        ----------
        dataset:
            The dataset to split.
        """


class TabularSplitPrimitiveParams(params.Params):
    dataset: typing.Optional[container.Dataset]
    main_resource_id: typing.Optional[str]
    splits: typing.Optional[typing.List[typing.Tuple[numpy.ndarray, numpy.ndarray]]]
    graph: typing.Optional[typing.Dict[str, typing.List[typing.Tuple[str, bool, int, int, typing.Dict]]]]


# TODO: Make clear the assumption that both output container type (List) and output Datasets should have metadata.
#       Redaction primitive expects that, while there is officially no reason for Datasets
#       to really have metadata: metadata is stored available on the input container type, not
#       values inside it.
class TabularSplitPrimitiveBase(DatasetSplitPrimitiveBase[TabularSplitPrimitiveParams, base.Hyperparams]):
    """
    A primitive base class for splitting tabular datasets.

    Primitives using this base class must implement:

    * ``_get_splits``: The function which describes how to split the tabular dataset.
    """

    def __init__(self, *, hyperparams: base.Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        # We need random seed multiple times. So we create our own random state we use everywhere.
        self._random_state = numpy.random.RandomState(self.random_seed)
        self._fitted: bool = False
        self._dataset: container.Dataset = None
        self._main_resource_id: str = None
        self._splits: typing.List[typing.Tuple[numpy.ndarray, numpy.ndarray]] = None
        self._graph: typing.Dict[str, typing.List[typing.Tuple[str, bool, int, int, typing.Dict]]] = None

    def produce(self, *, inputs: DatasetSplitInputs, timeout: float = None, iterations: int = None) -> base.CallResult[DatasetSplitOutputs]:
        return self._produce(inputs, True)

    def produce_score_data(self, *, inputs: DatasetSplitInputs, timeout: float = None, iterations: int = None) -> base.CallResult[DatasetSplitOutputs]:
        return self._produce(inputs, False)

    def set_training_data(self, *, dataset: container.Dataset) -> None:  # type: ignore
        main_resource_id, main_resource = base_utils.get_tabular_resource(dataset, None, has_hyperparameter=False)

        self._main_resource_id = main_resource_id
        self._dataset = dataset
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        """
        This function computes everything in advance, including generating the relation graph.
        """

        if self._dataset is None:
            raise exceptions.InvalidStateError('Missing training data.')

        if self._fitted:
            return base.CallResult(None)

        targets, target_columns = self._get_target_columns(self._dataset, self._main_resource_id)
        attributes = self._get_attribute_columns(self._dataset, self._main_resource_id, target_columns)

        # Get splits' indices.
        self._splits = self._get_splits(attributes, targets, self._dataset, self._main_resource_id)

        # Graph is the adjacency representation for the relations graph. Make it not be a "defaultdict".
        self._graph = dict(self._dataset.get_relations_graph())

        self._fitted = True

        return base.CallResult(None)

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: DatasetSplitInputs,  # type: ignore
                          dataset: container.Dataset, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, dataset=dataset)  # type: ignore

    @abc.abstractmethod
    def _get_splits(self, attributes: pandas.DataFrame, targets: pandas.DataFrame, dataset: container.Dataset, main_resource_id: str) -> typing.List[typing.Tuple[numpy.ndarray, numpy.ndarray]]:
        pass

    def _get_target_columns(self, dataset: container.Dataset, main_resource_id: str) -> typing.Tuple[pandas.DataFrame, typing.Sequence[int]]:
        target_columns = dataset.metadata.list_columns_with_semantic_types(['https://metadata.datadrivendiscovery.org/types/TrueTarget'], at=(main_resource_id,))

        # It is OK if there are no target columns. "_get_splits" should raise an exception
        # if this is a problem for a given split logic.

        return dataset[main_resource_id].iloc[:, list(target_columns)], target_columns

    def _get_attribute_columns(self, dataset: container.Dataset, main_resource_id: str, target_columns: typing.Sequence[int]) -> pandas.DataFrame:
        attribute_columns = dataset.metadata.list_columns_with_semantic_types(['https://metadata.datadrivendiscovery.org/types/Attribute'], at=(main_resource_id,))

        if not attribute_columns:
            # No attribute columns with semantic types, let's use all
            # non-target columns as attributes then.
            all_columns = list(range(dataset.metadata.query((main_resource_id, metadata_base.ALL_ELEMENTS,))['dimension']['length']))
            attribute_columns = [column_index for column_index in all_columns if column_index not in target_columns]

        if not attribute_columns:
            raise ValueError("No attribute columns.")

        return dataset[main_resource_id].iloc[:, list(attribute_columns)]

    def _produce(self, inputs: DatasetSplitInputs, is_train: bool) -> base.CallResult[DatasetSplitOutputs]:
        """
        This function splits the fitted Dataset.

        Parameters
        ----------
        inputs:
            A list of 0-based indices which specify which splits to be used as test split in output.
        is_train:
            Whether we are producing train or test data.

        Returns
        -------
        Returns a list of Datasets.
        """

        if not self._fitted:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        output_datasets = container.List(generate_metadata=True)

        for index in inputs:
            train_indices, test_indices = self._splits[index]

            if is_train:
                output_dataset = base_utils.sample_rows(
                    self._dataset,
                    self._main_resource_id,
                    set(train_indices),
                    self._graph,
                    delete_recursive=self.hyperparams.get('delete_recursive', False),
                )
            else:
                output_dataset = base_utils.sample_rows(
                    self._dataset,
                    self._main_resource_id,
                    set(test_indices),
                    self._graph,
                    delete_recursive=self.hyperparams.get('delete_recursive', False),
                )

            output_datasets.append(output_dataset)

        output_datasets.metadata = metadata_base.DataMetadata({
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.List,
            'dimension': {
                'length': len(output_datasets),
            },
        })

        # We update metadata based on metadata of each dataset.
        # TODO: In the future this might be done automatically by generate_metadata.
        #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/119
        for index, dataset in enumerate(output_datasets):
            output_datasets.metadata = dataset.metadata.copy_to(output_datasets.metadata, (), (index,))

        return base.CallResult(output_datasets)

    def get_params(self) -> TabularSplitPrimitiveParams:
        if not self._fitted:
            return TabularSplitPrimitiveParams(
                dataset=None,
                main_resource_id=None,
                splits=None,
                graph=None,
            )

        return TabularSplitPrimitiveParams(
            dataset=self._dataset,
            main_resource_id=self._main_resource_id,
            splits=self._splits,
            graph=self._graph,
        )

    def set_params(self, *, params: TabularSplitPrimitiveParams) -> None:
        self._dataset = params['dataset']
        self._main_resource_id = params['main_resource_id']
        self._splits = params['splits']
        self._graph = params['graph']
        self._fitted = all(param is not None for param in params.values())

    def __getstate__(self) -> dict:
        state = super().__getstate__()

        state['random_state'] = self._random_state

        return state

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)

        self._random_state = state['random_state']
