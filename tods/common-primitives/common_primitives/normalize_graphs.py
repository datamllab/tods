import os
import typing

import networkx  # type: ignore
import pandas  # type: ignore

from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('NormalizeGraphsPrimitive',)

Inputs = container.Dataset
Outputs = container.Dataset


class Hyperparams(hyperparams.Hyperparams):
    pass


class NormalizeGraphsPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which converts all graphs found in a dataset into a standard two-table representation
    (one of nodes and one of edges, using foreign keys to link between nodes and edges).

    See for more information `this issue`_.

    .. _this issue: https://gitlab.com/datadrivendiscovery/d3m/issues/134
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'dbb3792d-a44b-4941-a88e-5520c0a23488',
            'version': '0.1.0',
            'name': "Normalize graphs",
            'python_path': 'd3m.primitives.data_transformation.normalize_graphs.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/normalize_graphs.py',
                    'https://gitlab.com/datadrivendiscovery/common-primitives.git',
                ],
            },
            'installation': [{
               'type': metadata_base.PrimitiveInstallationType.PIP,
               'package_uri': 'git+https://gitlab.com/datadrivendiscovery/common-primitives.git@{git_commit}#egg=common_primitives'.format(
                   git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
               ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        outputs = inputs.copy()

        for resource_id, resource in inputs.items():
            if isinstance(resource, networkx.classes.graph.Graph):
                self._convert_networkx(outputs, resource_id)

            if isinstance(resource, container.DataFrame) and outputs.metadata.has_semantic_type((resource_id,), 'https://metadata.datadrivendiscovery.org/types/EdgeList'):
                self._update_edge_list(outputs, resource_id)

        return base.CallResult(outputs)

    def _convert_networkx(self, dataset: container.Dataset, resource_id: str) -> None:
        resource = dataset[resource_id]

        # DataFrame index contains networkX node IDs (which come from GML node IDs).
        # We see them as internal to the networkX structure and we use them only
        # to align nodes with edges but then discard them.
        nodes = pandas.DataFrame.from_dict(resource.nodes, orient='index')

        if len(nodes) != len(resource.nodes):
            raise exceptions.InvalidStateError(f"Converted nodes DataFrame has {len(nodes)} nodes, but graph has {len(resource.nodes)} nodes.")

        if not nodes.loc[:, 'nodeID'].is_unique:
            raise exceptions.UnexpectedValueError(f"'nodeID' column should be unique, but it is not in the graph with resource ID '{resource_id}'.")
        if nodes.loc[:, 'nodeID'].hasnans:
            raise exceptions.UnexpectedValueError(f"'nodeID' column should not have missing values, but it has them in the graph with resource ID '{resource_id}'.")

        # "source" and "target" columns point to "nodes" index values, not "nodeID" column.
        # TODO: What if edge attributes contain "source" and "target" keys?
        edgelist = networkx.to_pandas_edgelist(resource)

        # We map "source" and "target" columns to "nodeID" column.
        edgelist.loc[:, 'source'] = edgelist.loc[:, 'source'].apply(lambda s: nodes.loc[s, 'nodeID'])
        edgelist.loc[:, 'target'] = edgelist.loc[:, 'target'].apply(lambda s: nodes.loc[s, 'nodeID'])

        nodes = container.DataFrame(nodes, metadata=dataset.metadata.query((resource_id,)), generate_metadata=True)
        edgelist = container.DataFrame(edgelist, metadata=dataset.metadata.query((resource_id,)), generate_metadata=True)

        nodes_resource_id = f'{resource_id}_nodes'
        edgelist_resource_id = f'{resource_id}_edges'

        if nodes_resource_id in dataset:
            raise exceptions.AlreadyExistsError(f"Resource with ID '{nodes_resource_id}' already exists.")
        if edgelist_resource_id in dataset:
            raise exceptions.AlreadyExistsError(f"Resource with ID '{edgelist_resource_id}' already exists.")

        node_id_column_index = nodes.metadata.get_column_index_from_column_name('nodeID')

        nodes.metadata = nodes.metadata.update((), {
            'dimension': {
                'name': 'nodes',
            },
        })
        nodes.metadata = nodes.metadata.remove_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/EdgeList')
        nodes.metadata = nodes.metadata.add_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/Graph')
        nodes.metadata = nodes.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, metadata_base.ALL_ELEMENTS),
            'https://metadata.datadrivendiscovery.org/types/Attribute',
        )
        nodes.metadata = nodes.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, metadata_base.ALL_ELEMENTS),
            'https://metadata.datadrivendiscovery.org/types/UnknownType',
        )
        nodes.metadata = nodes.metadata.update(
            (metadata_base.ALL_ELEMENTS, node_id_column_index),
            {
                'semantic_types': [
                    # "nodeID" is always an integer.
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                ],
            },
        )

        source_column_index = edgelist.metadata.get_column_index_from_column_name('source')
        target_column_index = edgelist.metadata.get_column_index_from_column_name('target')

        edgelist.metadata = edgelist.metadata.update((), {
            'dimension': {
                'name': 'edges',
            },
        })
        edgelist.metadata = edgelist.metadata.remove_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/Graph')
        edgelist.metadata = edgelist.metadata.add_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/EdgeList')
        edgelist.metadata = edgelist.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, metadata_base.ALL_ELEMENTS),
            'https://metadata.datadrivendiscovery.org/types/Attribute',
        )
        edgelist.metadata = edgelist.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, metadata_base.ALL_ELEMENTS),
            'https://metadata.datadrivendiscovery.org/types/UnknownType',
        )
        edgelist.metadata = edgelist.metadata.update(
            (metadata_base.ALL_ELEMENTS, source_column_index),
            {
                'semantic_types': [
                    # "nodeID" is always an integer.
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
                'foreign_key': {
                    'type': 'COLUMN',
                    'resource_id': nodes_resource_id,
                    'column_index': node_id_column_index,
                },
            },
        )
        edgelist.metadata = edgelist.metadata.update(
            (metadata_base.ALL_ELEMENTS, target_column_index),
            {
                'semantic_types': [
                    # "nodeID" is always an integer.
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
                'foreign_key': {
                    'type': 'COLUMN',
                    'resource_id': nodes_resource_id,
                    'column_index': node_id_column_index,
                },
            },
        )

        directed = isinstance(resource, networkx.DiGraph)
        multi_graph = isinstance(resource, networkx.MultiGraph)

        edgelist.metadata = self._set_edges_metadata(
            edgelist.metadata,
            source_column_index,
            target_column_index,
            directed=directed,
            multi_graph=multi_graph,
            at=(),
        )

        del dataset[resource_id]
        dataset.metadata = dataset.metadata.remove((resource_id,), recursive=True)

        dataset[nodes_resource_id] = nodes
        dataset[edgelist_resource_id] = edgelist

        dataset.metadata = nodes.metadata.copy_to(dataset.metadata, (), (nodes_resource_id,))
        dataset.metadata = edgelist.metadata.copy_to(dataset.metadata, (), (edgelist_resource_id,))

        dataset.metadata = dataset.metadata.update((), {
            'dimension': {
                'length': len(dataset),
            },
        })

        node_references = self._get_node_references(dataset)

        for column_reference, reference_resource_id in node_references.items():
            if reference_resource_id == resource_id:
                dataset.metadata = dataset.metadata.update(
                    (column_reference.resource_id, metadata_base.ALL_ELEMENTS, column_reference.column_index),
                    {
                        'foreign_key': metadata_base.NO_VALUE,
                    },
                )
                dataset.metadata = dataset.metadata.update(
                    (column_reference.resource_id, metadata_base.ALL_ELEMENTS, column_reference.column_index),
                    {
                        'foreign_key': {
                            'type': 'COLUMN',
                            'resource_id': nodes_resource_id,
                            'column_index': node_id_column_index,
                        },
                    },
                )

    def _set_edges_metadata(
        self, metadata: metadata_base.DataMetadata, source_column_index: int,
        target_column_index: int, *, directed: bool, multi_graph: bool,
        at: metadata_base.Selector,
    ) -> metadata_base.DataMetadata:
        metadata = metadata.add_semantic_type(
            list(at) + [metadata_base.ALL_ELEMENTS, source_column_index],
            'https://metadata.datadrivendiscovery.org/types/EdgeSource',
        )
        metadata = metadata.add_semantic_type(
            list(at) + [metadata_base.ALL_ELEMENTS, target_column_index],
            'https://metadata.datadrivendiscovery.org/types/EdgeTarget',
        )
        metadata = metadata.add_semantic_type(
            list(at) + [metadata_base.ALL_ELEMENTS, source_column_index],
            f'''https://metadata.datadrivendiscovery.org/types/{'Directed' if directed else 'Undirected'}EdgeSource''',
        )
        metadata = metadata.add_semantic_type(
            list(at) + [metadata_base.ALL_ELEMENTS, target_column_index],
            f'''https://metadata.datadrivendiscovery.org/types/{'Directed' if directed else 'Undirected'}EdgeTarget''',
        )
        metadata = metadata.add_semantic_type(
            list(at) + [metadata_base.ALL_ELEMENTS, source_column_index],
            f'''https://metadata.datadrivendiscovery.org/types/{'Multi' if multi_graph else 'Simple'}EdgeSource''',
        )
        metadata = metadata.add_semantic_type(
            list(at) + [metadata_base.ALL_ELEMENTS, target_column_index],
            f'''https://metadata.datadrivendiscovery.org/types/{'Multi' if multi_graph else 'Simple'}EdgeTarget''',
        )

        return metadata

    # TODO: Support also "edge", "nodeAttribute", and "edgeAttribute" references.
    #       See: https://gitlab.datadrivendiscovery.org/MIT-LL/d3m_data_supply/merge_requests/35
    #       See: https://gitlab.datadrivendiscovery.org/MIT-LL/d3m_data_supply/issues/183
    def _get_node_references(self, dataset: container.Dataset) -> typing.Dict[metadata_base.ColumnReference, str]:
        references = {}

        for resource_id, resource in dataset.items():
            if not isinstance(resource, container.DataFrame):
                continue

            for column_index in range(dataset.metadata.query_field((resource_id, metadata_base.ALL_ELEMENTS), 'dimension')['length']):
                column_metadata = dataset.metadata.query_column(column_index, at=(resource_id,))

                column_reference = metadata_base.ColumnReference(resource_id, column_index)

                if 'foreign_key' in column_metadata and column_metadata['foreign_key']['type'] == 'NODE':
                    reference_resource_id = column_metadata['foreign_key']['resource_id']

                    references[column_reference] = reference_resource_id

        return references

    def _update_edge_list(self, dataset: container.Dataset, resource_id: str) -> None:
        # We want to allow this primitive to be run multiple times in a row.
        # So we have to determine if we have already processed this resource.
        if dataset.metadata.list_columns_with_semantic_types([
            'https://metadata.datadrivendiscovery.org/types/EdgeSource',
            'https://metadata.datadrivendiscovery.org/types/EdgeTarget',
        ], at=(resource_id,)):
            return

        dataset.metadata = dataset.metadata.update((resource_id,), {
            'dimension': {
                'name': 'edges',
            },
        })

        reference_column_indices = []
        for column_index in range(dataset.metadata.query_field((resource_id, metadata_base.ALL_ELEMENTS), 'dimension')['length']):
            column_metadata = dataset.metadata.query_column(column_index, at=(resource_id,))

            if 'foreign_key' in column_metadata and column_metadata['foreign_key']['type'] == 'COLUMN':
                reference_column_indices.append(column_index)

        # If there is a different number of columns than it is tricky for us to
        # know which ones belong to edges. We would need some additional metadata
        # in D3M dataset format to handle such case.
        if len(reference_column_indices) != 2:
            raise exceptions.NotSupportedError("Edge list with number of references different than 2 is not supported.")

        source_column_index, target_column_index = reference_column_indices

        # All edge list graphs are undirected.
        # See: https://gitlab.datadrivendiscovery.org/MIT-LL/d3m_data_supply/issues/184
        directed = False
        multi_graph = self._is_multi_graph(dataset[resource_id], source_column_index, target_column_index)

        dataset.metadata = self._set_edges_metadata(
            dataset.metadata,
            source_column_index,
            target_column_index,
            directed=directed,
            multi_graph=multi_graph,
            at=(resource_id,),
        )

    def _is_multi_graph(self, edgelist: container.DataFrame, source_column_index: int, target_column_index: int) -> bool:
        edges = edgelist.iloc[:, [source_column_index, target_column_index]]

        return len(edges) != len(edges.drop_duplicates())


if __name__ == '__main__':
    import logging
    import pprint
    import sys

    logging.basicConfig()

    for dataset_file_path in sys.argv[1:]:
        try:
            dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=os.path.abspath(dataset_file_path)))
        except Exception as error:
            raise Exception("Unable to load dataset: {dataset_doc_path}".format(dataset_doc_path=dataset_file_path)) from error

        primitive = NormalizeGraphsPrimitive(hyperparams=Hyperparams.defaults())

        try:
            normalized_dataset = primitive.produce(inputs=dataset).value

            pprint.pprint(normalized_dataset)
            normalized_dataset.metadata.pretty_print()
        except Exception as error:
            raise Exception("Unable to normalize dataset: {dataset_doc_path}".format(dataset_doc_path=dataset_file_path)) from error
