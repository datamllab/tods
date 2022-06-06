import os
import typing
import itertools

import numpy
import pandas

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer


__all__ = ('DenormalizePrimitive',)

Inputs = container.Dataset
Outputs = container.Dataset


class Hyperparams(hyperparams.Hyperparams):
    starting_resource = hyperparams.Hyperparameter[typing.Union[str, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="From which resource to start denormalizing. If \"None\" then it starts from the dataset entry point.",
    )
    recursive = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Denormalize recursively?",
    )
    many_to_many = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Denormalize also many-to-many relations?",
    )
    discard_not_joined_tabular_resources = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should tabular resources which were not joined be discarded?",
    )


# TODO: Implement support for M2M relations.
# TODO: Consider the case where there are loops in foreign keys.
# TODO: Add all column names together to "other names" metadata for column.
# TODO: Consider denormalizing deep-first instead of current iterative approach.
#       It seems it might be better because when one table is referencing the second one twice,
#       which might reference other tables further, then currently we first join the second table
#       and then have to repeat joining other tables twice. But we could first join other tables
#       once to the second table, and then just do the join with already joined second table.
#       Not sure how to behave in "recursive == False" case then.
# TODO: Add a test where main table has a foreign key twice to same table (for example, person 1 and person 2 to table of persons).
class DenormalizePrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):   # pragma: no cover
    """
    A primitive which converts a Dataset with multiple tabular resources into a Dataset with only one tabular resource,
    based on known relations between tabular resources. Any resource which can be joined is joined (thus the resource
    itself is removed), and other resources are by default discarded (controlled by ``discard_resources`` hyper-parameter).

    If hyper-parameter ``recursive`` is set to ``True``, the primitive will join tables recursively. For example,
    if table 1 (main table) has a foreign key that points to table 2, and table 2 has a foreign key that points to table 3,
    then after table 2 is jointed into table 1, table 1 will have a foreign key that points to table 3. So now the
    primitive continues to join table 3 into the main table.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'f31f8c1f-d1c5-43e5-a4b2-2ae4a761ef2e',
            'version': '0.2.0',
            'name': "Denormalize datasets",
            'python_path': 'd3m.primitives.tods.common.denormalize',
            'source': {
                'name': "DATALab@Texas A&M University",
                'contact': 'mailto:khlai037@tamu.edu',
            },
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_DENORMALIZATION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        # If only one tabular resource is in the dataset, we do not have anything to do.
        tabular_resource_ids = [dataset_resource_id for dataset_resource_id, dataset_resource in inputs.items() if isinstance(dataset_resource, container.DataFrame)]
        if len(tabular_resource_ids) == 1:
            return base.CallResult(inputs)

        # We could set "pick_one" to "False" because we already checked for that, but we leave it
        # as "True" because then error messages are more meaningful for this case.
        main_resource_id, main_resource = base_utils.get_tabular_resource(inputs, self.hyperparams['starting_resource'])

        # Graph is the adjacency representation for the relations graph.
        graph = inputs.get_relations_graph()

        resources = dict(inputs)
        metadata = inputs.metadata
        all_resources_joined = set()

        while self._has_forward_edges(graph, main_resource_id):
            # "resources" and "graph" are modified in-place.
            metadata, resources_joined = self._denormalize(resources, metadata, main_resource_id, graph)

            all_resources_joined.update(resources_joined)

            if not self.hyperparams['recursive']:
                break

        # Do we discard all other tabular resources (including joined ones)?
        if self.hyperparams['discard_not_joined_tabular_resources']:
            resources_to_remove = []
            for resource_id, resource in resources.items():
                if resource_id == main_resource_id:
                    continue
                if not isinstance(resource, container.DataFrame):
                    continue
                resources_to_remove.append(resource_id)

        # Discard only joined tabular resources and which no other resource depends on.
        else:
            # We deal only with tabular resources here.
            dependent_upon_resources = self._get_dependent_upon_resources(graph)
            resources_to_remove = [resource_id for resource_id in sorted(all_resources_joined - dependent_upon_resources) if resource_id != main_resource_id]

        for resource_id in resources_to_remove:
            assert resource_id != main_resource_id

            del resources[resource_id]
            metadata = metadata.remove((resource_id,), recursive=True)

        metadata = metadata.update((), {
            'dimension': {
                'length': len(resources),
            },
        })

        return base.CallResult(container.Dataset(resources, metadata))

    def _has_forward_edges(self, graph: typing.Dict[str, typing.List[typing.Tuple[str, bool, int, int, typing.Dict]]], resource_id: str) -> bool:
        # We check first to not create a list in "graph" when accessing it.
        if resource_id not in graph:
            return False

        for edge_resource_id, edge_direction, edge_from_index, edge_to_index, custom_state in graph[resource_id]:
            if edge_direction:
                return True

        return False

    def _has_edges_to_process(self, graph: typing.Dict[str, typing.List[typing.Tuple[str, bool, int, int, typing.Dict]]], resource_id: str) -> bool:
        # We check first to not create a list in "graph" when accessing it.
        if resource_id not in graph:
            return False

        for edge_resource_id, edge_direction, edge_from_index, edge_to_index, custom_state in graph[resource_id]:
            if custom_state.get('process', False):
                return True

        return False

    def _denormalize(self, resources: typing.Dict, metadata: metadata_base.DataMetadata, main_resource_id: str,
                     graph: typing.Dict[str, typing.List[typing.Tuple[str, bool, int, int, typing.Dict]]]) -> typing.Tuple[metadata_base.DataMetadata, typing.Set[str]]:
        """
        Finds all tables which are pointed to by the main resource and join them into the main table.

        ``resources`` and ``graph`` are modified in-place.
        """

        resources_joined: typing.Set[str] = set()
        main_resource = resources[main_resource_id]

        # Should not really happen.
        if main_resource_id not in graph:
            return metadata, resources_joined

        # We mark all current edges to be processed. We might be adding more edges to the list,
        # but we want to do for this call only those which existed at the beginning.
        for edge_resource_id, edge_direction, edge_from_index, edge_to_index, custom_state in graph[main_resource_id]:
            custom_state['process'] = True

        while self._has_edges_to_process(graph, main_resource_id):
            edge_resource_id, edge_direction, edge_from_index, edge_to_index, custom_state = graph[main_resource_id][0]

            if not custom_state.get('process', False):
                continue
            del custom_state['process']

            if not edge_direction:
                # For now we just remove this relation.
                # TODO: Support M2M relations.

                # We remove the relation we would have joined, backward.
                self._remove_graph_edge(graph, main_resource_id, edge_resource_id, False, edge_from_index, edge_to_index)

                # We remove the relation we would have joined, forward.
                self._remove_graph_edge(graph, edge_resource_id, main_resource_id, True, edge_to_index, edge_from_index)

                continue

            if main_resource_id == edge_resource_id:
                # TODO: Implement.
                raise NotImplementedError("Support for loops is not implemented yet.")

            # Calling "_join" updates column indices in "graph" and "metadata"
            # and also removes the current joined edge from "graph"
            main_resource, metadata = self._join(
                main_resource_id, main_resource, edge_from_index,
                edge_resource_id, resources[edge_resource_id], edge_to_index,
                metadata, graph,
            )

            resources_joined.add(edge_resource_id)

        resources[main_resource_id] = main_resource

        return metadata, resources_joined

    def _row_of_missing_values(self, resource: container.DataFrame, metadata: metadata_base.DataMetadata, resource_id: str) -> typing.List[typing.Any]:
        row = []
        for column_index, dtype in enumerate(resource.dtypes):
            if dtype.kind in ['b', 'i', 'u', 'f', 'c']:
                row.append(numpy.nan)
            elif dtype.kind == 'O' and issubclass(metadata.query_column_field(column_index, 'structural_type', at=(resource_id,)), str):
                row.append('')
            else:
                row.append(None)

        return row

    def _join(self, main_resource_id: str, main_resource: container.DataFrame, main_column_index: int, foreign_resource_id: str,
              foreign_resource: container.DataFrame, foreign_column_index: int, metadata: metadata_base.DataMetadata,
              graph: typing.Dict[str, typing.List[typing.Tuple[str, bool, int, int, typing.Dict]]]) -> typing.Tuple[container.DataFrame, metadata_base.DataMetadata]:
        if main_resource_id == foreign_resource_id:
            # TODO: Implement.
            raise NotImplementedError("Support for loops is not implemented yet.")

        # We use this information later on.
        one_to_one_relation = foreign_resource.iloc[:, foreign_column_index].sort_values().equals(main_resource.iloc[:, main_column_index].sort_values())

        foreign_indexer = pandas.Index(foreign_resource.iloc[:, foreign_column_index]).get_indexer(main_resource.iloc[:, main_column_index])
        # "get_indexer" sets all unresolved values to -1.
        unresolved_rows = foreign_indexer == -1

        # We store dtypes so that we can later on compare.
        foreign_resource_dtypes = foreign_resource.dtypes

        # -1 is converted into the last row, but we set it to row of missing values if it exists.
        resolved_foreign_resource = foreign_resource.take(foreign_indexer).reset_index(drop=True)
        if unresolved_rows.any():
            # Set all unresolved rows to a row of missing values.
            resolved_foreign_resource.iloc[unresolved_rows, :] = self._row_of_missing_values(foreign_resource, metadata, foreign_resource_id)

        # And store final dtypes so that we can later on compare.
        resolved_foreign_resource_dtypes = resolved_foreign_resource.dtypes

        # This makes a copy so that we can modify metadata in-place.
        metadata = metadata.update(
            (metadata_base.ALL_ELEMENTS,),
            {},
        )

        # TODO: Move this to metadata API.
        # We reorder metadata for rows.
        for element_metadata_entry in [
            metadata._current_metadata.all_elements,
            metadata._current_metadata.elements[foreign_resource_id],
        ]:
            if element_metadata_entry is None:
                continue

            elements = element_metadata_entry.elements
            new_elements_evolver = d3m_utils.EMPTY_PMAP.evolver()
            for i, row_index in enumerate(foreign_indexer):
                if row_index == -1:
                    continue

                if row_index in elements:
                    new_elements_evolver.set(i, elements[row_index])
            element_metadata_entry.elements = new_elements_evolver.persistent()
            element_metadata_entry.is_elements_empty = not element_metadata_entry.elements
            element_metadata_entry.update_is_empty()

        assert resolved_foreign_resource.shape[1] > 0

        main_resource = pandas.concat([
            main_resource.iloc[:, 0:main_column_index],
            resolved_foreign_resource,
            main_resource.iloc[:, main_column_index + 1:],
        ], axis=1)

        old_semantic_types = metadata.query_column(main_column_index, at=(main_resource_id,)).get('semantic_types', [])

        # First we remove metadata for the existing column.
        # This makes a copy so that we can modify metadata in-place.
        metadata = metadata.remove_column(main_column_index, at=(main_resource_id,), recursive=True)

        # TODO: Move this to metadata API.
        # Move columns and make space for foreign metadata to be inserted.
        # We iterate over a list so that we can change dict while iterating.
        for element_metadata_entry in itertools.chain(
            [metadata._current_metadata.all_elements.all_elements if metadata._current_metadata.all_elements is not None else None],
            metadata._current_metadata.all_elements.elements.values() if metadata._current_metadata.all_elements is not None else iter([None]),
            [metadata._current_metadata.elements[main_resource_id].all_elements],
            metadata._current_metadata.elements[main_resource_id].elements.values(),
        ):
            if element_metadata_entry is None:
                continue

            new_elements_evolver = element_metadata_entry.elements.evolver()
            for element_index in element_metadata_entry.elements.keys(reverse=True):
                # We removed metadata for "main_column_index".
                assert element_index != main_column_index

                element_index = typing.cast(int, element_index)

                if main_column_index < element_index:
                    metadata_dict = new_elements_evolver[element_index]
                    new_elements_evolver.remove(element_index)
                    new_elements_evolver.set(element_index + resolved_foreign_resource.shape[1] - 1, metadata_dict)
            element_metadata_entry.elements = new_elements_evolver.persistent()
            element_metadata_entry.is_elements_empty = not element_metadata_entry.elements
            element_metadata_entry.update_is_empty()

        # And copy over metadata for new (replaced) columns in place of the existing column.
        for column_index in range(resolved_foreign_resource.shape[1]):
            # To go over "ALL_ELEMENTS" and all rows.
            for element in metadata.get_elements((foreign_resource_id,)):
                metadata = metadata.copy_to(
                    metadata,
                    [foreign_resource_id, element, metadata_base.ALL_ELEMENTS],
                    [main_resource_id, element, main_column_index + column_index],
                    ignore_all_elements=True,
                )
                metadata = metadata.copy_to(
                    metadata,
                    [foreign_resource_id, element, column_index],
                    [main_resource_id, element, main_column_index + column_index],
                    ignore_all_elements=True,
                )

        # Update metadata for new (replaced) columns.
        for column_index in range(main_column_index, main_column_index + resolved_foreign_resource.shape[1]):
            # We copy semantic types describing the role of the column from the original column to all new (replaced) columns.
            # TODO: Do not hard-code this list here but maybe extract it from "definitions.json"?
            for semantic_type in [
                'https://metadata.datadrivendiscovery.org/types/Attribute',
                'https://metadata.datadrivendiscovery.org/types/Boundary',
                'https://metadata.datadrivendiscovery.org/types/BoundingPolygon',
                'https://metadata.datadrivendiscovery.org/types/Interval',
                'https://metadata.datadrivendiscovery.org/types/IntervalEnd',
                'https://metadata.datadrivendiscovery.org/types/IntervalStart',
                'https://metadata.datadrivendiscovery.org/types/InstanceWeight',
                'https://metadata.datadrivendiscovery.org/types/PrivilegedData',
                'https://metadata.datadrivendiscovery.org/types/RedactedPrivilegedData',
                'https://metadata.datadrivendiscovery.org/types/RedactedTarget',
                'https://metadata.datadrivendiscovery.org/types/SuggestedPrivilegedData',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                'https://metadata.datadrivendiscovery.org/types/Target',
                'https://metadata.datadrivendiscovery.org/types/PredictedTarget',
                'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                'https://metadata.datadrivendiscovery.org/types/Score',
                'https://metadata.datadrivendiscovery.org/types/Confidence',
                'https://metadata.datadrivendiscovery.org/types/Time',
                'https://metadata.datadrivendiscovery.org/types/Location',
            ]:
                if semantic_type in old_semantic_types:
                    metadata = metadata.add_semantic_type((main_resource_id, metadata_base.ALL_ELEMENTS, column_index), semantic_type)

            is_column_unique = main_resource.iloc[:, column_index].is_unique
            column_semantic_types = metadata.query_column(column_index, at=(main_resource_id,)).get('semantic_types', [])
            was_column_unique = 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in column_semantic_types \
                or 'https://metadata.datadrivendiscovery.org/types/UniqueKey' in column_semantic_types

            # Foreign keys can reference same foreign row multiple times, so values in this column might not be even
            # unique anymore, nor they are a primary key at all. So we remove the semantic type marking a column as such.
            # We re-set semantic type for any real primary key later on.
            metadata = metadata.remove_semantic_type((main_resource_id, metadata_base.ALL_ELEMENTS, column_index), 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
            metadata = metadata.remove_semantic_type((main_resource_id, metadata_base.ALL_ELEMENTS, column_index), 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey')
            metadata = metadata.remove_semantic_type((main_resource_id, metadata_base.ALL_ELEMENTS, column_index), 'https://metadata.datadrivendiscovery.org/types/UniqueKey')

            # We re-set semantic type for column which was and is still unique.
            if was_column_unique and is_column_unique:
                metadata = metadata.add_semantic_type((main_resource_id, metadata_base.ALL_ELEMENTS, column_index), 'https://metadata.datadrivendiscovery.org/types/UniqueKey')

            old_dtype = foreign_resource_dtypes.iloc[column_index - main_column_index]
            new_dtype = resolved_foreign_resource_dtypes.iloc[column_index - main_column_index]
            if old_dtype is not new_dtype:
                # Not a nice way to convert a dtype to Python type, but it works.
                old_type = type(numpy.zeros(1, old_dtype).tolist()[0])
                new_type = type(numpy.zeros(1, new_dtype).tolist()[0])
                if old_type is not new_type:
                    # Type changed, we have to update metadata about the structural type.
                    metadata = metadata.update((main_resource_id, metadata_base.ALL_ELEMENTS, column_index), {
                        'structural_type': new_type,
                    })

        # If the original column was a primary key, we should re-set it back.
        if 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in old_semantic_types and (one_to_one_relation or not unresolved_rows.any()):
            if main_resource.iloc[:, main_column_index].is_unique:
                metadata = metadata.add_semantic_type((main_resource_id, metadata_base.ALL_ELEMENTS, main_column_index), 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
                # Removing "UniqueKey" if it was set before, "PrimaryKey" surpasses it.
                metadata = metadata.remove_semantic_type((main_resource_id, metadata_base.ALL_ELEMENTS, main_column_index), 'https://metadata.datadrivendiscovery.org/types/UniqueKey')
            else:
                metadata = metadata.add_semantic_type((main_resource_id, metadata_base.ALL_ELEMENTS, main_column_index), 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey')
        elif 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey' in old_semantic_types and (one_to_one_relation or not unresolved_rows.any()):
            metadata = metadata.add_semantic_type((main_resource_id, metadata_base.ALL_ELEMENTS, main_column_index), 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey')

        # TODO: Update boundary columns and "confidence for" references.
        #       This is not currently needed because all file collections are just one column so they do not
        #       move the column indices. But as a general case we should be updating all standard column references.

        # Update columns number in the main resource.
        metadata = metadata.update((main_resource_id, metadata_base.ALL_ELEMENTS), {
            'dimension': {
                'length': main_resource.shape[1],
            },
        })

        # We remove the relation we just joined, forward.
        self._remove_graph_edge(graph, main_resource_id, foreign_resource_id, True, main_column_index, foreign_column_index)

        # We remove the relation we just joined, backward.
        self._remove_graph_edge(graph, foreign_resource_id, main_resource_id, False, foreign_column_index, main_column_index)

        # We have to update column indices if they have changed because we inserted new columns.
        for resource_id, edges in graph.items():
            if resource_id == main_resource_id:
                for i, (edge_resource_id, edge_direction, edge_from_index, edge_to_index, custom_state) in enumerate(edges):
                    if edge_direction and main_column_index < edge_from_index:
                        # We replaced one column with "resolved_foreign_resource.shape[1]" columns, so there is
                        # "resolved_foreign_resource.shape[1] - 1" new columns to shift indices for.
                        edges[i] = (edge_resource_id, edge_direction, edge_from_index + resolved_foreign_resource.shape[1] - 1, edge_to_index, custom_state)
            else:
                for i, (edge_resource_id, edge_direction, edge_from_index, edge_to_index, custom_state) in enumerate(edges):
                    if edge_resource_id == main_resource_id and not edge_direction and main_column_index < edge_to_index:
                        # We replaced one column with "resolved_foreign_resource.shape[1]" columns, so there is
                        # "resolved_foreign_resource.shape[1] - 1" new columns to shift indices for.
                        edges[i] = (edge_resource_id, edge_direction, edge_from_index, edge_to_index + resolved_foreign_resource.shape[1] - 1, custom_state)

        # If foreign resource has any additional relations, we copy them to new columns in the main resource.
        if foreign_resource_id in graph:
            # We iterate over a list so that we can change graph while iterating.
            for edge_resource_id, edge_direction, edge_from_index, edge_to_index, custom_state in list(graph[foreign_resource_id]):
                if edge_resource_id in [main_resource_id, foreign_resource_id]:
                    # TODO: Implement.
                    raise NotImplementedError("Support for loops is not implemented yet.")

                if edge_direction:
                    graph[main_resource_id].append((edge_resource_id, True, main_column_index + edge_from_index, edge_to_index, {}))
                    graph[edge_resource_id].append((main_resource_id, False, edge_to_index, main_column_index + edge_from_index, {}))
                else:
                    # TODO: What should we do about backward relations?
                    #       For now we just ignore backward relations because we do not support M2M relations.
                    #       For the foreign resource we just joined, we could change all relations to instead point
                    #       to the main resource. This might be tricky though if we have a situation where main table
                    #       includes table 1 twice, and table 1 has a relation to table 2. If we after joining table 1
                    #       once rewrite all backward relations from table 2 to table 1 to point to main table now,
                    #       when we get to join the table 1 the second time we might have issues. This is why it might
                    #       better to start joining deep-first. See another TODO.
                    # TODO: We might have to also update foreign key metadata in this case.
                    #       We might want to update metadata so that if table 1 is joined to the main table, and there is
                    #       also table 2 which has a foreign key that points to table 1, then the foreign key in table 2
                    #       should point to the main table after joining. But what if main table has a foreign key to
                    #       table 1 twice? How do we then update metadata in table 2 to point twice to table 1?
                    #       Metadata does not support that.

                    # A special case for now. If relation is one-to-one, then we can move backwards relations to the
                    # main resource without complications mentioned in TODOs above. Maybe some additional columns might
                    # be joined through M2M relations in this case, once that is supported, but generally this should not
                    # be a problem. It might add some duplicated columns at that point. This special case is useful
                    # when "learningData" with only targets is pointing to some other table with real attributes.
                    if one_to_one_relation:
                        self._remove_graph_edge(graph, edge_resource_id, foreign_resource_id, True, edge_to_index, edge_from_index)
                        self._remove_graph_edge(graph, foreign_resource_id, edge_resource_id, False, edge_from_index, edge_to_index)

                        graph[main_resource_id].append((edge_resource_id, False, main_column_index + edge_from_index, edge_to_index, custom_state))
                        graph[edge_resource_id].append((main_resource_id, True, edge_to_index, main_column_index + edge_from_index, custom_state))

                        # We override metadata for foreign key to make it point to the main resource (and not to foreign resource anymore).
                        metadata = metadata.update((edge_resource_id, metadata_base.ALL_ELEMENTS, edge_to_index), {
                            'foreign_key': {
                                'type': 'COLUMN',
                                'resource_id': main_resource_id,
                                'column_index': main_column_index + edge_from_index,
                                'column_name': metadata_base.NO_VALUE,
                            },
                        })

        return main_resource, metadata

    def _get_dependent_upon_resources(self, graph: typing.Dict[str, typing.List[typing.Tuple[str, bool, int, int, typing.Dict]]]) -> typing.Set[str]:
        """
        Returns a set of resources which have other resources depend on them.
        """

        dependent_upon_resources = set()

        for resource_id, edges in graph.items():
            for edge_resource_id, edge_direction, edge_from_index, edge_to_index, custom_state in edges:
                if edge_direction:
                    dependent_upon_resources.add(edge_resource_id)

        return dependent_upon_resources

    def _remove_graph_edge(self, graph: typing.Dict[str, typing.List[typing.Tuple[str, bool, int, int, typing.Dict]]],
                           resource_id: str, edge_resource_id: str, edge_direction: bool, edge_from_index: int, edge_to_index: int) -> None:
        assert resource_id in graph

        for i, edge in enumerate(graph[resource_id]):
            if edge[0:4] == (edge_resource_id, edge_direction, edge_from_index, edge_to_index):
                del graph[resource_id][i]
                break

        if not graph[resource_id]:
            del graph[resource_id]


# if __name__ == '__main__':
#     import logging
#     import pprint
#     import sys

#     logging.basicConfig()

#     for dataset_file_path in sys.argv[1:]:
#         try:
#             dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=os.path.abspath(dataset_file_path)))
#         except Exception as error:
#             raise Exception("Unable to load dataset: {dataset_doc_path}".format(dataset_doc_path=dataset_file_path)) from error

#         primitive = DenormalizePrimitive(hyperparams=Hyperparams.defaults().replace({
#             'recursive': True,
#             'discard_not_joined_tabular_resources': False,
#         }))

#         try:
#             denormalized_dataset = primitive.produce(inputs=dataset).value

#             pprint.pprint(denormalized_dataset)
#             denormalized_dataset.metadata.pretty_print()
#         except Exception as error:
#             raise Exception("Unable to denormalize dataset: {dataset_doc_path}".format(dataset_doc_path=dataset_file_path)) from error
