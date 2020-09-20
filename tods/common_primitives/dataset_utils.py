import collections
import typing

from d3m import container


def sample_rows(
    dataset: container.Dataset, main_resource_id: str, main_resource_indices_to_keep: typing.Set[int],
    relations_graph: typing.Dict[str, typing.List[typing.Tuple[str, bool, int, int, typing.Dict]]], *,
    delete_recursive: bool = False,
) -> container.Dataset:
    # We store rows as sets, but later on we sort them when we select rows.
    row_indices_to_keep_sets: typing.Dict[str, typing.Set[int]] = collections.defaultdict(set)
    row_indices_to_keep_sets[main_resource_id] = main_resource_indices_to_keep

    # If "delete_recursive" is set to "False", we do not populate "row_indices_to_keep_sets"
    # with other resources, making "select_rows" simply keep them.
    if delete_recursive:
        # We sort to be deterministic.
        for main_resource_row_index in sorted(row_indices_to_keep_sets[main_resource_id]):
            queue = []
            queue.append((main_resource_id, [main_resource_row_index]))
            while queue:
                current_resource_id, current_row_indices = queue.pop(0)
                current_resource = dataset[current_resource_id]

                for edge_resource_id, edge_direction, edge_from_index, edge_to_index, custom_state in relations_graph[current_resource_id]:
                    # All rows from the main resource we want are already there.
                    # TODO: What to do if we get a reference to the row in the main resource which is not part of this sample?
                    #       This means that probably the sample is invalid. We should not be generating such samples which do not
                    #       preserve reference loops and their consistency. Otherwise it is not really possible to denormalize
                    #       such Dataset properly: a reference is referencing a row in the main resource which does not exist.
                    if edge_resource_id == main_resource_id:
                        continue

                    edge_resource = dataset[edge_resource_id]

                    to_column_values = edge_resource.iloc[:, edge_to_index]
                    for from_column_value in current_resource.iloc[current_row_indices, edge_from_index]:
                        # We assume here that "index" corresponds to the default index with row indices.
                        rows_with_value = edge_resource.index[to_column_values == from_column_value]
                        # We sort to be deterministic.
                        new_rows_list = sorted(set(rows_with_value) - row_indices_to_keep_sets[edge_resource_id])
                        row_indices_to_keep_sets[edge_resource_id].update(new_rows_list)
                        queue.append((edge_resource_id, new_rows_list))

    # We sort indices to get deterministic outputs from sets (which do not have deterministic order).
    # We also do not want to change the row order but keep the original row order.
    # Sorting by row indices values assure that.
    row_indices_to_keep = {resource_id: sorted(indices) for resource_id, indices in row_indices_to_keep_sets.items()}

    return dataset.select_rows(row_indices_to_keep)
