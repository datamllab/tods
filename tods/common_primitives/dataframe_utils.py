import typing

from d3m import exceptions, utils
from d3m.container import pandas as container_pandas


def select_rows(resource: container_pandas.DataFrame, row_indices_to_keep: typing.Sequence[int]) -> container_pandas.DataFrame:
    if not isinstance(resource, container_pandas.DataFrame):
        raise exceptions.InvalidArgumentTypeError("Only DataFrame resources can have rows selected, not '{type}'.".format(type=type(resource)))

    row_indices = sorted(row_indices_to_keep)
    resource = resource.iloc[row_indices, :].reset_index(drop=True)

    # TODO: Expose this as a general metadata method.
    #       In that case this has to be done recursively over all nested ALL_ELEMENTS.
    #       Here we are operating at resource level so we have to iterate only over first
    #       ALL_ELEMENTS and resource's element itself.

    # Change the metadata. Update the number of rows in the split.
    # This makes a copy so that we can modify metadata in-place.
    resource.metadata = resource.metadata.update(
        (),
        {
            'dimension': {
                'length': len(row_indices),
            },
        },
    )

    # Remove all rows not in this split and reorder those which are.
    for element_metadata_entry in [
        resource.metadata._current_metadata,
    ]:
        if element_metadata_entry is None:
            continue

        elements = element_metadata_entry.elements
        new_elements_evolver = utils.EMPTY_PMAP.evolver()
        for i, row_index in enumerate(row_indices):
            if row_index in elements:
                new_elements_evolver.set(i, elements[row_index])
        element_metadata_entry.elements = new_elements_evolver.persistent()
        element_metadata_entry.is_elements_empty = not element_metadata_entry.elements
        element_metadata_entry.update_is_empty()

    return resource
