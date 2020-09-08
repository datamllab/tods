import uuid
import os
import json
import typing

from d3m import container as container_module, exceptions, utils
from d3m.container import dataset as dataset_module


def save_container(container: typing.Any, output_dir: str) -> None:
    # Saving data.
    if isinstance(container, container_module.Dataset):
        dataset_root_metadata = container.metadata.query(())

        missing_metadata: typing.Dict = {}
        for d3m_path, (dataset_path, required) in dataset_module.D3M_TO_DATASET_FIELDS.items():
            if not required:
                continue

            if utils.get_dict_path(dataset_root_metadata, dataset_path) is None:
                # TODO: Use some better value instead of this random value?
                utils.set_dict_path(missing_metadata, dataset_path, str(uuid.uuid4()))

        if missing_metadata:
            container = container.copy()
            container.metadata = container.metadata.update((), missing_metadata)

        # Dataset saver creates any missing directories.
        dataset_uri = 'file://{dataset_path}'.format(dataset_path=os.path.abspath(os.path.join(output_dir, 'datasetDoc.json')))
        container.save(dataset_uri)
    else:
        # We do not want to override anything.
        os.makedirs(output_dir, exist_ok=False)
        dataframe_path = os.path.join(output_dir, 'data.csv')

        if isinstance(container, container_module.DataFrame):
            container.to_csv(dataframe_path)
        elif isinstance(container, (container_module.List, container_module.ndarray)):
            container = container_module.DataFrame(container)
            container.to_csv(dataframe_path)
        else:
            raise exceptions.NotSupportedError("Value with type '{value_type}' cannot be saved as a container type.".format(value_type=type(container)))

    # Saving metadata. This is just for debugging purposes, so we are
    # using "to_json_structure" and not "to_internal_json_structure".
    input_metadata = container.metadata.to_json_structure()
    metadata_path = os.path.join(output_dir, 'metadata.json')

    with open(metadata_path, 'w') as outfile:
        json.dump(input_metadata, outfile, indent=2, sort_keys=True, allow_nan=False)
