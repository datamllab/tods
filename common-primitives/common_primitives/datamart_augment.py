import os

import datamart  # type: ignore

from d3m import container
from d3m import utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

import common_primitives

__all__ = ('DataMartAugmentPrimitive',)

Inputs = container.Dataset
Outputs = container.Dataset


class ColumnHyperparams(hyperparams.Hyperparams, set_names=False):
    resource_id = hyperparams.Hyperparameter[str]('')
    column_index = hyperparams.Hyperparameter[int](-1)


class Hyperparams(hyperparams.Hyperparams):
    search_result = hyperparams.Hyperparameter[str](
        default='',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        ],
        description="Serialized search result provided by Datamart",
    )
    system_identifier = hyperparams.Hyperparameter[str](
        default='',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/ControlParameter',
        ],
        description="Which Datamart system this search result is from",
    )
    augment_columns = hyperparams.Set(
        elements=ColumnHyperparams,
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Optional list of columns from the Datamart dataset that will be added"
    )


class DataMartAugmentPrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Augment supplied dataset with additional columns.

    Use ``DATAMART_NYU_URL`` and ``DATAMART_ISI_URL`` environment variables to control where
    can the primitive connect to respective DataMarts.
    """

    metadata = metadata_base.PrimitiveMetadata({
        'id': 'fe0f1ac8-1d39-463a-b344-7bd498a31b91',
        'version': '0.1',
        'name': "Perform dataset augmentation using Datamart",
        'python_path': 'd3m.primitives.data_augmentation.datamart_augmentation.Common',
        'source': {
            'name': common_primitives.__author__,
            'contact': 'mailto:remi.rampin@nyu.edu',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/datamart_augment.py',
                'https://gitlab.com/datadrivendiscovery/common-primitives.git',
            ],
        },
        'installation': [{
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://gitlab.com/datadrivendiscovery/common-primitives.git@{git_commit}#egg=common_primitives'.format(
                git_commit=d3m_utils.current_git_commit(
                    os.path.dirname(__file__)),
            ),
        }],
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.DATA_RETRIEVAL,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_AUGMENTATION,
        'pure_primitive': False,
    })

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        search_result = self.hyperparams['search_result']
        system_identifier = self.hyperparams['system_identifier']
        augment_columns = [datamart.DatasetColumn(**augment_column) for augment_column in self.hyperparams['augment_columns']]
        if not augment_columns:
            augment_columns = None

        # Get the URL for this system from the environment (can be None)
        system_url = os.environ.get('DATAMART_URL_{}'.format(system_identifier))

        # Deserialize search result
        if system_identifier == 'NYU':
            import datamart_rest  # type: ignore

            search_result_loaded = datamart_rest.RESTSearchResult.deserialize(search_result)
        elif system_identifier == 'ISI':
            import datamart_isi.rest  # type: ignore

            search_result_loaded = datamart_isi.rest.RESTSearchResult.deserialize(search_result)
        else:
            raise ValueError("Unknown Datamart system {}".format(system_identifier))

        # Perform augment
        output = search_result_loaded.augment(supplied_data=inputs, augment_columns=augment_columns, connection_url=system_url)
        return CallResult(output)
