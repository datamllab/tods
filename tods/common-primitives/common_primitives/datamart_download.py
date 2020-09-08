import os

from d3m import container
from d3m import utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

import common_primitives

__all__ = ('DataMartDownloadPrimitive',)


Inputs = container.Dataset
Outputs = container.Dataset


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


class DataMartDownloadPrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Download a dataset from DataMart.

    Use ``DATAMART_NYU_URL`` and ``DATAMART_ISI_URL`` environment variables to control where
    can the primitive connect to respective DataMarts.
    """

    metadata = metadata_base.PrimitiveMetadata({
        'id': '9e2077eb-3e38-4df1-99a5-5e647d21331f',
        'version': '0.1',
        'name': "Download a dataset from Datamart",
        'python_path': 'd3m.primitives.data_augmentation.datamart_download.Common',
        'source': {
            'name': common_primitives.__author__,
            'contact': 'mailto:remi.rampin@nyu.edu',
            'uris': [
                'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/datamart_download.py',
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

        # Perform download
        output = search_result_loaded.download(supplied_data=inputs, connection_url=system_url)
        return CallResult(output)
