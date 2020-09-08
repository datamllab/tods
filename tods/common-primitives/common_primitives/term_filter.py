import os
import re

from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives
from common_primitives import dataframe_utils

__all__ = ('TermFilterPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    column = hyperparams.Hyperparameter[int](
        default=-1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Index of column filter applies to.',
    )
    inclusive = hyperparams.Hyperparameter[bool](
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='True when values that contain a match against the term list are retained, False when they are removed.',
    )
    terms = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='A set of terms to filter against. A row will be filtered if any term in the list matches.',
    )
    match_whole = hyperparams.Hyperparameter[bool](
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='True if a term is matched only against a full word, False if a word need only contain the term.',
    )


class TermFilterPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which filters rows from a DataFrame based on a column value containing a match
    against a caller supplied term list.  Supports search-style matching where the target need only
    contain a term, as well as whole word matching where the target is tokenized using regex word boundaries.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'a6b27300-4625-41a9-9e91-b4338bfc219b',
            'version': '0.1.0',
            'name': "Term list dataset filter",
            'python_path': 'd3m.primitives.data_preprocessing.term_filter.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/term_filter.py',
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
                metadata_base.PrimitiveAlgorithmType.ARRAY_SLICING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
        },
    )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        # to make sure index matches row indices
        resource = inputs.reset_index(drop=True)

        try:
            escaped_terms = [re.escape(t) for t in self.hyperparams['terms']]

            if self.hyperparams['match_whole']:
                # convert term list into a regex that matches whole words
                pattern = re.compile(r'\b(?:%s)\b' % '|'.join(escaped_terms))
            else:
                # convert term list into a regex that does a partial match
                pattern = re.compile('|'.join(escaped_terms))

            matched = resource.iloc[:, self.hyperparams['column']].astype(str).str.contains(pattern)
            to_keep = matched if self.hyperparams['inclusive'] else ~matched

            to_keep_indices = resource.loc[to_keep].index

        except re.error as error:
            raise exceptions.InvalidArgumentValueError("Failed to compile regex for terms: {terms}".format(terms=self.hyperparams['terms'])) from error

        # remove dataframe and metadata rows by index
        outputs = dataframe_utils.select_rows(inputs, to_keep_indices)

        return base.CallResult(outputs)
