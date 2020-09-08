import collections
import copy
import os.path
import typing

from d3m import container, exceptions, index, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, params
from d3m.primitive_interfaces import base, transformer, unsupervised_learning

import common_primitives


Inputs = container.Dataset
Outputs = container.Dataset


class Params(params.Params):
    # For resource in a dataset we have potentially params of a primitive.
    # Or we have one for all resources if "continue_fit" is enabled.
    # TODO: Remove workaround of "Any" once resolved in pytypes.
    #       See: https://github.com/Stewori/pytypes/issues/56
    #       Restore to: resource_params: typing.Optional[typing.Union[typing.Dict[str, params.Params], params.Params]]
    resource_params: typing.Optional[typing.Any]


class Hyperparams(hyperparams_module.Hyperparams):
    # TODO: How to specify that input type of allowed primitive has to be "DataFrame".
    #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/335
    primitive = hyperparams_module.Union[typing.Union[transformer.TransformerPrimitiveBase, unsupervised_learning.UnsupervisedLearnerPrimitiveBase]](
        configuration=collections.OrderedDict(
            transformer=hyperparams_module.Primitive[transformer.TransformerPrimitiveBase](  # type: ignore
                # TODO: This default in fact gets List as input and produces List. Not DataFrame.
                #       But in fact it just passes through whatever it gets, so it works out.
                #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/214
                default=index.get_primitive('d3m.primitives.operator.null.TransformerTest'),
                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
                description="A transformer primitive.",
            ),
            unsupervised_learner=hyperparams_module.Primitive[unsupervised_learning.UnsupervisedLearnerPrimitiveBase](  # type: ignore
                # TODO: This default in fact gets List as input and produces List. Not DataFrame.
                #       But in fact it just passes through whatever it gets, so it works out.
                #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/214
                default=index.get_primitive('d3m.primitives.operator.null.UnsupervisedLearnerTest'),
                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
                description="An unsupervised learner primitive. If it is already fitted and you do not want to re-fit it, "
                            "set \"fit_primitive\" to \"no\".",
            ),
        ),
        default='transformer',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A primitive to use for mapping of \"DataFrame\" resources. Has to take \"DataFrame\" as input.",
    )
    fit_primitive = hyperparams_module.Enumeration(
        values=['no', 'fit', 'continue_fit'],
        default='fit',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Fit an unsupervised learner primitive or not.",
    )
    produce_method = hyperparams_module.Hyperparameter[str](
        default='produce',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Name of primitive's produce method to use.",
    )
    resources = hyperparams_module.Union[typing.Union[typing.Sequence[str], str]](
        configuration=collections.OrderedDict(
            resource_ids=hyperparams_module.Set(
                elements=hyperparams_module.Hyperparameter[str](
                    # Default is ignored.
                    # TODO: Remove default. See: https://gitlab.com/datadrivendiscovery/d3m/issues/141
                    default='',
                    semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
                    description="Resource ID to map.",
                ),
                default=(),
                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
                description="Map resources matching specified resource IDs.",
            ),
            all=hyperparams_module.Constant(
                default="all",
                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
                description="Map all dataset resources.",
            ),
            entry_point=hyperparams_module.Constant(
                default='entry_point',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
                description="Map the dataset entry point, if dataset has one, "
                            "or the only resource in the dataset, if there is only one.",
            ),
        ),
        default='entry_point',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Which resources should the primitive map.",
    )
    error_on_no_resources = hyperparams_module.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Throw an exception if no resource is selected/provided. Otherwise issue a warning.",
    )


# TODO: Implement optimized "fit_multi_produce" which calls "fit_multi_produce" of underlying primitive.
class DataFrameDatasetMapPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive which for dataset entry point ``DataFrame`` resource (by default)
    runs provided ``primitive`` on it, producing a new resource.

    ``primitive`` can be transformer or fitted or unfitted unsupervised learner primitive.
    If it is already fitted and you do not want to re-fit it, set ``fit_primitive`` to ``no``.
    Otherwise, if ``fit_primitive`` is set to ``fit``, for resource a copy of the
    primitive will be made and it will be first fitted and then produced on that resource.
    If ``fit_primitive`` is set to ``continue_fit``, the primitive is continue fitted on
    all resources in the dataset, in resource ID order.

    Input to the ``primitive`` has to be container ``DataFrame``, but output can be any
    container type.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '5bef5738-1638-48d6-9935-72445f0eecdc',
            'version': '0.1.0',
            'name': "Map DataFrame resources to new resources using provided primitive",
            'python_path': 'd3m.primitives.operator.dataset_map.DataFrameCommon',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/dataset_map.py',
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
                # TODO: Change to "MAP".
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
        },
    )

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

        self._training_inputs: Inputs = None
        self._resource_primitives: typing.Union[typing.Dict[str, base.PrimitiveBase], base.PrimitiveBase] = None
        self._fitted: bool = False

    def _should_fit(self) -> bool:
        if self.hyperparams['fit_primitive'] == 'no':
            return False

        if isinstance(self.hyperparams['primitive'], transformer.TransformerPrimitiveBase):
            return False

        if self.hyperparams['fit_primitive'] == 'continue_fit' and not isinstance(self.hyperparams['primitive'], base.ContinueFitMixin):
            raise exceptions.InvalidArgumentValueError("\"fit_primitive\" hyper-parameter is set to \"continue_fit\", but primitive does not inherit the \"ContinueFitMixin\" class.")

        return True

    def set_training_data(self, *, inputs: Inputs) -> None:  # type: ignore
        if not self._should_fit():
            return

        self._training_inputs = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        if not self._should_fit():
            return base.CallResult(None)

        if self._training_inputs is None:
            raise exceptions.InvalidStateError("Missing training data.")

        self._resource_primitives = self._fit_resources(self._training_inputs)
        self._fitted = True

        return base.CallResult(None)

    def _fit_resources(self, inputs: Inputs) -> typing.Union[typing.Dict[str, base.PrimitiveBase], base.PrimitiveBase]:
        resources_to_use = self._get_resources(inputs)

        if self.hyperparams['fit_primitive'] == 'fit':
            primitive = None
            resource_primitives: typing.Union[typing.Dict[str, base.PrimitiveBase], base.PrimitiveBase] = {}
        else:
            # We just use provided primitive as-is. Runtime already copies it once for us.
            primitive = self.hyperparams['primitive']
            resource_primitives = primitive

        for resource_id in resources_to_use:
            resource = self._prepare_resource(inputs.metadata, inputs[resource_id], resource_id)

            # If "fit_primitive" is "continue_fit" we have only
            # one primitive instance for the whole dataset.
            if self.hyperparams['fit_primitive'] == 'fit':
                primitive = copy.deepcopy(self.hyperparams['primitive'])
                typing.cast(typing.Dict[str, base.PrimitiveBase], resource_primitives)[resource_id] = primitive

            primitive.set_training_data(inputs=resource)

            if self.hyperparams['fit_primitive'] == 'fit':
                primitive.fit()
            else:
                assert self.hyperparams['fit_primitive'] == 'continue_fit'
                primitive.continue_fit()

        return resource_primitives

    def _prepare_resource(self, inputs_metadata: metadata_base.DataMetadata, resource: container.DataFrame, resource_id: str) -> container.DataFrame:
        assert isinstance(resource, container.DataFrame)

        resource = copy.copy(resource)

        resource.metadata = metadata_base.DataMetadata({
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
        })

        resource.metadata = inputs_metadata.copy_to(
            resource.metadata,
            (resource_id,),
        )

        return resource

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        if self._should_fit() and not self._fitted:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        assert self._should_fit() == self._fitted
        assert (self._resource_primitives is not None) == self._fitted

        if self.hyperparams['produce_method'] != 'produce' and not self.hyperparams['produce_method'].startswith('produce_'):
            raise exceptions.InvalidArgumentValueError(f"Invalid produce method name in \"produce_method\" hyper-parameter: {self.hyperparams['produce_method']}")

        outputs = self._produce_dataset(inputs, self._resource_primitives)

        return base.CallResult(outputs)

    def _get_resources(self, inputs: Inputs) -> typing.List[str]:
        if self.hyperparams['resources'] == 'all':
            # We sort so that we potentially continue fit in resource ID order.
            resources_to_use = sorted(
                resource_id for resource_id, resource in inputs.items()
                if isinstance(resource, container.DataFrame)
            )
            resources_not_to_use: typing.List[str] = []
        elif self.hyperparams['resources'] == 'entry_point':
            try:
                resources_to_use = [
                    base_utils.get_tabular_resource_metadata(
                        inputs.metadata,
                        None,
                        pick_entry_point=True,
                        pick_one=True,
                    ),
                ]
            except ValueError:
                resources_to_use = []
            resources_not_to_use = []
        else:
            resources_not_to_use = [
                resource_id for resource_id in self.hyperparams['resources']
                if resource_id not in inputs or not isinstance(inputs[resource_id], container.DataFrame)
            ]
            # We sort so that we potentially continue fit in resource ID order.
            resources_to_use = sorted(
                resource_id for resource_id in self.hyperparams['resources']
                if resource_id not in resources_not_to_use
            )

        if not resources_to_use:
            if self.hyperparams['error_on_no_resources']:
                raise ValueError("No inputs resources.")
            else:
                self.logger.warning("No inputs resources.")

        if self.hyperparams['resources'] not in ['all', 'entry_point'] and resources_not_to_use:
            self.logger.warning("Not all specified inputs resources can be used. Skipping resources: %(resources)s", {
                'resources': resources_not_to_use,
            })

        return resources_to_use

    def _produce_dataset(
        self, inputs: Inputs,
        resource_primitives: typing.Optional[typing.Union[typing.Dict[str, base.PrimitiveBase], base.PrimitiveBase]],
    ) -> Outputs:
        resources_to_use = self._get_resources(inputs)

        outputs = inputs.copy()

        for resource_id in resources_to_use:
            self._produce_resource(outputs, resource_id, resource_primitives)

        return outputs

    # TODO: Instead of copying metadata to a resource and then back, we could maybe just hack it by setting a correct reference.
    #       So resource metadata would point directly into dataset's metadata object for
    #       element corresponding to the resource. How would that work if there is any metadata
    #       on dataset's ALL_ELEMENTS? For updating it does not matter because resource metadata
    #       has precedence anyway? But we would still first have to copy metadata from ALL_ELEMENTS
    #       to resource metadata so that it is available there for querying.
    def _produce_resource(
        self, outputs: Outputs, resource_id: str,
        resource_primitives: typing.Optional[typing.Union[typing.Dict[str, base.PrimitiveBase], base.PrimitiveBase]],
    ) -> None:
        if resource_primitives is not None:
            if self.hyperparams['fit_primitive'] == 'fit':
                primitive = typing.cast(typing.Dict[str, base.PrimitiveBase], resource_primitives)[resource_id]
            else:
                assert self.hyperparams['fit_primitive'] == 'continue_fit'
                # When "fit_primitive" is "continue_fit", we have only
                # one primitive instance for the whole dataset.
                primitive = typing.cast(base.PrimitiveBase, resource_primitives)
        else:
            # It could be that "fit_primitive" is "no" or that we have a transformer primitive.
            primitive = self.hyperparams['primitive']

        resource = self._prepare_resource(outputs.metadata, outputs[resource_id], resource_id)

        output_resource = getattr(primitive, self.hyperparams['produce_method'])(inputs=resource).value

        outputs[resource_id] = output_resource

        outputs.metadata = outputs.metadata.remove((resource_id,), recursive=True)
        outputs.metadata = output_resource.metadata.copy_to(
            outputs.metadata,
            (),
            (resource_id,),
        )

        # TODO: Should we compact metadata? It could make it nicer.

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                resource_params=None,
            )

        elif isinstance(self._resource_primitives, dict):
            return Params(
                resource_params={
                    resource_id: primitive.get_params()
                    for resource_id, primitive in self._resource_primitives.items()
                },
            )

        else:
            return Params(resource_params=self._resource_primitives.get_params())

    def set_params(self, *, params: Params) -> None:
        if params['resource_params'] is None:
            self._resource_params = None
            self._fitted = False

        elif isinstance(params['resource_params'], dict):
            resource_primitives = {}
            for resource_id, params in params['resource_params'].items():
                primitive = copy.deepcopy(self.hyperparams['primitive'])
                primitive.set_params(params)
                resource_primitives[resource_id] = primitive

            self._resource_primitives = resource_primitives
            self._fitted = True

        else:
            self.hyperparams['primitive'].set_params(params['resource_params'])
            self._resource_primitives = self.hyperparams['primitive']
            self._fitted = True
