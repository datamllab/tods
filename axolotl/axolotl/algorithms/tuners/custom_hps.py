import sys
from collections import OrderedDict

from d3m.metadata import hyperparams

epsilon = sys.float_info.epsilon

clf_xgboost_config = dict(
    n_estimators=hyperparams.UniformInt(
        lower=10,
        upper=50,
        default=20,
        description='The number of trees in the forest.',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter',
            'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
        ],
    ),
    n_more_estimators=hyperparams.UniformInt(
        lower=10,
        upper=50,
        default=20,
        description='When continuing a fit, it controls how many more trees to add every time.',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter',
            'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
        ],
    ),
    max_depth=hyperparams.UniformInt(
        lower=5,
        upper=50,
        default=30,
        lower_inclusive=True,
        upper_inclusive=True,
        description='The maximum depth of the tree.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    learning_rate=hyperparams.LogUniform(
        lower=1e-4,
        upper=1e-1,
        default=0.05,
        lower_inclusive=True,
        upper_inclusive=True,
        description=r'Boosting learning rate (xgb\`s \"eta\")',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    gamma=hyperparams.Constant[float](
        default=0.0,
        description='Minimum loss reduction required to make a further partition on a leaf node of the tree',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    min_child_weight = hyperparams.Constant[int](
        default=1,
        description='Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results '
                    'in a leaf node with the sum of instance weight less than min_child_weight, then the building '
                    'process will give up further partitioning ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    # max_delta_step = hyperparams.Union[Union[int, None]](
    #     configuration=OrderedDict(
    #         limit=hyperparams.Bounded[int](
    #             lower=1,
    #             upper=None,
    #             default=1,
    #             description='Maximum delta step we allow each leaf output to be.'
    #         ),
    #         unlimited=hyperparams.Enumeration[int](
    #             values=[0],
    #             default=0,
    #             description='No constraint.',
    #         ),
    #     ),
    #     default='unlimited',
    #     description='Maximum delta step we allow.',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    # ),
    subsample=hyperparams.Constant[float](
        default=1.0,
        description='Subsample ratio of the training instances,this will prevent overfitting. Subsampling will occur '
                    'once in every boosting iteration.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    colsample_bytree=hyperparams.Constant[float](
        default=1.0,
        description='Subsample ratio of columns when constructing each tree. Subsampling will occur once in every '
                    'boosting iteration',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    colsample_bylevel=hyperparams.Constant[float](
        default=1.0,
        description='Subsample ratio of columns for each split, in each level. Subsampling will occur each time a new '
                    'split is made',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    reg_alpha=hyperparams.Uniform(
        lower=0.1,
        upper=1.0,
        default=0.5,
        lower_inclusive=True,
        upper_inclusive=True,
        description='L1 regularization term on weights. Increasing this value will make model more conservative.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    reg_lambda=hyperparams.Uniform(
        lower=0.1,
        upper=1.0,
        default=0.5,
        lower_inclusive=True,
        upper_inclusive=True,
        description='L2 regularization term on weights. Increasing this value will make model more conservative.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    # scale_pos_weight = hyperparams.Bounded[float](
    #     lower=0,
    #     upper=None,
    #     default=1,
    #     description='Control the balance of positive and negative weights, useful for unbalanced classes',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    # ),
    base_score=hyperparams.Bounded[float](
        lower=0,
        upper=None,
        default=0.5,
        description='The initial prediction score of all instances, global bias.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
)

dfs_single_tab_config = dict(
    max_percent_null=hyperparams.Uniform(
        lower=0,
        upper=1,
        default=0.9,
        lower_inclusive=True,
        upper_inclusive=True,
        description='The maximum allowed correlation between any two features returned. A lower value means features will be more uncorrelated',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )

)

lgbm_clf_config = dict(
    n_estimators=hyperparams.UniformInt(
        lower=10,
        upper=50,
        default=20,
        description='The number of trees in the forest.',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter',
            'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
        ],
    ),
    n_more_estimators=hyperparams.UniformInt(
        lower=10,
        upper=50,
        default=20,
        description='When continuing a fit, it controls how many more trees to add every time.',
        semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/TuningParameter',
            'https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter',
        ],
    ),
    max_depth=hyperparams.UniformInt(
        lower=5,
        upper=50,
        default=30,
        lower_inclusive=True,
        upper_inclusive=True,
        description='The maximum depth of the tree.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    # num_leaves_base=hyperparams.Bounded[float](
    #     lower=1,
    #     upper=2,
    #     default=2,
    #     description='Maximum tree leaves for base learners, this value is the base of the formula num_leaves_base^(max_depth)',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    # ),
    # subsample_for_bin=hyperparams.Bounded[int](
    #     lower=1,
    #     upper=None,
    #     default=200000,
    #     description='number of data that sampled to construct histogram bins',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    # ),
    learning_rate=hyperparams.LogUniform(
        lower=1e-4,
        upper=1e-1,
        default=0.05,
        lower_inclusive=True,
        upper_inclusive=True,
        description=r'Boosting learning rate (xgb\`s \"eta\")',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    min_child_weight = hyperparams.Constant[int](
        default=1,
        description='Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results '
                    'in a leaf node with the sum of instance weight less than min_child_weight, then the building '
                    'process will give up further partitioning ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    # min_child_samples=hyperparams.Bounded[int](
    #     lower=0,
    #     upper=None,
    #     default=20,
    #     description='minimal number of data in one leaf. Can be used to deal with over-fitting',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    # ),
    # max_delta_step = hyperparams.Union[Union[int, None]](
    #     configuration=OrderedDict(
    #         limit=hyperparams.Bounded[int](
    #             lower=1,
    #             upper=None,
    #             default=1,
    #             description='Maximum delta step we allow each leaf output to be.'
    #         ),
    #         unlimited=hyperparams.Enumeration[int](
    #             values=[0],
    #             default=0,
    #             description='No constraint.',
    #         ),
    #     ),
    #     default='unlimited',
    #     description='Maximum delta step we allow.',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    # ),
    subsample=hyperparams.Constant[float](
        default=1.0,
        description='Subsample ratio of the training instances,this will prevent overfitting. Subsampling will occur '
                    'once in every boosting iteration.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    subsample_freq=hyperparams.Bounded[int](
        lower=0,
        upper=1,
        default=0,
        description='frequency for bagging',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    colsample_bytree=hyperparams.Constant[float](
        default=1.0,
        description='Subsample ratio of columns when constructing each tree. Subsampling will occur once in every '
                    'boosting iteration',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    min_split_gain=hyperparams.Bounded[float](
        lower=0,
        upper=None,
        default=0,
        description='the minimal gain to perform split',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    reg_alpha=hyperparams.Uniform(
        lower=0.1,
        upper=1.0,
        default=0.5,
        lower_inclusive=True,
        upper_inclusive=True,
        description='L1 regularization term on weights. Increasing this value will make model more conservative.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
    reg_lambda=hyperparams.Uniform(
        lower=0.1,
        upper=1.0,
        default=0.5,
        lower_inclusive=True,
        upper_inclusive=True,
        description='L2 regularization term on weights. Increasing this value will make model more conservative.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    ),
)

sk_logistic_regression_config = dict(
    dual=hyperparams.Constant[bool](
        default=False,
        description='Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    ),
    penalty=hyperparams.Choice(
        choices={
            'l1': hyperparams.Hyperparams.define(
                configuration=OrderedDict({})
            ),
            'l2': hyperparams.Hyperparams.define(
                configuration=OrderedDict({})
            ),
            'none': hyperparams.Hyperparams.define(
                configuration=OrderedDict({})
            ),
            'elasticnet': hyperparams.Hyperparams.define(
                configuration=OrderedDict({
                    'l1_ratio': hyperparams.Union(
                        configuration=OrderedDict({
                            'float': hyperparams.Uniform(
                                lower=0,
                                upper=1,
                                default=0.001,
                                lower_inclusive=True,
                                upper_inclusive=True,
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                            ),
                            # 'l1_ratio must be between 0 and 1; got (l1_ratio=None)'
                            # 'none': hyperparams.Constant(
                            #     default=None,
                            #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                            # )
                        }),
                        default='float',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    )
                })
            )
        },
        default='l2',
        description='Used to specify the norm used in the penalization. The \'newton-cg\', \'sag\' and \'lbfgs\' solvers support only l2 penalties.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    ),
    intercept_scaling=hyperparams.Constant[float](
        default=1,
        description='Useful only when the solver \'liblinear\' is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a "synthetic" feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes ``intercept_scaling * synthetic_feature_weight``.  Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    ),

)

sk_decision_tree_clf_config = dict(
    min_samples_split=hyperparams.Union(
        configuration=OrderedDict({
            'absolute': hyperparams.Constant[int](
                default=2,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'percent': hyperparams.Bounded[float](
                default=0.25,
                lower=0,
                upper=1,
                lower_inclusive=False,
                # upper_inclusive=False,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='absolute',
        description='The minimum number of samples required to split an internal node:  - If int, then consider `min_samples_split` as the minimum number. - If float, then `min_samples_split` is a percentage and `ceil(min_samples_split * n_samples)` are the minimum number of samples for each split.  .. versionchanged:: 0.18 Added float values for percentages.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    ),
    max_features=hyperparams.Union(
        configuration=OrderedDict({
            # max_features must be in (0, n_features]
            # 'specified_int': hyperparams.Bounded[int](
            #     lower=0,
            #     upper=None,
            #     default=0,
            #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            # ),
            'calculated': hyperparams.Enumeration[str](
                values=['auto', 'sqrt', 'log2'],
                default='auto',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'percent': hyperparams.Bounded[float](
                default=0.25,
                lower=0,
                upper=1,
                lower_inclusive=False,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        description='The number of features to consider when looking for the best split:  - If int, then consider `max_features` features at each split. - If float, then `max_features` is a percentage and `int(max_features * n_features)` features are considered at each split. - If "auto", then `max_features=sqrt(n_features)`. - If "sqrt", then `max_features=sqrt(n_features)`. - If "log2", then `max_features=log2(n_features)`. - If None, then `max_features=n_features`.  Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than ``max_features`` features.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    ),
    # 'max_leaf_nodes 0 must be either None or larger than 1'
    max_leaf_nodes=hyperparams.Constant(
        default=None,
        description='Grow a tree with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    ),
)

sk_sgd_clf_config = dict(
    validation_fraction=hyperparams.Bounded[float](
        default=0.1,
        lower=0,
        upper=0.99999999999,
        lower_inclusive=False,
        # upper_inclusive=False,
        description='The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    ),
    # eta0 must be > 0
    eta0=hyperparams.Bounded[float](
        lower=0.0,
        upper=1.0,
        default=0.1,
        lower_inclusive=False,
        description='The initial learning rate for the \'constant\' or \'invscaling\' schedules. The default value is 0.0 as eta0 is not used by the default schedule \'optimal\'.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    ),


)

sk_random_forest_clf_config = dict(
    max_features=hyperparams.Union(
        configuration=OrderedDict({
            # max_features must be in (0, n_features]
            # 'specified_int': hyperparams.Bounded[int](
            #     lower=0,
            #     upper=None,
            #     default=0,
            #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            # ),
            'calculated': hyperparams.Enumeration[str](
                values=['auto', 'sqrt', 'log2'],
                default='auto',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'percent': hyperparams.Uniform(
                default=0.25,
                lower=0,
                upper=1,
                lower_inclusive=True,
                upper_inclusive=False,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='calculated',
        description='The number of features to consider when looking for the best split:  - If int, then consider `max_features` features at each split. - If float, then `max_features` is a percentage and `int(max_features * n_features)` features are considered at each split. - If "auto", then `max_features=sqrt(n_features)`. - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto"). - If "log2", then `max_features=log2(n_features)`. - If None, then `max_features=n_features`.  Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than ``max_features`` features.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    ),
    max_samples=hyperparams.Union(
        configuration=OrderedDict({
            'absolute': hyperparams.Bounded[int](
                lower=0,
                upper=None,
                lower_inclusive=False,
                default=1,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'percent': hyperparams.Bounded[float](
                default=0.9,
                lower=0 + epsilon,
                upper=1,
                upper_inclusive=False,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    ),
)

sk_extra_tree_tree_clf_config = dict(
    max_features=hyperparams.Union(
        configuration=OrderedDict({
            'calculated': hyperparams.Enumeration[str](
                values=['auto', 'sqrt', 'log2'],
                default='auto',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'percent': hyperparams.Bounded[float](
                default=0.25,
                lower=0,
                upper=1,
                lower_inclusive=False,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='calculated',
        description='The number of features to consider when looking for the best split:  - If int, then consider `max_features` features at each split. - If float, then `max_features` is a percentage and `int(max_features * n_features)` features are considered at each split. - If "auto", then `max_features=sqrt(n_features)`. - If "sqrt", then `max_features=sqrt(n_features)`. - If "log2", then `max_features=log2(n_features)`. - If None, then `max_features=n_features`.  Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than ``max_features`` features.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    ),
    max_samples=hyperparams.Union(
        configuration=OrderedDict({
            'absolute': hyperparams.Bounded[int](
                lower=0,
                upper=None,
                lower_inclusive=False,
                default=1,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'percent': hyperparams.Bounded[float](
                default=0.9,
                lower=0 + epsilon,
                upper=1,
                upper_inclusive=False,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
)

# To avoid the issue, https://gitlab.com/TAMU_D3M/d3m_primitives/-/issues/1
tamu_feature_selection_config = dict(
    percentage_selected_features=hyperparams.Uniform(
        default=0.5,
        upper=1,
        lower=0.25,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="percentage of features to select, between 0 and 1")
)

config = {
    'd3m.primitives.classification.xgboost_gbtree.DataFrameCommon': clf_xgboost_config,
    'd3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization': dfs_single_tab_config,
    'd3m.primitives.classification.light_gbm.DataFrameCommon': lgbm_clf_config,
    'd3m.primitives.classification.logistic_regression.SKlearn': sk_logistic_regression_config,
    'd3m.primitives.classification.decision_tree.SKlearn': sk_decision_tree_clf_config,
    'd3m.primitives.classification.sgd.SKlearn': sk_sgd_clf_config,
    'd3m.primitives.classification.random_forest.SKlearn': sk_random_forest_clf_config,
    'd3m.primitives.classification.extra_trees.SKlearn': sk_extra_tree_tree_clf_config,
    'd3m.primitives.feature_selection.skfeature.TAMU': tamu_feature_selection_config,
}
