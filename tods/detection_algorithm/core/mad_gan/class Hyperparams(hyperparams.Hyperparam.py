class Hyperparams(hyperparams.Hyperparams):
    # options pertaining to data
    seq_length = hyperparams.Hyperparameter[int](
        default=30,
        description='Selecting a suitable subsequence resolution (ie. sub-sequence length)',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    num_signals = hyperparams.Hyperparameter[int](
        default=1,
        description='Number of Signals',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    normalise = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="normalise the training/vali/test data (during split)?",
    )
    cond_dim = hyperparams.Hyperparameter[int](
        default=0,
        description='dimension of *conditional* input',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_val = hyperparams.Hyperparameter[int](
        default=0,
        description='assume conditional codes come from [0, max_val)',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    one_hot = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="convert categorical conditional information to one-hot encoding",
    )
    predict_labels = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="instead of conditioning with labels, require model to output them",
    )

    # hyperparameters of the model
    hidden_units_g = hyperparams.Hyperparameter[int](
        default=1000,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    hidden_units_d = hyperparams.Hyperparameter[int](
        default=1000,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    kappa = hyperparams.Uniform(
        lower=0,
        upper=1.0,
        default=1.0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="weight between final output and intermediate steps in discriminator cost (1 = all intermediate",
    )
    latent_dim = hyperparams.Hyperparameter[int](
        default=5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="dimensionality of the latent/noise space",
    )
    batch_mean = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="append the mean of the batch to all variables for calculating discriminator loss",
    )
    learn_scale = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="make the 'scale' parameter at the output of the generator learnable (else fixed to 1",
    )

    # options pertaining to training
    learning_rate = hyperparams.Uniform(
        lower=0,
        upper=100000000,
        default=0.1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    batch_size = hyperparams.Hyperparameter[int](
        default=28,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    num_epochs = hyperparams.Hyperparameter[int](
        default=100,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    D_rounds = hyperparams.Hyperparameter[int](
        default=5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="number of rounds of discriminator training",
    )
    G_rounds = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="number of rounds of generator training",
    )
    use_time = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="enforce latent dimension 0 to correspond to time",
    )
    WGAN = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    WGAN_clip = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    shuffle = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )
    wrong_labels = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="augment discriminator loss with real examples with wrong (~shuffled, sort of) labels",
    )

    # options pertaining to evaluation and exploration
    identifier = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="identifier string for output files",
    )

    # options pertaining to differential privacy
    dp = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="train discriminator with differentially private SGD?",
    )
    l2norm_bound = hyperparams.Uniform(
        lower=0,
        upper=10000,
        default=1e-5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="bound on norm of individual gradients for DP training",
    )
    batches_per_lot = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="number of batches per lot (for DP)",
    )
    dp_sigma = hyperparams.Uniform(
        lower=0,
        upper=100000000,
        default=1e-5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="sigma for noise added (for DP)",
    )




