import pandas as pd

from tods import schemas as schemas_utils
from tods import generate_dataset, evaluate_pipeline, fit_pipeline, load_fitted_pipeline_by_pipeline, load_fitted_pipeline_by_id, save, load, save2, load_pipeline


from d3m.metadata import base as metadata_base
from axolotl.backend.simple import SimpleRunner
import uuid


from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# Creating pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

# Step 0: dataset_to_dataframe
step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

# Step 1: column_parser
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

# Step 2: extract_columns_by_semantic_types(attributes)
step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_output('produce')
step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
							  data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
pipeline_description.add_step(step_2)

# Step 3: extract_columns_by_semantic_types(targets)
step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_3.add_output('produce')
step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
							data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
pipeline_description.add_step(step_3)

attributes = 'steps.2.produce'
targets = 'steps.3.produce'

# Step 4: processing
#step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.transformation.axiswise_scaler'))
step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_maximum'))
#step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_minimum'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
step_4.add_output('produce')
pipeline_description.add_step(step_4)

# Step 5: algorithm`
step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_ae'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_output('produce')
pipeline_description.add_step(step_5)

# Step 6: Predictions
step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_6.add_output('produce')
pipeline_description.add_step(step_6)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')

# Output to json
data = pipeline_description.to_json()
with open('autoencoder_pipeline.json', 'w') as f:
    f.write(data)
    print(data)



table_path = 'datasets/anomaly/raw_data/yahoo_sub_5.csv'
df = pd.read_csv(table_path)
dataset = generate_dataset(df, 6)
# pipeline = schemas_utils.load_default_pipeline()
pipeline = load_pipeline('autoencoder_pipeline.json')
# pipeline_result, output, fitted_pipeline = temp(dataset, pipeline, 'F1_MACRO')
id_ = save(dataset, pipeline, 'F1_MACRO')
# print(fitted_pipeline)


table_path = 'datasets/anomaly/raw_data/yahoo_sub_5.csv'
df = pd.read_csv(table_path)
dataset = generate_dataset(df, 5)


def sampling(args):
  """Reparametrisation by sampling from Gaussian, N(0,I)
  To sample from epsilon = Norm(0,I) instead of from likelihood Q(z|X)
  with latent variables z: z = z_mean + sqrt(var) * epsilon
  Parameters
  ----------
  args : tensor
      Mean and log of variance of Q(z|X).

  Returns
  -------
  z : tensor
      Sampled latent variable.
  """
  from tensorflow.keras import backend as K
  z_mean, z_log = args
  batch = K.shape(z_mean)[0]  # batch size
  dim = K.int_shape(z_mean)[1]  # latent dimension
  epsilon = K.random_normal(shape=(batch, dim))  # mean=0, std=1.0

  return z_mean + K.exp(0.5 * z_log) * epsilon


print(load(dataset, id_))

# from pyod.models.VAE import sampling

# import tensorflow.keras as keras

# model = keras.models.load_model('fitted_pipelines/' + '06f1d169-bb9f-4f4b-8031-41b99e0bee88' + '/model/keras_model', custom_objects={'sampling': sampling})

# print(model)

# model = keras.models.load_model('fitted_pipelines/' + '805a3048-f7ea-47e1-83d1-09c88b0e7953' + '/model/keras_model')


# works:
# tods.detection_algorithm.pyod_ae
# tods.detection_algorithm.pyod_cof directly joblib plz 


# print(evaluate_pipeline(dataset, pipeline, 'F1_MACRO'))