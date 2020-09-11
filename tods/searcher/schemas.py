import os

resource_dir = os.path.dirname(__file__)

DEFAULT_PIPELINE_DIR = os.path.join(resource_dir, 'resources', 'default_pipeline.json')

def load_default_pipeline():
    from axolotl.utils import pipeline as pipeline_utils
    pipeline = pipeline_utils.load_pipeline(DEFAULT_PIPELINE_DIR)
    return pipeline
