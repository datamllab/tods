import os
import json
import re

from axolotl.utils.resources import check_directory


# A class to wrap envrioment variables under d3m scope.
class EnvVars:
    # A label what is the setting under which the pod is being run; possible
    # values: ta2, ta2ta3; this variable is available only for informative
    # purposes but it is not used anymore to change an overall mode of operation
    #  of TA2 system because now TA2 evaluation will happen through TA2-TA3 API
    # as well
    D3MRUN = 'run'
    PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '../..')
    # A location of dataset(s), can contain multiple datasets in arbitrary
    # directory structure, read-only
    D3MINPUTDIR = '/input_dir'
    # A location to problem description to use (should be under D3MINPUTDIR),
    # datasets are linked from the problem description using IDs, those datasets
    #  should exist inside D3MINPUTDIR
    D3MPROBLEMPATH = 'problem_path'
    # A location of output files, shared by TA2 and TA3 pods (and probably data
    # mart)
    D3MOUTPUTDIR = os.path.join(PROJECT_ROOT, 'output_dir')
    # A local-to-host directory provided; used by memory sharing mechanisms
    D3MLOCALDIR = os.path.join(D3MOUTPUTDIR, 'temp', 'plasma')
    # A path to the volume with primitives' static files
    D3MSTATICDIR = None
    # Available CPU units in Kubernetes specification
    D3MCPU = 0
    # Available CPU units in Kubernetes specification
    D3MRAM = 0
    # Time limit for the search phase (available to the pod), in seconds
    D3MTIMEOUT = -1

    # Plasma socket
    PLASMA_SOCKET = '/tmp/plasma'

    # datamart uri DATAMART_URL_NYU
    DATAMART_URL_NYU = 'https://datamart.d3m.vida-nyu.org'

    if 'D3MRUN' in os.environ:
        D3MRUN = os.environ['D3MRUN']
    if 'D3MINPUTDIR' in os.environ:
        D3MINPUTDIR = os.environ['D3MINPUTDIR']
    if 'D3MPROBLEMPATH' in os.environ:
        D3MPROBLEMPATH = os.environ['D3MPROBLEMPATH']
    if 'D3MOUTPUTDIR' in os.environ:
        D3MOUTPUTDIR = os.environ['D3MOUTPUTDIR']
    if 'D3MLOCALDIR' in os.environ:
        D3MLOCALDIR = os.environ['D3MLOCALDIR']
    if 'D3MSTATICDIR' in os.environ:
        D3MSTATICDIR = os.environ['D3MSTATICDIR']
    if 'D3MCPU' in os.environ:
        D3MCPU = int(float(os.environ['D3MCPU']))
    # if we don't set it or its to low set to 4
    # if D3MCPU < 4:
    #     D3MCPU = 4
    if 'D3MRAM' in os.environ:
        D3MRAM = int(re.search(r'\d+', os.environ['D3MRAM']).group())
    if 'D3MTIMEOUT' in os.environ:
        D3MTIMEOUT = os.environ['D3MTIMEOUT']
    if 'PLASMA_SOCKET' in os.environ:
        PLASMA_SOCKET = os.environ['PLASMA_SOCKET']
    if 'DATAMART_URL_NYU' in os.environ:
        DATAMART_URL_NYU = os.environ['DATAMART_URL_NYU']


# #
class Path:
    # Temporary directories.
    # A temporary directory for other things.
    TEMP_STORAGE_ROOT = os.path.join(EnvVars.D3MOUTPUTDIR, 'temp/')
    # A temporary directory to store other stuff between ta2-ta3
    OTHER_OUTPUTS = os.path.join(TEMP_STORAGE_ROOT, 'other_outputs')
    # To deprecate after figure out what to do with executables.
    TEMP_PROBLEM_DESC = os.path.join(TEMP_STORAGE_ROOT, 'problem_description')

    check_directory(TEMP_STORAGE_ROOT)
    check_directory(OTHER_OUTPUTS)
    check_directory(TEMP_PROBLEM_DESC)


class SearchPath:

    def __init__(self, search_id):
        self.base_path = os.path.join(EnvVars.D3MOUTPUTDIR, search_id)

        # A directory with ranked pipelines to be evaluated, named
        # <pipeline id>.json; these files should have additional field pipeline_rank
        self.pipelines_ranked = os.path.join(self.base_path, 'pipelines_ranked')
        check_directory(self.pipelines_ranked)

        # A directory with successfully scored pipelines during the search,
        # named <pipeline id>.json
        self.pipelines_scored = os.path.join(self.base_path, 'pipelines_scored')
        check_directory(self.pipelines_scored)
        # A directory of full pipelines which have not been scored or ranked for any
        #  reason, named <pipeline id>.json
        self.pipelines_searched = os.path.join(self.base_path, 'pipelines_searched')
        check_directory(self.pipelines_searched)
        # A directory with any subpipelines referenced from pipelines in
        # pipelines_* directories, named <pipeline id>.json
        self.subpipelines = os.path.join(self.base_path, 'subpipelines')
        check_directory(self.subpipelines)
        # A directory with pipeline run records in YAML format, multiple can be
        # stored in the same file, named <pipeline run id>.yml
        self.pipeline_runs = os.path.join(self.base_path, 'pipeline_runs')
        check_directory(self.pipeline_runs)
        # A directory where TA2 system can store any additional datasets to be
        # provided during training and testing to their pipelines; each dataset
        # should be provided in a sub-directory in a D3M dataset format; all
        # datasets here should have an unique ID; in the case that additional
        # datasets are provided, TA2 should output also pipeline run documents for
        # their ranked pipelines because those pipeline run documents contain
        # information how to map these additional inputs to pipeline inputs
        self.additional_inputs = os.path.join(self.base_path, 'additional_inputs')
        check_directory(self.additional_inputs)


# A class that wraps a block list of primitives
# To generate this list is necessary to run modules.utils.primitive_selection
class PrimitivesList:
    with open(os.path.join(os.path.dirname(__file__), '..', 'utils', 'resources', 'blocklist.json'), 'r') as file:
        BlockList = json.load(file)
