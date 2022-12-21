from searcher import RaySearcher
import argparse
import os
import ray
import json
import pandas as pd
from tods import generate_dataset,generate_problem, evaluate_pipeline, fit_pipeline, load_pipeline, produce_fitted_pipeline, load_fitted_pipeline, save_fitted_pipeline, fit_pipeline,datapath_to_dataset
def argsparser():
    parser = argparse.ArgumentParser("Automatically searching hyperparameters for video recognition")
    parser.add_argument('--alg', type=str, default='nevergrad',
            choices=['random', 'hyperopt', 'zoopt', 'skopt', 'nevergrad'])
    parser.add_argument('--beta', type=float, default=1.0,
                        help='with respect to a user who attaches beta times as much importance to recall as precision')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--gpu', help='Which gpu device to use. Empty string for CPU', type=str, default='0')
    parser.add_argument('--data_dir', help='The path of CSV file', type=str, default='../../datasets/anomaly/raw_data/yahoo_sub_5.csv')

    parser.add_argument('--target_index', help = 'Target index', type = int, default = 6)

    #add choice?
    parser.add_argument('--metric', help = 'pipeline evaluation metric', type = str, default = 'ALL')

    parser.add_argument('--search_space_path', help = 'The path of the search space', type = str, default = './search_space/test.json')
    parser.add_argument('--use_all_combinations', help = 'generate all possible combinations when reading search space from json', type = bool, default = True)
    parser.add_argument('--ignore_hyperparameters', help = 'if you want to ignore hyperparmeter when reading search space from json', type = bool, default = True)

    parser.add_argument('--run_mode', help = 'mode of tune.run', type = str, default = 'min', choices = ['min', 'max'])

    return parser




def run(args):
    # get the dataset
    dataset = datapath_to_dataset(args.data_dir, args.target_index)
    dataframe=pd.read_csv(args.data_dir)
    
    # initialize the searcher
    searcher = RaySearcher(dataframe,args.target_index,dataset, args.metric,args.beta)

    # get the ray searcher config
    config = {
    'metric':'F1_MACRO',
    "searching_algorithm": args.alg,
    "num_samples": args.num_samples,
    "mode": args.run_mode,
    "use_all_combinations": args.use_all_combinations,
    "ignore_hyperparameters": args.ignore_hyperparameters
    }

    import time
    start_time = time.time()

    # get the json search space 
    with open(args.search_space_path) as f:
        search_space= json.load(f)
        
    # start searching
    search_result = searcher.search(search_space = search_space,config=config)

    print("--- %s seconds ---" % (time.time() - start_time))
    
    if isinstance(search_result,list):
        best_config = search_result[1]['best_config']    
        best_pipeline_id = search_result[1]['best_pipeline_id']
    else:
        best_config = search_result['best_config']    
        best_pipeline_id = search_result['best_pipeline_id']
        
    print("Best config: ", best_config)

    print("best config pipeline id: ", best_pipeline_id)

    # load the fitted pipeline based on id
    loaded = load_fitted_pipeline(best_pipeline_id)

    # use the loaded fitted pipelines
    result = produce_fitted_pipeline(dataset, loaded)


    loaded = load_fitted_pipeline(best_pipeline_id)

    # use the loaded fitted pipelines
    result = produce_fitted_pipeline(dataset, loaded)

    print(result)

if __name__ == '__main__':

    parser = argsparser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
    print(args)
    # Search
    run(args)
