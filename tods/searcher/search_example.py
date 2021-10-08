from searcher import RaySearcher, datapath_to_dataset, json_to_searchspace
import argparse
import os
import ray
from tods import generate_dataset, evaluate_pipeline, fit_pipeline, load_pipeline, produce_fitted_pipeline, load_fitted_pipeline, save_fitted_pipeline, fit_pipeline
def argsparser():
  parser = argparse.ArgumentParser("Automatically searching hyperparameters for video recognition")
  parser.add_argument('--alg', type=str, default='hyperopt',
          choices=['random', 'hyperopt'])
  parser.add_argument('--num_samples', type=int, default=3)
  parser.add_argument('--gpu', help='Which gpu device to use. Empty string for CPU', type=str, default='0')
  parser.add_argument('--data_dir', help='The path of CSV file', type=str, default='datasets/anomaly/raw_data/yahoo_sub_5.csv')

  parser.add_argument('--target_index', help = 'Target index', type = int, default = 6)
  parser.add_argument('--metric', help = 'pipeline evaluation metric', type = str, default = 'F1_MACRO')

  parser.add_argument('--search_space_path', help = 'The path of the search space', type = str, default = 'tods/searcher/example_search_space.json')
  parser.add_argument('--use_all_combinations', help = 'generate all possible combinations when reading search space from json', type = bool, default = False)
  parser.add_argument('--ignore_hyperparameters', help = 'if you want to ignore hyperparmeter when reading search space from json', type = bool, default = False)

  parser.add_argument('--run_mode', help = 'mode of tune.run', type = str, default = 'max', choices = ['min', 'max'])
  return parser

def run(args):
  # get the dataset
  dataset = datapath_to_dataset(args.data_dir, args.target_index)

  # initialize the searcher

  searcher = RaySearcher(dataset, args.metric)

  # get the ray searcher config
  config = {
    "searching_algorithm": args.alg,
    "num_samples": args.num_samples,
    "mode": args.run_mode
  }

  # define the search space
  # search_space = json_to_searchspace(path = args.search_space_path,
  #                                   config = config,
  #                                   use_all_combination = args.ignore_hyperparameters,
  #                                   ignore_hyperparams = args.ignore_hyperparameters
  # )

  # or you can define seach space here like this
  # from ray import tune
  search_space = {
    "feature_analysis": ray.tune.choice(["statistical_maximum", "statistical_minimum"]),
    "detection_algorithm": ray.tune.choice(["pyod_ae"])
  }

  # start searching
  best_config, best_pipeline_id = searcher.search(search_space=search_space, config=config)

  print("Best config: ", best_config)
  print("best config pipeline id: ", best_pipeline_id)

  # load the fitted pipeline based on id
  loaded = load_fitted_pipeline(best_pipeline_id)

  # use the loaded fitted pipeline
  result = produce_fitted_pipeline(dataset, loaded)

  print(result)

if __name__ == '__main__':
  parser = argsparser()
  args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

  # Search
  run(args)