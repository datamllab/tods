from searcher import RaySearcher, datapath_to_dataset, json_to_searchspace
from hyper_searcher import HyperSearcher
import argparse
import os
import ray
from tods import generate_dataset, evaluate_pipeline, fit_pipeline, load_pipeline, produce_fitted_pipeline, load_fitted_pipeline, save_fitted_pipeline, fit_pipeline
def argsparser():
  parser = argparse.ArgumentParser("Automatically searching hyperparameters for video recognition")
  parser.add_argument('--alg', type=str, default='nevergrad',
          choices=['random', 'hyperopt', 'zoopt', 'skopt', 'nevergrad'])

  parser.add_argument('--beta', type=float, default=1,
                      help='Evaluation Metric (F1, F1_MACRO)')
  parser.add_argument('--num_samples', type=int, default=6)
  parser.add_argument('--gpu', help='Which gpu device to use. Empty string for CPU', type=str, default='1')
  parser.add_argument('--data_dir', help='The path of CSV file', type=str, default='../../datasets/anomaly/raw_data/yahoo_sub_5.csv')

  parser.add_argument('--target_index', help = 'Target index', type = int, default = 4)
  parser.add_argument('--metric', help = 'pipeline evaluation metric', type = str, default = 'ALL')

  parser.add_argument('--search_space_path', help = 'The path of the search space', type = str, default = 'tods/searcher/example_search_space.json')
  parser.add_argument('--use_all_combinations', help = 'generate all possible combinations when reading search space from json', type = bool, default = True)

  # parser.add_argument('--use_all_combinations', help = 'generate all possible combinations when reading search space from json', type = bool, default = False)
  parser.add_argument('--ignore_hyperparameters', help = 'if you want to ignore hyperparmeter when reading search space from json', type = bool, default = False)

  parser.add_argument('--run_mode', help = 'mode of tune.run', type = str, default = 'min', choices = ['min', 'max'])
  return parser

def run(args):
  # get the dataset
  dataset = datapath_to_dataset(args.data_dir, args.target_index)

  searcher = HyperSearcher(dataset, args.metric)


  config = {
    "searching_algorithm": args.alg,
    "num_samples": args.num_samples,
    "mode": args.run_mode
  }


# search space
  search_space = {
    "feature_analysis": ray.tune.choice(["statistical_maximum"]),
    "detection_algorithm": ray.tune.choice(["pyod_loda","telemanom"])
  }

  import time
  start_time = time.time()

  # start searching
  best_config, best_pipeline_id = searcher.search(search_space=search_space, config=config)

  print("--- %s seconds ---" % (time.time() - start_time))

  print("Best config: ", best_config)


  best_feature_analysis = best_config['feature_analysis']
  best_detection_algorithm = best_config['detection_algorithm']
  print(best_feature_analysis)
  print(best_detection_algorithm)

  dictt = {"statistical_maximum_window_size": [1,2,3],
  "telemanom_smoothing_perc": [0.05, 0.06, 0.07],
  "telemanom_error_buffer": [55, 60, 65],
  "telemanom_batch_size": [70, 80, 90],
  "telemanom_dropout": [0.3, 0.4, 0.5],
  "telemanom_validation_split": [0.2, 0.25, 0.3],
  "telemanom_lstm_batch_size": [64, 60, 68],
  "telemanom_patience": [10, 13, 16],
  "pyod_loda_n_bins": [10, 20, 30],
  "pyod_loda_n_random_cuts": [100, 110, 120]
  }



  hyperparam_search_space = {
    "feature_analysis": ray.tune.choice([str(best_feature_analysis)]),
    "detection_algorithm": ray.tune.choice([str(best_detection_algorithm)])

  }

  for key, value in dictt.items():
    if str(best_feature_analysis) in str(key):
      hyperparam_search_space[str(key)] = ray.tune.choice(value)
    if str(best_detection_algorithm) in str(key):
      hyperparam_search_space[str(key)] = ray.tune.choice(value)

  # print(hyperparam_search_space)





  # print(hyperparam_search_space)

  # best_config, best_pipeline_id = searcher.search(search_space=hyperparam_search_space, config=config)

  # print(best_config)

  print()




  # hyper_searcher = HyperSearcher(dataset, args.metric)


if __name__ == '__main__':
  parser = argsparser()
  args = parser.parse_args()

  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

  os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
  print(args)
  # Search
  run(args)
