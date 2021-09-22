from searcher import RaySearcher, datapath_to_dataset, json_to_searchspace
import argparse
import os
#config for ray searcher
# config = {
#   "searching_algorithm": 'hyperopt',
#   "num_samples": 15,
# }






#2 dataset for test
# dataset = datapath_to_dataset('../../datasets/anomaly/kpi/kpi_dataset/tables/learningData.csv', 3)
# dataset = datapath_to_dataset('../../datasets/anomaly/raw_data/yahoo_sub_5.csv', 6)




#transform json to search space
#all_combination True means calculating all possibilities
#ignore_hyperparams False means not adding hyperparams into search space
# search_space = json_to_searchspace(path = 'test.json', config = config, all_combination = False, ignore_hyperparams = False)

#a self defined seach space
# search_space2 = {
#   "timeseries_processing": tune.choice(["moving_average_transform"]),
#   "feature_analysis": tune.choice(["statistical_maximum", "statistical_minumum"]),
#   "detection_algorithm": tune.choice(["pyod_ae"])
# }



# print(search_space)
# #create instance
# s = RaySearcher(dataset, 'F1_MACRO')
# #run search function
# print(s.search(search_space=search_space,
#         config=config))


def argsparser():
  parser = argparse.ArgumentParser("Automatically searching hyperparameters for video recognition")
  parser.add_argument('--alg', type=str, default='hyperopt',
          choices=['random', 'hyperopt'])
  parser.add_argument('--num_samples', type=int, default=15)
  parser.add_argument('--gpu', help='Which gpu device to use. Empty string for CPU', type=str, default='0')
  parser.add_argument('--data_dir', help='The path of CSV file', type=str, default='../../datasets/anomaly/raw_data/yahoo_sub_5.csv')

  parser.add_argument('--target_index', help = 'Target index', type = int, default = 6)
  parser.add_argument('--metric', help = 'pipeline evaluation metric', type = str, default = 'F1_MACRO')

  parser.add_argument('--search_space_path', help = 'The path of the search space', type = str, default = 'test.json')
  parser.add_argument('--use_all_combinations', help = 'generate all possible combinations when reading search space from json', type = bool, default = False)
  parser.add_argument('--ignore_hyperparameters', help = 'if you want to ignore hyperparmeter when reading search space from json', type = bool, default = False)

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
  }

  # define the search space
  search_space = json_to_searchspace(path = args.search_space_path,
                                    config = config,
                                    use_all_combination = args.ignore_hyperparameters,
                                    ignore_hyperparams = args.ignore_hyperparameters
  )

  # or you can define seach space here like this

  # search_space = {
  #   "timeseries_processing": tune.choice(["moving_average_transform"]),
  #   "feature_analysis": tune.choice(["statistical_maximum", "statistical_minumum"]),
  #   "detection_algorithm": tune.choice(["pyod_ae"])
  # }

  # start searching
  best_config = searcher.search(search_space=search_space, config=config)

  print("Best config: ", best_config)

if __name__ == '__main__':
  parser = argsparser()
  args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

  # Search
  run(args)
