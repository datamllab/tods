from searcher import RaySearcher, datapath_to_dataset, json_to_searchspace

#config for ray searcher
config = {
  "searching_algorithm": 'hyperopt',
  "num_samples": 15,
}






#2 dataset for test
# dataset = datapath_to_dataset('../../datasets/anomaly/kpi/kpi_dataset/tables/learningData.csv', 3)
dataset = datapath_to_dataset('../../datasets/anomaly/raw_data/yahoo_sub_5.csv', 6)




#transform json to search space
#all_combination True means calculating all possibilities
#ignore_hyperparams False means not adding hyperparams into search space
search_space = json_to_searchspace(path = 'test.json', config = config, all_combination = False, ignore_hyperparams = False)

#a self defined seach space
# search_space2 = {
#   "timeseries_processing": tune.choice(["moving_average_transform"]),
#   "feature_analysis": tune.choice(["statistical_maximum", "statistical_minumum"]),
#   "detection_algorithm": tune.choice(["pyod_ae"])
# }



print(search_space)
#create instance
s = RaySearcher(dataset, 'F1_MACRO')
#run search function
print(s.search(search_space=search_space,
        config=config))
