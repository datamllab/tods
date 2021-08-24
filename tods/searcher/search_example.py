from searcher import RaySearcher, datapath_to_dataset, json_to_searchspace


config = {
  "searching_algorithm": 'hyperopt',
  "num_samples": 15,
}








# dataset = datapath_to_dataset('../../datasets/anomaly/kpi/kpi_dataset/tables/learningData.csv', 3)

dataset = datapath_to_dataset('../../yahoo_sub_5.csv', 6)





search_space = json_to_searchspace(path = 'test2.json', config = config, all_combination = False, ignore_hyperparams = False)

# search_space2 = {
#   "timeseries_processing": tune.choice(["moving_average_transform"]),
#   "feature_analysis": tune.choice(["statistical_maximum", "statistical_minumum"]),
#   "detection_algorithm": tune.choice(["pyod_ae"])
# }



print(search_space)

s = RaySearcher(dataset, 'F1_MACRO')

print(s.search(search_space=search_space,
        config=config))

# {'moving_average_transform_window_size': <ray.tune.sample.Categorical object at 0x7f23178b8278>, 
# 'timeseries_processing': <ray.tune.sample.Categorical object at 0x7f23178b82e8>, 
# 'statistical_h_mean_window_size': <ray.tune.sample.Categorical object at 0x7f23178b8358>, 
# 'statistical_maximum_window_size': <ray.tune.sample.Categorical object at 0x7f23178b83c8>, 
# 'statistical_minimum_window_size': <ray.tune.sample.Categorical object at 0x7f23178b8438>, 
# 'feature_analysis': <ray.tune.sample.Categorical object at 0x7f23178b84a8>, 
# 'pyod_ae_dropout_rate': <ray.tune.sample.Categorical object at 0x7f23178b8518>, 
# 'pyod_loda_n_bins': <ray.tune.sample.Categorical object at 0x7f23178b8588>, 
# 'detection_algorithm': <ray.tune.sample.Categorical object at 0x7f23178b8668>}




# ['moving_average_transform']
# ['statistical_h_mean', 
# 'statistical_maximum', 
# 'statistical_minimum', 
# 'statistical_h_mean statistical_maximum', 
# 'statistical_h_mean statistical_minimum', 
# 'statistical_maximum statistical_h_mean', 
# 'statistical_maximum statistical_minimum', 
# 'statistical_minimum statistical_h_mean', 
# 'statistical_minimum statistical_maximum', 
# 'statistical_h_mean statistical_maximum statistical_minimum', 
# 'statistical_h_mean statistical_minimum statistical_maximum', 
# 'statistical_maximum statistical_h_mean statistical_minimum', 
# 'statistical_maximum statistical_minimum statistical_h_mean', 
# 'statistical_minimum statistical_h_mean statistical_maximum', 
# 'statistical_minimum statistical_maximum statistical_h_mean']
# ['pyod_ae', 
# 'pyod_loda', 
# 'pyod_ae pyod_loda', 
# 'pyod_loda pyod_ae']