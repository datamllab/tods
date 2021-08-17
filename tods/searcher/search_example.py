from searcher import RaySearcher, datapath_to_dataset, json_to_searchspace


config = {
  "searching_algorithm": 'hyperopt',
  "num_samples": 15,
}

dataset = datapath_to_dataset('../../datasets/anomaly/kpi/kpi_dataset/tables/learningData.csv', 3)

dataset = datapath_to_dataset('../../yahoo_sub_5.csv', 6)

search_space = json_to_searchspace('test2.json')

s = RaySearcher(dataset, 'F1_MACRO')

print(s.search(search_space=search_space,
        config=config))

