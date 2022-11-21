#this file generates the _main.py files to test the primitives
import re
import os
import sys
#sys.path.insert(0, 'tods/utils/skinterface')
print(sys.path)
with open('../utils/entry_points/entry_points_feature_analysis.txt','r',encoding='utf-8') as f:
    entry_file = f.read()

output_dir = '../test/feature_analysis'   #output directory

fit_available_primitives = ['SKTruncatedSVDPrimitive']

primitive_folder_start_loc_buf = [i.start()+2 for i in re.finditer('=', entry_file)]
primitive_start_loc_buf = [i.start()+1 for i in re.finditer(':', entry_file)]
primitive_end_loc_buf = [i.start() for i in re.finditer('\n', entry_file)]

for primitive_index, primitive_start_loc in enumerate(primitive_start_loc_buf):

    primitive_folder_start_loc = primitive_folder_start_loc_buf[primitive_index]
    primitive_end_loc = primitive_end_loc_buf[primitive_index]

    primitive_folder = entry_file[primitive_folder_start_loc:primitive_start_loc-1]
    primitive_name = entry_file[primitive_start_loc:primitive_end_loc]
    algorithm_name = primitive_name.replace('Primitive', '')

    # print(entry_file[primitive_folder_start_loc:primitive_start_loc-1])
    # print(entry_file[primitive_start_loc:primitive_end_loc])

    primitve_api_name = primitive_name.replace('Primitive', '_skinterface')
    class_name = primitive_name.replace('Primitive', 'SKI')

    import_line1 = """import numpy as np
import pandas as pd
import os
""" + 'from tods.sk_interface.feature_analysis.' + primitve_api_name + ' import ' + class_name + '\n\n'

    import_line2 = """from pyod.utils.data import generate_data
import unittest
from sklearn.metrics import roc_auc_score\n
"""

    main_line1 = 'class ' + algorithm_name + 'SKI_TestCase(unittest.TestCase):\n' + \
"""    def setUp(self):

        self.n_train = 200
        self.n_test = 100
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
             n_train=self.n_train, n_test=self.n_test, n_features=5,
             contamination=0., random_state=42)\n
"""

    main_line2 = '        self.transformer = ' + class_name + '()\n'
    if primitive_name in fit_available_primitives:
        main_line3 = '        self.transformer.fit(self.X_train)\n\n'
    else:
        main_line3 = '\n'

    main_line4 = """    def test_produce(self):
        X_transform = self.transformer.produce(self.X_test)
        


if __name__ == '__main__':
    unittest.main()
"""

    python_content = import_line1 + import_line2 + main_line1+main_line2+main_line3+main_line4
    python_name = 'test_ski_' + algorithm_name + '.py'
    
    with open(os.path.join(output_dir, python_name), 'w', encoding='utf-8') as f:
        f.write(python_content)
    print(os.path.join(output_dir, python_name))
    print(python_content)



