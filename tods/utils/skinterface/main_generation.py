#this file generates the _main.py files to test the primitives
import re
import os
import sys
#sys.path.insert(0, 'tods/utils/skinterface')
print(sys.path)
with open('tods/utils/skinterface/entry_points.txt','r',encoding='utf-8') as f:
    entry_file = f.read()

output_dir = 'tods/utils'   #output directory


primitive_folder_start_loc_buf = [i.start()+2 for i in re.finditer('=', entry_file)]
primitive_start_loc_buf = [i.start()+1 for i in re.finditer(':', entry_file)]
primitive_end_loc_buf = [i.start() for i in re.finditer('\n', entry_file)]

for primitive_index, primitive_start_loc in enumerate(primitive_start_loc_buf):

    primitive_folder_start_loc = primitive_folder_start_loc_buf[primitive_index]
    primitive_end_loc = primitive_end_loc_buf[primitive_index]

    primitive_folder = entry_file[primitive_folder_start_loc:primitive_start_loc-1]
    primitive_name = entry_file[primitive_start_loc:primitive_end_loc]
    # print(entry_file[primitive_folder_start_loc:primitive_start_loc-1])
    # print(entry_file[primitive_start_loc:primitive_end_loc])

    primitve_api_name = primitive_name.replace('Primitive', '_skinterface')
    class_name = primitive_name.replace('Primitive', 'SKI')
# import sys
# sys.path.insert(0, 'tods/utils')"""
    import_line1 = 'import numpy as np' 

    import_line2 = '\nfrom skinterface.primitiveSKI.detection_algorithm.'+ primitve_api_name + ' import ' + class_name + '\n\n'
    #print(import_line)

    main_line1 = """X_train = np.array([[3., 4., 8., 16, 18, 13., 22., 36., 59., 128, 62, 67, 78, 100]])
X_test = np.array([[3., 4., 8.6, 13.4, 22.5, 17, 19.2, 36.1, 127, -23, 59.2]])\n
"""
    main_line2 = 'transformer = ' + class_name + '()'
    main_line3 = """
transformer.fit(X_train)
prediction_labels = transformer.predict(X_test)
prediction_score = transformer.predict_score(X_test)

print("Primitive: ", transformer.primitive)
print("Prediction Labels\\n", prediction_labels)
print("Prediction Score\\n", prediction_score)
"""

    python_content = import_line1 + import_line2 + main_line1+main_line2+main_line3
    python_name = primitive_name.replace('Primitive', '_main.py')
    
    with open(os.path.join(output_dir, python_name), 'w', encoding='utf-8') as f:
        f.write(python_content)
    print(os.path.join(output_dir, python_name))
    print(python_content)

"""
    main_line1 = 'X_train = np.array([[3., 4., 8., 16, 18, 13., 22., 36., 59., 128, 62, 67, 78, 100]])\n'
    main_line2 = 'X_test = np.array([[3., 4., 8.6, 13.4, 22.5, 17, 19.2, 36.1, 127, -23, 59.2]])\n\n'
    main_line3 = 'transformer = AutoRegODetectorSKI(contamination=0.2, window_size=2)\n'
    main_line4 = 'transformer.fit(X_train)\n'
    main_line5 = 'prediction_labels = transformer.predict(X_test)\n'
    main_line6 = 'prediction_score = transformer.predict_score(X_test)\n\n'
    main_line7 = 'print("Prediction Labels\n", prediction_labels)\n'
    main_line8 = 'print("Prediction Score\n", prediction_score)\n'
"""
# import numpy as np
# from test_interface import SKInterface
# from tods.detection_algorithm.AutoRegODetect import AutoRegODetectorPrimitive
#
# class AutoRegODetect(SKInterface):
# def __init__(self, **hyperparams):
# super().__init__(primitive=AutoRegODetectorPrimitive, hyperparams)


