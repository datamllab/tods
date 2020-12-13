import numpy as np
from test_interface import SKInterface
from tods.detection_algorithm.AutoRegODetect import AutoRegODetectorPrimitive

class AutoRegODetect(SKInterface):
    def __init__(self, **hyperparams):
        super().__init__(primitive=AutoRegODetectorPrimitive, **hyperparams)

    


if __name__ == '__main__':
    X_train = np.array([[3., 4., 8., 16, 18, 13., 22., 36., 59., 128, 62, 67, 78, 100]])
    X_test = np.array([[3., 4., 8.6, 13.4, 22.5, 17, 19.2, 36.1, 127, -23, 59.2]])

    transformer = AutoRegODetect(contamination=0.2, window_size=2)#use_column=(1,)
    transformer.fit(X_train)
    prediction_labels = transformer.produce(X_test)
    prediction_score = transformer.produce_score(X_test)

    print("Prediction Labels\n", prediction_labels)
    print("Prediction Score\n", prediction_score)

