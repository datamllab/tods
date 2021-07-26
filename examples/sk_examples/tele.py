import numpy as np
from tods.sk_interface.detection_algorithm.Telemanom_skinterface import TelemanomSKI
# from tods.tods_skinterface.primitiveSKI.detection_algorithm.Telemanom_skinterface import TelemanomSKI
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
#prepare the data

data = pd.read_csv("./yahoo_sub_5.csv").to_numpy()
# print("shape:", data.shape)
# print("datatype of data:",data.dtype)
# print("First 5 rows:\n", data[:5])

# X_train = np.expand_dims(data[:10000], axis=1)
# X_test = np.expand_dims(data[10000:], axis=1)

# print("First 5 rows train:\n", X_train[:5])
# print("First 5 rows test:\n", X_test[:5])

transformer = TelemanomSKI(l_s= 2, n_predictions= 1)
transformer.fit(data)
# prediction_labels_train = transformer.predict(X_train)
prediction_labels = transformer.predict(data)
prediction_score = transformer.predict_score(data)

# print("Primitive: ", transformer.primitive)
print("Prediction Labels\n", prediction_labels)
print("Prediction Score\n", prediction_score)