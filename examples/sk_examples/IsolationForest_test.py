import numpy as np
from tods.sk_interface.detection_algorithm.IsolationForest_skinterface import IsolationForestSKI
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#prepare the data

data = np.loadtxt("./500_UCR_Anomaly_robotDOG1_10000_19280_19360.txt")

# print("shape:", data.shape)
# print("datatype of data:",data.dtype)
# print("First 5 rows:\n", data[:5])

X_train = np.expand_dims(data[:10000], axis=1)
X_test = np.expand_dims(data[10000:], axis=1)

# print("First 5 rows train:\n", X_train[:5])
# print("First 5 rows test:\n", X_test[:5])

transformer = IsolationForestSKI()
transformer.fit(X_train)
prediction_labels_train = transformer.predict(X_train)
prediction_labels = transformer.predict(X_test)
prediction_score = transformer.predict_score(X_test)

print("Prediction Labels\n", prediction_labels)
print("Prediction Score\n", prediction_score)

y_true = prediction_labels_train
y_pred = prediction_labels

print('Accuracy Score: ', accuracy_score(y_true, y_pred))

confusion_matrix(y_true, y_pred)

print(classification_report(y_true, y_pred))
