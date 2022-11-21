import numpy as np
from tods.sk_interface.detection_algorithm.MatrixProfile_skinterface import MatrixProfileSKI
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import metrics

#prepare the data
data = np.loadtxt("./500_UCR_Anomaly_robotDOG1_10000_19280_19360.txt")

# first 10000 data points are training data
X_train = np.expand_dims(data[:10000], axis=1)
# second 10000 data points are testing data
X_test = np.expand_dims(data[10000:20000], axis=1)

# creating labels
y_test = np.zeros(10000).astype(np.int32)
y_test[9280:9360] = 1
y_test = np.expand_dims(y_test, axis=1)

# TODS pipeline
transformer = MatrixProfileSKI()
transformer.fit(X_train)

prediction_labels_train = transformer.predict(X_train)

prediction_labels = transformer.predict(X_test)
prediction_score = transformer.predict_score(X_test)

print("Prediction Labels\n", prediction_labels)
print("Prediction Score\n", prediction_score)

# set y_true to ground truth label information
y_true = y_test
# set y_pred to X_test pred labels
y_pred = prediction_labels

print("Check shapes:", y_true.shape, y_pred.shape)

print('Accuracy Score: ', accuracy_score(y_true, y_pred))

confusion_matrix(y_true, y_pred)

print(classification_report(y_true, y_pred))

precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
np.seterr(invalid='ignore')
f1_scores = 2*recall*precision/(recall+precision)

print('Best threshold: ', thresholds[np.argmax(f1_scores)])
print('Best F1-Score: ', np.max(f1_scores))

fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
