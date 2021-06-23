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

df1 = pd.DataFrame(prediction_labels)
df2 = pd.DataFrame(prediction_score)

# df1.to_csv(r'./labels.csv', index = False)
df2.to_csv(r'./scores.csv', index = False)
# result = pd.merge(df1, df2[[]])
# result = [prediction_labels, prediction_score]
# # result = pd.DataFrame({'label': prediction_labels, 'score': prediction_score}, columns=['label', 'score'], index=[0])
# print(result)
# pd.DataFrame(result).to_csv("./teleSKI.csv")
# y_true = prediction_labels_train
# y_pred = prediction_labels

# print('Accuracy Score: ', accuracy_score(y_true, y_pred))

# confusion_matrix(y_true, y_pred)

# print(classification_report(y_true, y_pred))

# precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
# f1_scores = 2*recall*precision/(recall+precision)

# print('Best threshold: ', thresholds[np.argmax(f1_scores)])
# print('Best F1-Score: ', np.max(f1_scores))

# fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
# roc_auc = metrics.auc(fpr, tpr)

# plt.title('ROC')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
