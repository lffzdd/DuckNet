import numpy as np
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, accuracy_score

# import tensorflow.keras.backend as K

a = [[1, 0, 1, 0, 1],
     [1, 0, 1, 0, 1],
     [1, 0, 1, 0, 1]]
b = [[1, 0, 1, 0, 0],
     [1, 0, 1, 0, 1],
     [1, 0, 1, 0, 1]]
c = [[1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1]]

a = np.array(a, dtype=bool)
b = np.array(b)
c=np.array(c)
b = b > 0.5
# print(a)
print(b)
a = np.ndarray.flatten(a)
b = np.ndarray.flatten(b)
c = np.ndarray.flatten(c)
# a = K.flatten(a)
# b = K.flatten(b)
f1 = f1_score(a, b)  # 28÷30
print(f1)

miou = jaccard_score(a, b)
print(miou)  # 前景交并比，只算前景，即像素1，所以是8÷9

precision = precision_score(a, b)
print(precision)  # 即预测是正例的结果中，确实是正例的比例。

recall = recall_score(a, b)
print(recall)

accuracy=accuracy_score(a,b)
print(accuracy)