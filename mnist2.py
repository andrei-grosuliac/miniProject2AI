# from sklearn import datasets
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.metrics import precision_score, recall_score
# import matplotlib.pyplot as plt


# digits = datasets.load_digits() # features matrix
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))

# X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.3, shuffle=False)


# mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=1e-4,
# solver='sgd', verbose=10, random_state=1,
# learning_rate_init=0.001)
# mlp.fit(X_train, y_train)


# y_pred = mlp.predict(X_test)
# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# pre_macro = precision_score(y_test, y_pred, average='macro')
# pre_micro = precision_score(y_test, y_pred, average='micro')
# recall_macro = recall_score(y_test, y_pred, average='macro')
# recall_micro = recall_score(y_test, y_pred, average='micro')


# cm = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(cm, display_labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).plot()
# plt.show()

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
digits = datasets.load_digits() # features matrix
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
Y = digits.target
kf = KFold(n_splits=10)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=1e-4,
solver='sgd', verbose=10, random_state=1,
learning_rate_init=0.001)
for train_index, test_index in kf.split(X, Y):
    x_train_fold = X[train_index]
    y_train_fold = Y[train_index]
    x_test_fold = X[test_index]
    y_test_fold = Y[test_index]
    mlp.fit(x_train_fold, y_train_fold)
    print(mlp.score(x_test_fold, y_test_fold))
