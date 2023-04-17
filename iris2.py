from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt


iris = load_iris() # features matrix

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)


mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, solver='sgd', random_state=1, learning_rate_init=0.001, verbose=10, alpha=1e-4)
mlp.fit(X_train, y_train)


y_pred = mlp.predict(X_test)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

pre_macro = precision_score(y_test, y_pred, average='macro')
pre_micro = precision_score(y_test, y_pred, average='micro')
recall_macro = recall_score(y_test, y_pred, average='macro')
recall_micro = recall_score(y_test, y_pred, average='micro')

print('pre_macro: %.2f' % pre_macro)
print('pre_micro: %.2f' % pre_micro)
print('recall_macro: %.2f' % recall_macro)
print('recall_micro: %.2f' % recall_micro)


cm = confusion_matrix(y_test, y_pred)
print(cm)
ConfusionMatrixDisplay(cm, display_labels=[1, 2, 3]).plot()
plt.show()