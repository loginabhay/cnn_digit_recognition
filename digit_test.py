import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = np.array(x_test).reshape(-1,28,28,1).astype('float32')

ex_model = tf.keras.models.load_model('digit.h5py')
predicted_classes = ex_model.predict(x_test)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
print(predicted_classes.shape, y_test.shape)
correct = np.where(predicted_classes == y_test)[0]
print('found correct labels:',len(correct))
for i, correct in enumerate(correct[:16]):
    plt.subplot(4,4,i+1)
    plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title('predicted {}, class {}'.format(predicted_classes[correct], y_test[correct]))
    plt.tight_layout()
plt.show()

wrong = np.where(predicted_classes != y_test)[0]
print('found wrong labels:',len(wrong))
for i, wrong in enumerate(wrong[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[wrong].reshape(28,28), cmap='gray', interpolation='none')
    plt.title('predicted {}, class {}'.format(predicted_classes[wrong], y_test[wrong]))
    plt.tight_layout()

plt.show()
num_classes = 10
target_names = ["Classe {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))