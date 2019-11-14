import sys
import keras
from keras.datasets import mnist
from keras.models import load_model
from keras import backend as K

if (sys.argv[0] == None or sys.argv[0] == '' ):
    print("\nError! No model file inputted")
    print("\nUsage: 'python testing.py [Model File]'")
    exit()

model = load_model('model.h5')

if ( model == None ):
    print("\nError! Model is not Found")
    exit()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test[:10000]
y_test = y_test[:10000]
img_rows = 28
img_cols = 28
num_clases = 10

if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_test = x_test.reshape (x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255
print(x_test.shape[0], 'test samples')

y_test = keras.utils.to_categorical(y_test, num_clases)


score = model.evaluate(x_test, y_test, verbose = 0)
print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])
