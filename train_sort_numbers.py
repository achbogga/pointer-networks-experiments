import numpy as np
from keras.layers import LSTM, Input
from keras.models import Model
from keras.utils.np_utils import to_categorical

from generate_data import generate_sort_data, generate_x_y_for_inference
from PointerLSTM import PointerLSTM
import argparse
from sklearn.metrics import accuracy_score

# Instantiate the parser
parser = argparse.ArgumentParser(description='Train LSTM_encoder+PointerLSTM_decoder for sorting numbers')

# Required positional argument
parser.add_argument('n_steps', type=int,
                    help='Sequence length (recommended: 5)')

# Optional argument
parser.add_argument('--n_examples', type=int, default=10000,
                    help='n_examples (recommended: 10000)')

# Optional argument
parser.add_argument('--upper_limit', type=int, default=5,
                    help='upper_limit of the input data (recommended: 5)')

# Optional argument
parser.add_argument('--epochs', type=int, default=10,
                    help='no_of_epochs to be trained for (recommended: 10)')

# Optional argument
parser.add_argument('--test_sequence', type=int, nargs='+', default=[3, 2, 1, 0, 4],
                    help='test_sequence to view predicted output sequence')


args = parser.parse_args()

n_steps = args.n_steps
upper_limit = args.upper_limit
n_examples = args.n_examples
epochs = args.epochs
split_at = int(.9*n_examples)
batch_size = 100

hidden_size = 64
# weights_file = 'model_weights/achbogga_model_weights_{}_steps_{}.hdf5'.format(n_steps, hidden_size)

x, y = generate_sort_data(n_steps, n_examples, upper_limit)
x = np.expand_dims(x, axis=2)
# prep position indicators

n = np.arange(n_steps)
nn = np.tile(n, (x.shape[0], 1))
nn = np.expand_dims(nn, axis=2)

xx = np.dstack((x, nn))

YY = []
for y_ in y:
    YY.append(to_categorical(y_))
YY = np.asarray(YY)

x_train = xx[:split_at]
x_test = xx[split_at:]

y_test = y[split_at:]
YY_train = YY[:split_at]
YY_test = YY[split_at:]

assert (n_steps == x_train.shape[1])
n_features = x_train.shape[2]

print("building model...")
main_input = Input(shape=(n_steps, n_features), name='main_input')

encoder = LSTM(output_dim=hidden_size, return_sequences=True, name="encoder")(main_input)
decoder = PointerLSTM(hidden_size, output_dim=hidden_size, name="decoder")(encoder)

model = Model(input=main_input, output=decoder)

# print(("loading weights from {}...".format(weights_file)))
# try:
#     model.load_weights(weights_file)
# except IOError:
#     print("no weights file, starting anew.")

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('training and saving model weights each epoch...')

validation_data = (x_test, YY_test)

history = model.fit(x_train, YY_train, epochs=epochs, batch_size=batch_size,
                    validation_data=validation_data)

# p = model.predict(x_test)
# print("test accuracy: ", accuracy_score(YY_test, p))



x, y = generate_x_y_for_inference(args.test_sequence, n_steps)
x = np.expand_dims(x, axis=2)
assert (n_steps == x.shape[1])

n = np.arange(n_steps)
nn = np.tile(n, (x.shape[0], 1))
nn = np.expand_dims(nn, axis=2)

xx = np.dstack((x, nn))

p = model.predict(xx)

print ("input_sequence: ", xx[0][:,0].flatten())

for i in range(len(xx)):
    y_true = xx[i][:,0][y].flatten()
    y_pred = xx[i][:,0][p.argmax(axis=1)].flatten()
    print(("ground_truth:", y_true))
    print(("predicted_sequence:     ", y_pred))
    print("accuracy: ", accuracy_score(y_true, y_pred))
