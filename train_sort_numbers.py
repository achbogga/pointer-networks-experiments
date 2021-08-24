import numpy as np
from keras.layers import LSTM, Input
from keras.models import Model
from keras.utils.np_utils import to_categorical

from generate_data import generate_sort_data, generate_x_y_for_inference
from PointerLSTM import PointerLSTM
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Train LSTM_encoder+PointerLSTM_decoder for sorting numbers')

# Required positional argument
parser.add_argument('n_steps', type=int,
                    help='Sequence length (recommended: 5)')

# Optional argument
parser.add_argument('--n_examples', type=int, default=10000,
                    help='n_examples (recommended: 10000)')

# Optional argument
parser.add_argument('--upper_limit', type=int, default=10,
                    help='upper_limit of the input data (recommended: 10)')

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
YY = []
for y_ in y:
    YY.append(to_categorical(y_))
YY = np.asarray(YY)
x_train = x[:split_at]
x_test = x[split_at:]
y_test = y[split_at:]
YY_train = YY[:split_at]
YY_test = YY[split_at:]
assert (n_steps == x.shape[1])

print("building model...")
main_input = Input(shape=(n_steps, 1), name='main_input')

encoder = LSTM(units=hidden_size, return_sequences=True, name="encoder")(main_input)
# print(encoder)
decoder = PointerLSTM(hidden_size, units=hidden_size, name="decoder")(encoder)

model = Model(inputs=main_input, outputs=decoder)

# print(("loading weights from {}...".format(weights_file)))
# try:
#     model.load_weights(weights_file)
# except IOError:
#     print("no weights file, starting anew.")

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# print('training and saving model weights each epoch...')

validation_data = (x_test, YY_test)

epoch_counter = 0

history = model.fit(x_train, YY_train, epochs=epochs, batch_size=batch_size,
                    validation_data=validation_data)

# p = model.predict(x_test)

# for y_, p_ in list(zip(y_test, p))[:5]:
#     print(("epoch_counter: ", epoch_counter))
#     print(("y_test:", y_))
#     print(("p:     ", p_.argmax(axis=1)))
#     print()

x, y = generate_x_y_for_inference(args.test_sequence, n_steps)
x = np.expand_dims(x, axis=2)
assert (n_steps == x.shape[1])

p = model.predict(x)

for i in range(len(x)):
    print(("ground_truth:", x[i][y]))
    print(("predicted_sequence:     ", x[i][p.argmax(axis=1)]))
    print()
