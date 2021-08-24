#!/usr/bin/env python

"order integer sequences of length given by n_steps"

import numpy as np
from keras.layers import LSTM, Input
from keras.models import Model
from keras.utils.np_utils import to_categorical

from generate_data import generate_sort_data
from PointerLSTM import PointerLSTM
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Train LSTM_encoder+PointerLSTM_decoder for sorting numbers')

# Required positional argument
parser.add_argument('n_steps', type=int,
                    help='Length of the sequence to be trained (recommended: 5)')

# Optional positional argument
parser.add_argument('n_examples', type=int, nargs='?',
                    help='An optional integer positional argument')

# Optional argument
parser.add_argument('--opt_arg', type=int,
                    help='An optional integer argument')

# Switch
parser.add_argument('--switch', action='store_true',
                    help='A boolean switch')

args = parser.parse_args()

print(args.pos_arg)
print(args.opt_pos_arg)
print(args.opt_arg)
print(args.switch)

n_steps = 5

n_examples = 10000
split_at = 9000
batch_size = 100

hidden_size = 64
weights_file = 'model_weights/model_weights_{}_steps_{}.hdf5'.format(n_steps, hidden_size)

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

#

print("building model...")
main_input = Input(shape=(n_steps, 1), name='main_input')

encoder = LSTM(units=hidden_size, return_sequences=True, name="encoder")(main_input)
print(encoder)
decoder = PointerLSTM(hidden_size, units=hidden_size, name="decoder")(encoder)

model = Model(inputs=main_input, outputs=decoder)

print(("loading weights from {}...".format(weights_file)))
try:
    model.load_weights(weights_file)
except IOError:
    print("no weights file, starting anew.")

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('training and saving model weights each epoch...')

validation_data = (x_test, YY_test)


epoch_counter = 0

history = model.fit(x_train, YY_train, epochs=1, batch_size=batch_size,
                    validation_data=validation_data)

p = model.predict(x_test)

for y_, p_ in list(zip(y_test, p))[:5]:
    print(("epoch_counter: ", epoch_counter))
    print(("y_test:", y_))
    print(("p:     ", p_.argmax(axis=1)))
    print()

model.save(weights_file)
