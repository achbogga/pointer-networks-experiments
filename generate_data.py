"""
generate sort data for training
format: x (input_sequence): 4,5,1
        y (output_argsort): 1,2,0
        for generating output_sequnce: x[y]
"""

import numpy as np
import random

def generate_sort_data(n_steps, n_examples, upper_limit):
	arange = upper_limit
	# no repeating numbers within a sequence
	x = np.arange( arange ).reshape( 1, -1 ).repeat( n_examples, axis = 0 )
	x = np.apply_along_axis( np.random.permutation, 1, x )
	x = x[:,:n_steps]
	y = np.argsort( x, axis = 1 )
	return x, y

def generate_x_y_for_inference(input_sequence, n_steps):
	x = np.array(input_sequence).reshape( -1, n_steps)
	y = np.argsort( x, axis = 1 )
	return x, y