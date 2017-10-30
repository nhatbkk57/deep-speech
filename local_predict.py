"""
Test a trained speech model over a dataset
"""

# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import tensorflow as tf
from new_data_generator import DataGenerator
import sys
#import kenlm
from char_map import index_map
from keras.models import Model,model_from_json
from keras.layers import (BatchNormalization, Convolution1D,Convolution2D, Dense,Activation,
                          Input, GRU,LSTM, Bidirectional, TimeDistributed,Lambda)
import time
import keras.backend as K
from utils import load_model
import math
#import correction
from keras.regularizers import l2
from keras.constraints import maxnorm


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def compile_output_fn(model):

    acoustic_input = model.inputs[0]
    network_output = model.outputs[0]
    
    output_fn = K.function([acoustic_input, K.learning_phase()],
                           [network_output])
    return output_fn

def relu20(x):
    return K.relu(x,max_value=20)
def compile_gru_model(input_dim=161, output_dim=91, recur_layers=7):
    
    """ Build a recurrent network (CTC) for speech with GRU units """
    print("Building gru model")
    # Main acoustic input
    acoustic_input = Input(shape=(None, input_dim,1), name='acoustic_input')
    conv2d_1 = Convolution2D(32, kernel_size = (11,41), strides=(2, 2), padding='valid',
                            data_format="channels_last")(acoustic_input)
    conv2d_1 = BatchNormalization(name='bn_conv1_2d')(conv2d_1)
    conv2d_1 =  Activation(relu20,name='conv1_relu20')(conv2d_1)

    
    
    conv2d_2 = Convolution2D(32, kernel_size = (11,21),strides=(1, 2), padding='valid',
                            data_format="channels_last")(conv2d_1)
    conv2d_2= BatchNormalization(name='bn_conv2_2d')(conv2d_2)
    conv2d_2 =  Activation(relu20, name='conv2_relu20')(conv2d_2)

        
    
    conv2d_3 = Convolution2D(96, kernel_size = (11,21),strides=(1, 2), padding='valid',
                            data_format="channels_last")(conv2d_2)
    conv2d_3 = BatchNormalization(name='bn_conv3_2d')(conv2d_3)
    conv2d_3 =  Activation(relu20,name='conv3_relu20')(conv2d_3)

    output = Lambda(function= lambda x : K.squeeze(x,axis=2))(conv2d_3)
    for r in range(recur_layers):
        output = Bidirectional(LSTM(units=768, return_sequences= True,activation='tanh',
                                    implementation=2,
                                    name='blstm_{}'.format(r+1)))(output)
        output = BatchNormalization(name='bn_rnn_{}'.format(r + 1))(output)
        

    network_output = TimeDistributed(Dense(output_dim,activation="linear",name="output"))(output)
    model = Model(inputs=acoustic_input,outputs=network_output)
    return model
def compile_asr_model(input_dim=161,output_dim=91,recur_layers=7):
    
    acoustic_input = Input(shape=(None, input_dim,1), name='acoustic_input')
    conv2d_1 = Convolution2D(32, kernel_size = (11,41), strides=(2, 2), padding='valid',
                            data_format="channels_last", kernel_constraint=maxnorm(5,axis=[0,1,2]),activation=relu20)(acoustic_input)
    conv2d_1 = BatchNormalization(name='bn_conv1_2d')(conv2d_1)

    
    
    conv2d_2 = Convolution2D(32, kernel_size = (11,21),strides=(1, 2), padding='valid',
                            data_format="channels_last", kernel_constraint=maxnorm(5,axis=[0,1,2]),activation=relu20)(conv2d_1)
    conv2d_2= BatchNormalization(name='bn_conv2_2d')(conv2d_2)

        
    
    conv2d_3 = Convolution2D(96, kernel_size = (11,21),strides=(1, 2), padding='valid',
                            data_format="channels_last", kernel_constraint=maxnorm(5,axis=[0,1,2]),activation=relu20)(conv2d_2)
    conv2d_3 = BatchNormalization(name='bn_conv3_2d')(conv2d_3)


    output = Lambda(function= lambda x : K.squeeze(x,axis=2))(conv2d_3)
    for r in range(recur_layers):
        output = Bidirectional(LSTM(units=768, return_sequences= True,activation='tanh',
                                implementation=2,
                                recurrent_regularizer=l2(0.01), kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01),
                                name='blstm_{}'.format(r+1)))(output)
        output = BatchNormalization(name='bn_rnn_{}'.format(r + 1))(output)
        

    network_output = TimeDistributed(Dense(output_dim,activation="linear",name="output"))(output)
    model = Model(inputs=acoustic_input,outputs=network_output)
    model.summary()
    return model
def argmax_decode(prediction):
    """ Decode a prediction using the highest probable character at each
        timestep. Then, simply convert the integer sequence to text
    Params:
        prediction (np.array): timestep * num_characters
    """
    int_sequence = []
    for timestep in prediction:
        int_sequence.append(np.argmax(timestep))
    tokens = []
    c_prev = -1
    count_blank = 0
    #print(int_sequence)
    for c in int_sequence:
        
        if c == c_prev:
            if c == 1:
                count_blank += 1
                if count_blank == 20:
                    tokens.append(91)
                    count_blank = 0
            continue
        if c != 0:  # Blank
            tokens.append(c)
            count_blank = 0

        c_prev = c

    text = ''.join([index_map[i] for i in tokens])
    return text

def main():
    # Prepare the data generator
    datagen = DataGenerator()

    print("fit done")
    b = time.time()

    model = compile_gru_model()
    model_weights_file = 'model_final.h5'
	    
    model.load_weights(model_weights_file)

    # # Compile the testing function
    # print("Load model done")
    test_fn = compile_output_fn(model)                    

    # Test the model
    while(True):	
        test_desc_file = input("data : ")
    
        if test_desc_file != '':
            file = str(test_desc_file) + ".wav"
            a = time.time()

            inputs = np.array([datagen.featurize(file)[...,np.newaxis]],dtype=np.float32)
            print(inputs.shape)
            output2 = np.squeeze(test_fn([inputs,True])[0])

            pre_arg2 = argmax_decode(output2)
            # print("argmax : {}".format(pre_arg))
            print("argmax 2 : {}".format(pre_arg2))
            print("predict done {}".format(time.time() - a))


    
if __name__ == '__main__':
    main()
