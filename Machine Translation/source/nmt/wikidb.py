#-*- coding=utf-8 -*-
'''
Wikipedia mention-entity pair database
'''

import os
import codecs
import numpy
import theano

global_vocab = {}

def load_data(path, vocab = None):
    '''
    Load train and test datasets. Extract validation set from the train set with given valid_portion if valid_portion >0
    '''
    global global_vocab

    dataset = [];
    print("loading dataset...")
    with codecs.open(path,"r",'UTF-8') as f:
        for line in f:
            line = line.strip()
            array = line.split('\t')
            try:
                dataset.append((array[0],array[1]))
            except:
                print(line.decode('UTF-8'))
    print("done with {0} samples".format(len(dataset)))
    if vocab is None:
        global_vocab = vocab = extract_word_table(dataset)                   # extract word table
    else:
        global_vocab = vocab
    dataset = convert_data(dataset, vocab)             # convert data format
    return zip(*dataset)

def divide_data(dataset, portion = 0.1):
    dataset = zip(*dataset)
    idx = numpy.random.permutation(len(dataset))
    n_train = int(numpy.round(len(dataset) * (1. - portion)))
    valid_set = [dataset[s] for s in idx[n_train:]]
    temp = [dataset[s] for s in idx[:n_train]]
    train_set = temp
    return zip(*train_set), zip(*valid_set)

def extract_word_table(train_set):
    '''
    Extract word table
    '''
    word_set = set()
    for input, output in train_set:
        for word in input:
            word_set.add(word)
        for word in output:
            word_set.add(word)
    word_list = list(word_set)
    vocab = dict(zip(word_list,range(len(word_list))))
    vocab["unk"] = len(vocab)
    return vocab

def get_vocab():
    global global_vocab
    return global_vocab

def convert_data(dataset, vocab):
    '''
    Convert string type input-output into numerical representation
    '''
    integer_dataset = []
    for input, output in dataset:
        input_seq = []
        for word in input:
            if vocab.has_key(word):
                input_seq.append(vocab[word])
            else:
                input_seq.append(vocab["unk"])
        output_seq = []
        for word in output:
            if vocab.has_key(word):
                output_seq.append(vocab[word])
            else:
                output_seq.append(vocab["unk"])
        integer_dataset.append((input_seq, output_seq))
    return integer_dataset

def prepare_data(dataset):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    return: zip(intputs, input_mask), zip(outputs, output_mask)
        with inputs and input_mask of n_sample*max_length_of_input_sequence
        and  outpus and output_mask of n_sample*max_length_of_output_sequence
    """
    # x: a list of sentences
    input_lengths = [len(s) for s in dataset[0]]
    output_lengths = [len(s) for s in dataset[1]]

    n_samples = len(dataset[0])
    input_maxlen = numpy.max(input_lengths)
    output_maxlen = numpy.max(output_lengths)

    inputs = numpy.zeros((n_samples,input_maxlen)).astype('int64')
    input_mask = numpy.zeros((n_samples,input_maxlen)).astype(theano.config.floatX)
    for idx, s in enumerate(dataset[0]):
        inputs[idx,:input_lengths[idx]] = s
        input_mask[idx,:input_lengths[idx]] = 1.
    outputs = numpy.zeros((n_samples,output_maxlen)).astype('int64')
    output_mask = numpy.zeros((n_samples,output_maxlen)).astype(theano.config.floatX)
    for idx, s in enumerate(dataset[1]):
        outputs[idx, :output_lengths[idx]] = s
        output_mask[idx, :output_lengths[idx]] = 1.
    return inputs.T, input_mask.T, outputs.T, output_mask.T   

