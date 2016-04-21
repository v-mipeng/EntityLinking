'''
Build a tweet sentiment analyzer
'''

from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import codecs

import wikidb

'''
h_t = tanh(W_hx*x+W_hh*h_t_+b_h)        # encoder
y_t = tanh(W_yv*h_T+W_yg*g_t_+b_y)      # decoder for y_t
g_t = tanh(W_gy*y_t+W_gg*g_t_+b_g)      # decoder for g_t
o_t = softmax(Emd*y_t)              # the output of time stamp t.
'''

model_options = {}
params = OrderedDict()
tparams = OrderedDict()

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)
def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

#region Datasets
datasets = {'wikidb': (wikidb.load_data, wikidb.divide_data, wikidb.prepare_data, wikidb.get_vocab)}


#endregion

#region Parameter Initialization

def init_params():
    """
    Initialize parameters:
    Emb: embedding parameters sample_number*proj_dim
    [W_hx,W_hh,W_yv,W_yg,W_gh,W_gg]: proj_dim*(6*proj_dim)
    """
    global params
    # initialize embedding parameters
    randn = numpy.random.rand(model_options['n_words'],
                              model_options['dim_proj'])
    p = (0.01 * randn).astype(config.floatX)
    if model_options['emb_file'] is not None:
       embs = load_emb(model_options['emb_file'])       # load embeding value from existing file
       if embs is not None:
           vocab = model_options['vocab']
           for word in embs.iterkeys():
               if vocab.has_key(word):
                   idx = vocab[word]
                   p[idx,:] = embs[word]

    params['Wemb'] = p   # initialize word embeding parameters
    
    # initialize rnn paramters
    p = numpy.concatenate([ortho_weight(model_options['dim_proj']),  #[W_hx,W_hh,W_yv,W_yg,W_gh,W_gg]
                           ortho_weight(model_options['dim_proj']),
                           ortho_weight(model_options['dim_proj']),
                           ortho_weight(model_options['dim_proj']),
                           ortho_weight(model_options['dim_proj']),
                           ortho_weight(model_options['dim_proj'])], axis=1) # shape: (128,512)
    params['Rnn_W'] = p
    b = numpy.zeros(3 * model_options['dim_proj'])
    params['Rnn_b'] = b.astype(config.floatX)

def load_emb(emb_file):
    embs = {}
    dim_proj = model_options['dim_proj']
    with codecs.open(emb_file,'r','UTF-8') as f:
        for line in f:
            array = line.split(' ')
            if len(array) != dim_proj+1:
                return None
            vector = []
            for i in range(1,len(array)):
                vector.append(float(array[i]))
            embs[array[0]] = numpy.asarray(vector,config.floatX)
    return embs

def load_params(path):
    global params
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

def init_tparams():
    global tparams
    for kk, pp in params.items():   # item: (parameter name, parameter value)
        tparams[kk] = theano.shared(params[kk], name=kk)    # initialize with the numerical values.

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

def load_model(model_path, option_path):
    global params
    global model_options

    args = numpy.load(model_path)
    params['Rnn_W'] = args['Rnn_W']
    params['Rnn_b'] = args['Rnn_b']
    params['Wemb'] = args['Wemb']
    args = pickle.load(open(option_path,"rb"))
    for item in args.items():
        model_options[item[0]] = item[1]

#endregion

#region Define Training Procedure

def build_model():
    trng = RandomStreams(SEED)
    global params
    # Used for dropout.
    x = tensor.matrix('x', dtype='int64')                           # input (tensor representation): input_len*dim_proj
    input_mask = tensor.matrix('input_mask', dtype=config.floatX)
    y = tensor.matrix('y', dtype='int64')                           # output: output_len*dim_proj
    output_mask = tensor.matrix('output_mask', dtype=config.floatX)

    input_timesteps = x.shape[0]                                    # maximum sequence length
    n_samples = x.shape[1]                                          # sample number

    input_embs = tparams['Wemb'][x.flatten()].reshape([input_timesteps,
                                                       n_samples,          # vector representaion of inputs
                                                       model_options['dim_proj']])
    output_timesteps = y.shape[0];
    outputs = encode_decode(input_embs = input_embs, input_mask = input_mask, output_mask = output_mask)
    pred_prob, updates = theano.scan(prob,
                            sequences = [outputs, output_mask, y],
                            outputs_info = tensor.ones([n_samples],theano.config.floatX),
                            non_sequences=tparams["Wemb"].T,
                            n_steps = output_timesteps
                            )
    pred_prob = pred_prob[-1]/output_mask.sum(axis=0)
    f_pred_prob = theano.function([x, input_mask, y, output_mask], pred_prob, name='f_pred_prob')

    off = 1e-8

    cost = -tensor.log(pred_prob + off).mean()  # cross-entropy
    
    return x, input_mask, y, output_mask, f_pred_prob, cost

def encode_decode(input_embs,input_mask, output_mask):
    
    global tparams
    global model_options

    W_hx = tparams['Rnn_W'][:,:model_options['dim_proj']]
    W_hh = tparams['Rnn_W'][:,model_options['dim_proj']:2*model_options['dim_proj']]
    W_yv = tparams['Rnn_W'][:,2*model_options['dim_proj']:3*model_options['dim_proj']]
    W_yg = tparams['Rnn_W'][:,3*model_options['dim_proj']:4*model_options['dim_proj']]
    W_gy = tparams['Rnn_W'][:,4*model_options['dim_proj']:5*model_options['dim_proj']]
    W_gg = tparams['Rnn_W'][:,5*model_options['dim_proj']:6*model_options['dim_proj']]
    b_h = tparams['Rnn_b'][:model_options['dim_proj']]
    b_y = tparams['Rnn_b'][model_options['dim_proj']:2*model_options['dim_proj']]
    b_g = tparams['Rnn_b'][2*model_options['dim_proj']:3*model_options['dim_proj']]


    def encode(m_, x_, h_):

        h=tensor.tanh(tensor.dot(h_ , W_hh)+x_)
        h = m_[:, None] * h + (1. - m_[:, None]) * h_
        return h

    def decode(g_,v):

        y = tensor.tanh(tensor.dot(g_ , W_yg) + v)
        g = tensor.tanh(tensor.dot(y , W_gy) + tensor.dot(g_ , W_gg) + b_g)
        return y, g

    dim_proj = model_options['dim_proj']
    n_samples = input_mask.shape[1]
    input_steps = input_mask.shape[0]
    input_states = tensor.dot(input_embs, W_hx)+b_h
    h_Ts, updates = theano.scan(encode,                                          # scan along the first dimension.
                                sequences=[input_mask, input_states],                  # state_below: Whx.X
                                outputs_info=[tensor.alloc(numpy_floatX(0.),    # h initialization
                                                           n_samples,
                                                           dim_proj,
                                                           )],
                                name='encode',
                                n_steps=input_steps) # run encoder on one all sequences.
    output_steps = output_mask.shape[0]
    v = tensor.dot(h_Ts[-1] , W_yv)+b_y
    outputs, updates = theano.scan(decode,                                       
                                outputs_info=[None,
                                              tensor.alloc(numpy_floatX(0.),     # g initialization
                                                           n_samples,
                                                           dim_proj)],
                                non_sequences=v,                 
                                name='encode',
                                n_steps=output_steps) 
    return outputs[0]           # return y_ts


def prob(o, m, y, p_, w):
    pred = tensor.nnet.softmax(tensor.dot(o,w))
    p = p=p_*pred[tensor.arange(p_.shape[0]), y]
    p = m * p + (1. - m) * p_
    return p

#endregion

#region Optimization (Parameter Update)

def adadelta(x, input_mask, y, output_mask, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """
    global tparams

    grads = tensor.grad(cost, wrt=list(tparams.values()))   # calculate symbolic gradient of cost on all the parameters

    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),      # initialize E[delta(x)^2]
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),   # initialize E[g^2]
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]
    
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))            # update E[g^2]
             for rg2, g in zip(running_grads2, grads)]

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([x, input_mask, y, output_mask], cost, updates=rg2up + ru2up + param_up,
                               name='adadelta_f_update')

    return  f_update

#endregion

#region Prediction

def load_database(database):
    id2index = {}
    kbp = []
    with codecs.open(data_base) as f:
        for line in f:
            array = line.split('\t')
            kbp.append(array[1])
            id2index[array[0]] = len(kbp)
    return id2index, kbp

def load_test_data(test_path):
    mentions = []
    ids =[]
    with codecs.open(test_path) as f:
        for line in f:
            array = line.split('\t')
            mentions.append(array[0])
            ids.append(array[1])
    return mentions, ids

def prepare_single_data(data_set, vocab):
    int_seqs = convert_single_data(data_set, vocab)
    n_samples = len(data_set)
    max_len = numpy.max(seq_lengths)
    aline_seqs = numpy.zeros((input_maxlen,n_samples)).astype('int64')
    mask = numpy.zeros((input_maxlen, n_samples)).astype(theano.config.floatX) 
    for idx, s in enumerate(int_seqs):
       aline_seqs[:seq_lengths[idx],idx] = s
       mask[:seq_lengths[idx],idx] = 1.
    return aline_seqs, mask

def convert_single_data(data_set, vocab):
    int_seqs = []
    seq_lengths=[]
    for piece in dataset:
        int_seq = []
        seq_lengths.append(len(piece))
        for word in piece:
            if vocab.has_key(word):
                int_seq.append(vocab[word])
            else:
                int_seq.append(vocab["unk"])
        int_seqs.append(int_seq)
    return int_seqs

    trng = RandomStreams(SEED)
    global params
    # Used for dropout.
    x = tensor.matrix('x', dtype='int64')                           # input (tensor representation): input_len*dim_proj
    input_mask = tensor.matrix('input_mask', dtype=config.floatX)
    y = tensor.matrix('y', dtype='int64')                           # output: output_len*dim_proj
    output_mask = tensor.matrix('output_mask', dtype=config.floatX)

    input_timesteps = x.shape[0]                                    # maximum sequence length
    n_samples = x.shape[1]                                          # sample number

    input_embs = tparams['Wemb'][x.flatten()].reshape([input_timesteps,
                                                       n_samples,          # vector representaion of inputs
                                                       model_options['dim_proj']])
    output_timesteps = y.shape[0];
    outputs = encode_decode(input_embs = input_embs, input_mask = input_mask, output_mask = output_mask)
    pred_prob, updates = theano.scan(prob,
                            sequences = [outputs, output_mask, y],
                            outputs_info = tensor.ones([n_samples],theano.config.floatX),
                            non_sequences=tparams["Wemb"].T,
                            n_steps = output_timesteps
                            )
    pred_prob = pred_prob[-1]/output_mask.sum(axis=0)
    f_pred_prob = theano.function([x, input_mask, y, output_mask], pred_prob, name='f_pred_prob')

    off = 1e-8

    cost = -tensor.log(pred_prob + off).mean()  # cross-entropy
    
    return x, input_mask, y, output_mask, f_pred_prob, cost

def build_single_model():
    trng = RandomStreams(SEED)
    global params
    # Used for dropout.
    input = tensor.matrix('input', dtype='int64')                           # input (tensor representation): input_len*dim_proj
    input_mask = tensor.matrix('input_mask', dtype=config.floatX)
    output = tensor.matrix('output', dtype='int64')                           # output: output_len*dim_proj
    output_mask = tensor.matrix('output_mask', dtype=config.floatX)

    input_timesteps = x.shape[0]                                    
    output_timesteps = y.shape[0];
    n_samples = x.shape[1]                                          # sample number

    hT, pred_prob = encode_pred(input, input_mask, output, output_mask)      # hT: n_sample*dim_proj; pred_prob: n_sample
      
    return input, input_mask, output, output_mask, hT, pred_prob

def encode_pred(input, input_mask, output, output_mask):
    
    global tparams
    global model_options
    W_hx = tparams['Rnn_W'][:,:model_options['dim_proj']]
    W_hh = tparams['Rnn_W'][:,model_options['dim_proj']:2*model_options['dim_proj']]
    W_yv = tparams['Rnn_W'][:,2*model_options['dim_proj']:3*model_options['dim_proj']]
    W_yg = tparams['Rnn_W'][:,3*model_options['dim_proj']:4*model_options['dim_proj']]
    W_gy = tparams['Rnn_W'][:,4*model_options['dim_proj']:5*model_options['dim_proj']]
    W_gg = tparams['Rnn_W'][:,5*model_options['dim_proj']:6*model_options['dim_proj']]
    b_h = tparams['Rnn_b'][:model_options['dim_proj']]
    b_y = tparams['Rnn_b'][model_options['dim_proj']:2*model_options['dim_proj']]
    b_g = tparams['Rnn_b'][2*model_options['dim_proj']:3*model_options['dim_proj']]


    def encode(m_, x_, h_):

        h=tensor.tanh(tensor.dot(h_ , W_hh)+x_)
        h = m_[:, None] * h + (1. - m_[:, None]) * h_
        return h

    def decode(output, m, p_, g_, v, w):

        y = tensor.tanh(tensor.dot(g_ , W_yg) + v)
        g = tensor.tanh(tensor.dot(y , W_gy) + tensor.dot(g_ , W_gg) + b_g)
        pred = tensor.nnet.softmax(tensor.dot(y,w))
        p = p_*pred[tensor.arange(p_.shape[0]), output]
        p = m * p + (1. - m) * p_
        return p_, g

    dim_proj = model_options['dim_proj']
    n_samples = input_mask.shape[1]
    input_steps = input_mask.shape[0]
    input_embs = tparams['Wemb'][input.flatten()].reshape([input_steps,
                                                       n_samples,          # vector representaion of inputs
                                                       model_options['dim_proj']])
    input_states = tensor.dot(input_embs, W_hx)+b_h
    h_Ts, updates = theano.scan(encode,                                          # scan along the first dimension.
                                sequences=[input_mask, input_states],                  # state_below: Whx.X
                                outputs_info=[tensor.alloc(numpy_floatX(0.),    # h initialization
                                                           n_samples,
                                                           dim_proj,
                                                           )],
                                name='encode',
                                n_steps=input_steps) # run encoder on one all sequences.
    output_steps = output_mask.shape[0]
    h_Ts = h_Ts[-1]
    v = tensor.dot(h_Ts , W_yv)+b_y
    pred_prob, updates = theano.scan(decode, 
                                sequences = [output, output_mask],                                      
                                outputs_info=[tensor.ones([n_samples],theano.config.floatX),
                                              tensor.alloc(numpy_floatX(0.),     # g initialization
                                                           n_samples,
                                                           dim_proj)],
                                non_sequences= [v,tparams["Wemb"].T],
                                name='encode',
                                n_steps=output_steps) 
    pred_prob = pred_prob[0][-1]/output_mask.sum(axis=0)
    return v, pred_prob

#endregion

#region Main

def train(
    dim_proj=100,  # word embeding dimension and LSTM number of hidden units.
    emb_file = "./input/wiki_zh/word embed.txt", # embedding file path
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=200,  # The maximum number of epoch to run
    dispFreq=100,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    validFreq= 2000,  # Compute the validation error after this number of update.
    saveFreq=2000,  # Save the parameters after every saveFreq updates
    batch_size=100,  # The batch size during training.
    valid_batch_size=100,  # The batch size used for validation/test set.
    dataset='wikidb',
    train_path = r"E:\Users\v-mipeng\Codes\Projects\EntityLinking\Machine Translation\input\wiki_zh\train.txt",
    test_path = "./input/wiki_zh/test.txt",
    saveto = "./result/models/rnn model.npz",
    valid_portion = 0.005,
    # Parameter for extra option
    reload_model=None,  # Path to a saved model we want to start from.
):

    # Model options
    global model_options 
    global params
    global datasets
    model_options = locals().copy()
    print("model options", model_options)
    load_data, divid_data, prepare_data, get_vocab = datasets[dataset]  # function entrance (this is interesting)

    train = load_data(train_path)
    train, valid = divid_data(train, valid_portion)

    model_options['vocab'] = get_vocab()
    model_options['n_words'] = len(get_vocab())

    print('Building model')
    #This create the initial parameters as numpy ndarrays.
    #Dict name (string) -> numpy ndarray
    init_params() # initialize numerical parameters for embeding, projection and output

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    init_tparams()  # initialize theano tensor parameter(shared): !!! Important to learn

    # use_noise is for dropout
    (x, input_mask,
     y, output_mask, f_pred_prob, cost) = build_model()    # construct cost representation and predict function.
    f_cost = theano.function([x, input_mask, y, output_mask], cost)
    f_update = adadelta(x, input_mask, y, output_mask, cost)

    print('Optimization')

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))

    history_costs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size
    x_valid = valid[0]
    y_valid = valid[1]
    x_valid, x_valid_mask, y_valid, y_valid_mask = prepare_data((x_valid,y_valid))
    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)   # tuple: (minibatch indexes(1,2,3...), index list of each minibatch[[1,3,2],[11,4,6]...])

            for _, train_index in kf:
                uidx += 1

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, input_mask, y, output_mask = prepare_data((x, y)) # for imdb, the word is represented with integers
                n_samples += x.shape[1]

                cost = f_update(x, input_mask, y, output_mask)    

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_costs=history_costs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if numpy.mod(uidx, validFreq) == 0:
                    valid_cost = 0
                    for i in range(int(numpy.ceil(1.0*x_valid.shape[1]/valid_batch_size))):
                        begin = i*valid_batch_size
                        end = (i+1)*valid_batch_size
                        if end > x_valid.shape[1]:
                            end = x_valid.shape[1]
                        valid_cost += f_cost(x_valid[:,begin:end], x_valid_mask[:,begin:end],
                                            y_valid[:,begin:end], y_valid_mask[:,begin:end])
                    history_costs.append(valid_cost)

                    if (valid_cost <= numpy.array(history_costs).min()):
                        best_p = unzip(tparams)
                        bad_counter = 0

                    print('Valid cost ', valid_cost)

                    if (len(history_costs) > patience and
                        valid_cost >= numpy.array(history_costs)[:-patience].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    valid_cost = f_cost(x_valid, x_valid_mask, y_valid, y_valid_mask)


    print('Valid cost ', valid_cost)
    if saveto:
        numpy.savez(saveto, train_cost=cost,
                    valid_cost=valid_cost,
                    history_costs=history_costs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)
    return cost, valid_cost


def predict(
    dataset='wikidb',
    model_path = "./output/result/models/rnn model.npz",
    option_path = "./output/result/models/model option.pkl",
    test_path = "./input/wiki_zh/test.txt", # file format: mention  entity_id
    database_path = "./input/wiki_zh/kbp/id2name.txt", # file format: entity_id entity_surface
    kbp_encode_file = "./output/kbp/kbp_encode.txt",
    pred_result_file = "./output/result/pred_rank.txt"
    ):

    load_model(model_path, option_path)
    (input, input_mask, output, output_mask, hT, pred_prob) = build_single_model()    # construct model for prediction: f_h is encode function
    f_encode = theano.function([input, input_mask], hT)                               # build encode function
    f_pred_prob = theano.function([output, output_mask, hT], pred_prob)               # build predict function
    id2index, kbp = load_database(database_path)
    kbp_seqs, kbp_mask = prepare_single_data(kbp, model_options['vocab'])
    kbp_hTs = numpy.zeros(kbp_seqs.shape[1], model_options['dim_proj'])
    encode_batch_size = 1000
    pred_batch_size = 1000
    for i in range(int(numpy.ceil(1.0*kbp_seqs.shape[1]/encode_batch_size))):
        begin = i*encode_batch_size;
        end = (i+1)*encode_batch_size;
        if end > len(kbp_seqs):
            end = len(kbp_seqs)
        kbp_hTs[begin:end,:] = f_encode(kbp_seqs[:, begin:end], kbp_mask[:, begin:end])
    numpy.savez(kbp_encode_file,kbp_hTs = kbp_hTs)
    mentions, ids = load_test_data(test_path)
    writer = open(pred_result_file, "w+")
    for mention, id in zip(mentions,ids):
        try:
            index = id2index[id]
        except:
            print('Cannot find %s in kbp data base' %id)
            pass
        output = numpy.asarray(convert_single_data(mention))[:,None]
        output_batch = output.repeat(pred_batch_size, 1)
        output_mask = numpy.ones_like(output)
        output_batch_mask = numpy.ones_like(output_batch)
        pred_prob = numpy.zeros(kbp_seqs.shape[1])
        for i in range(int(numpy.ceil(1.0*kbp_seqs.shape[1]/pred_batch_size))):
            begin = i*pred_batch_size;
            end = (i+1)*pred_batch_size;
            if end > len(kbp_seqs):
                end = len(kbp_seqs)
                pred_prob[begin:end] = f_pred_prob(output.repeat(end-begin,1), output_mask.repeat(end-begin, 1), kbp_hTs[begin:end, :])
            else:
                pred_prob[begin:end] = f_pred_prob(output_batch, output_batch_mask, kbp_hTs[begin:end, :])
        sorted_idx = sorted(range(len(pred_prob)), key = pred_prob.__getitem__)
        rank = sorted_idx.index(index)
        writer.write("%s\t%s" % mention, rank)
        for i in range(50):
            writer.write("\t"+kbp[sorted_idx[i]])
        writer.write("\n")

if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    predict()
#endregion
