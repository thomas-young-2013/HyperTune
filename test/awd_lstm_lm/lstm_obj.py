import time
import math
import numpy as np
import torch
import torch.nn as nn
from math import ceil
import os
import hashlib
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter, Constant

import data
from splitcross import SplitCrossEntropyLoss
from model import RNNModel

try:
    from utils import batchify, get_batch, repackage_hidden
except Exception:
    import sys
    sys.path.insert(0, '.')
    from test.awd_lstm_lm.utils import batchify, get_batch, repackage_hidden

# Set the random seed manually for reproducibility.
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

log_interval = 200
tied = True
bptt = 70
max_epoch = 200
n_layers = 3
clip = 0.25
alpha = 2
beta = 1
decay_epoch = [100, 150]


def get_lstm_configspace():
    cs = ConfigurationSpace()
    hp_batch_size = CategoricalHyperparameter('batch_size', [10, 20, 40], default_value=20)
    hp_dropouti = UniformFloatHyperparameter('dropouti', 0, 0.9, default_value=0.5)
    hp_dropouth = UniformFloatHyperparameter('dropouth', 0, 0.9, default_value=0.5)
    hp_dropoute = UniformFloatHyperparameter('dropoute', 0, 0.9, default_value=0.5)
    hp_dropout = UniformFloatHyperparameter('dropout', 0, 0.9, default_value=0.5)
    hp_wdrop = UniformFloatHyperparameter('wdrop', 0, 0.9, default_value=0.5)
    hp_emsize = Constant('emsize', 400)
    hp_hdsize = UniformIntegerHyperparameter('hidden_size', 500, 2000, default_value=1000)
    hp_weight_decay = UniformFloatHyperparameter('wdecay', 1e-6, 1e-2, log=True, default_value=1e-5)
    hp_lr = UniformFloatHyperparameter('lr', 1e-1, 100, log=True, default_value=10)
    cs.add_hyperparameters(
        [hp_batch_size, hp_dropout, hp_dropoute, hp_dropouth, hp_dropouti, hp_wdrop, hp_emsize,
         hp_hdsize, hp_weight_decay, hp_lr])
    return cs


###############################################################################
# Load data
###############################################################################

def model_save(fn, model, criterion, optimizer, epoch):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer, epoch], f)


def model_load(fn):
    with open(fn, 'rb') as f:
        model, criterion, optimizer, epoch = torch.load(f)
    return model, criterion, optimizer, epoch


def get_corpus(data_path):
    fn = 'corpus.{}.data'.format(hashlib.md5(data_path.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = data.Corpus(data_path)
        torch.save(corpus, fn)
    return corpus

eval_batch_size = 10
test_batch_size = 1


###############################################################################
# Build the model
###############################################################################

def mf_objective_func_gpu(config, n_resource, extra_conf, device, total_resource, run_test=False,
                          model_dir='./data/lstm_save_models/unnamed_trial', eta=3, corpus=None):  # device='cuda' 'cuda:0'
    assert corpus is not None

    print('extra_conf:', extra_conf)
    initial_run = extra_conf['initial_run']
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    except FileExistsError:
        pass

    device = torch.device(device)

    criterion = None
    dropout = config['dropout']
    dropouth = config['dropouth']
    dropouti = config['dropouti']
    dropoute = config['dropoute']
    wdrop = config['wdrop']
    emsize = config['emsize']
    hidden_size = config['hidden_size']
    weight_decay = config['wdecay']
    batch_size = config['batch_size']
    lr = config['lr']
    print('worker receive config:', config)

    epoch_ratio = float(n_resource) / float(total_resource)
    config_model_path = os.path.join(model_dir,
                                     'tmp_' + get_path_by_config(config) + '_%d' % int(n_resource / eta) + '.pt')
    save_path = os.path.join(model_dir,
                             'tmp_' + get_path_by_config(config) + '_%d' % int(n_resource) + '.pt')

    ntokens = len(corpus.dictionary)
    model = RNNModel('LSTM', ntokens, emsize, hidden_size, n_layers, dropout, dropouth,
                     dropouti, dropoute, wdrop, tied)

    train_data = batchify(corpus.train, batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)

    if not initial_run:
        print('Resuming model ...')
        if not os.path.exists(config_model_path):
            raise ValueError('not initial_run but config_model_path not exists. check if exists duplicated configs '
                             'and saved model were removed.')
        model, criterion, optimizer, init_epoch_num = model_load(config_model_path)
        optimizer.param_groups[0]['lr'] = lr
        model.dropouti, model.dropouth, model.dropout, dropoute = dropouti, dropouth, dropout, dropoute
        if wdrop:
            from weight_drop import WeightDrop

            for rnn in model.rnns:
                if type(rnn) == WeightDrop:
                    rnn.dropout = wdrop
                elif rnn.zoneout > 0:
                    rnn.zoneout = wdrop

        epoch_num = ceil(max_epoch * epoch_ratio) - ceil(
            max_epoch * epoch_ratio / eta)
    else:
        init_epoch_num = 1
        epoch_num = ceil(max_epoch * epoch_ratio)
    print('epoch_num', epoch_num)
    ###
    if not criterion:
        splits = []
        if ntokens > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif ntokens > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
        print('Using splits:', splits)
        criterion = SplitCrossEntropyLoss(emsize, splits=splits, verbose=False)
    ###

    model = model.to(device)
    criterion = criterion.to(device)
    ###
    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Model total parameters:', total_params)
    # Loop over epochs.
    best_val_loss = []
    stored_loss = 100000000

    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)

    return_pp = 1e10
    for epoch in range(init_epoch_num, init_epoch_num + epoch_num):
        epoch_start_time = time.time()
        train(corpus, model, criterion, optimizer, epoch, batch_size, train_data, bptt)
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                if 'ax' in optimizer.state[prm]:
                    prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(corpus, model, criterion, val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)

            for prm in model.parameters():
                if prm in tmp:
                    prm.data = tmp[prm].clone()
            return_pp = math.exp(val_loss2)

        else:
            val_loss = evaluate(corpus, model, criterion, val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)

            if epoch in decay_epoch:
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)
            return_pp = math.exp(val_loss)

    model_save(save_path, model, criterion, optimizer, init_epoch_num + epoch_num)

    try:
        if epoch_ratio == 1:
            s_max = int(math.log(total_resource) / math.log(eta))
            for i in range(0, s_max + 1):
                if os.path.exists(os.path.join(model_dir,
                                               'tmp_' + get_path_by_config(config) + '_%d' % int(eta ** i) + '.pt')):
                    os.remove(os.path.join(model_dir,
                                           'tmp_' + get_path_by_config(config) + '_%d' % int(eta ** i) + '.pt'))
    except Exception as e:
        print('unexpected exception!')
        import traceback
        traceback.print_exc()

    # Turn it into a minimization problem.
    result = dict(
        objective_value=return_pp,
    )

    return result


###############################################################################
# Training code
###############################################################################

def evaluate(corpus, model, criterion, data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, bptt):
        data, targets = get_batch(data_source, i, bptt, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train(corpus, model, criterion, optimizer, epoch, batch_size, train_data, bptt):
    params = list(model.parameters()) + list(criterion.parameters())
    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        _bptt = bptt if np.random.random() < 0.95 else bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(_bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / bptt
        model.train()
        data, targets = get_batch(train_data, i, bptt, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if alpha: loss = loss + sum(
            alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if beta: loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if clip: torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, int(len(train_data) // bptt), optimizer.param_groups[0]['lr'],
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len


def get_configuration_id(config, is_dict=False):
    if is_dict:
        data_dict = config
    else:
        data_dict = config.get_dictionary()
    data_list = []
    for key, value in sorted(data_dict.items(), key=lambda t: t[0]):
        if isinstance(value, float):
            value = round(value, 5)
        data_list.append('%s-%s' % (key, str(value)))
    data_id = '_'.join(data_list)
    sha = hashlib.sha1(data_id.encode('utf8'))
    return sha.hexdigest()


def get_path_by_config(config, is_dict=False):
    return '%s.pt' % get_configuration_id(config, is_dict=is_dict)


if __name__ == '__main__':
    data_path = './test/awd_lstm_lm/data/penn'
    cs = get_lstm_configspace()
    default_config = cs.get_default_configuration()
    corpus = get_corpus(data_path)
    extra_conf = dict(initial_run=True)
    result = mf_objective_func_gpu(default_config, 27, extra_conf, device='cuda:0', total_resource=27,
                                   model_dir='./data/lstm_save_models/unnamed_trial',
                                   eta=3, corpus=corpus)
    print(result)
