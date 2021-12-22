from EinsumNetwork import Graph, EinsumNetwork
import datasets
import utils
from collections import defaultdict
import matplotlib.pyplot as plt
import cifar10
import torch
import numpy as np
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = [0,1,2,3,4]
test_classes = [5,6,7,8,9]
K = 10
structure = 'binary-trees'

pd_num_pieces = [4]
# pd_num_pieces = [7]
# pd_num_pieces = [7, 28]
width = 32
height = 32

fft_components = height // 2 + 1
input_size = width * fft_components

use_pair = True
num_var = 32 * (17 if use_pair else 34)

depth = 3
num_repetitions = 20

num_epochs = 20
batch_size = 100
online_em_frequency = 1
online_em_stepsize = 0.05

exponential_family_args = None
exponential_family = EinsumNetwork.NormalArray

if exponential_family == EinsumNetwork.BinomialArray:
        exponential_family_args = {'N': 255}
if exponential_family == EinsumNetwork.CategoricalArray:
        exponential_family_args = {'K': 256}
if exponential_family == EinsumNetwork.NormalArray:
        exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}

train_x = []
train_labels = []
test_x = []
test_labels = []

for image, label in cifar10.data_batch_generator():
    if label in classes:
        train_x.append(image)
        train_labels.append(label)
    else:
        test_x.append(image)
        test_labels.append(label)

train_x = np.array(train_x, dtype = np.float32)
test_x = np.array(test_x, dtype = np.float32)

if not exponential_family != EinsumNetwork.NormalArray:
        train_x /= 255.
        test_x /= 255.
        train_x -= .5
        test_x -= .5

train_num = 100
valid_num = 100
test_num = 100
valid_x = train_x[train_num:train_num+valid_num]
train_x = train_x[:train_num]
test_x = test_x[:test_num]
valid_labels = train_labels[train_num:train_num+valid_num]
train_labels = train_labels[:train_num]
test_labels = test_labels[:test_num]

train_x = torch.from_numpy(train_x).to(device)
valid_x = torch.from_numpy(valid_x).to(device)
test_x = torch.from_numpy(test_x).to(device)

if structure == 'poon-domingos':
    pd_delta = [[height / d, width / d] for d in pd_num_pieces]
    graph = Graph.poon_domingos_structure(shape=(height, width), delta=pd_delta)
elif structure == 'binary-trees':
    graph = Graph.random_binary_trees(num_var=input_size, depth=depth, num_repetitions=num_repetitions)
else:
    raise AssertionError("Unknown Structure")

args = EinsumNetwork.Args(
        num_var=input_size,
        num_dims= 2,
        num_classes= len(classes),
        num_sums=K,
        num_input_distributions=K,
        exponential_family=exponential_family,
        exponential_family_args=exponential_family_args,
        online_em_frequency=online_em_frequency,
        online_em_stepsize=online_em_stepsize)
einet = EinsumNetwork.EinsumNetwork(graph, args)
einet.initialize()
einet.to(device)
print(einet)

# train
train_N = train_x.shape[0]
valid_N = valid_x.shape[0]
test_N = test_x.shape[0]

performance = defaultdict(list)

einet = EinsumNetwork.EinsumNetwork(graph, args)
einet.initialize()
einet.to(device)
print(einet)
print(train_x.shape)
for epoch_count in range(num_epochs):
    einet.eval()
    train_ll = 0
    valid_ll = 0
    test_ll = 0
    for i in range(train_x.shape[-1]):
        print('train_x',train_x[:,:,:,0].shape)
        train_ll += EinsumNetwork.eval_loglikelihood_batched(einet, utils.fft2d(train_x[:,:,:,i], use_pair, width, height))
        valid_ll += EinsumNetwork.eval_loglikelihood_batched(einet, utils.fft2d(valid_x[:,:,:,i], use_pair, width, height))
        test_ll += EinsumNetwork.eval_loglikelihood_batched(einet, utils.fft2d(test_x[:,:,:,i], use_pair, width, height))
    train_ll /= 3
    valid_ll /= 3
    test_ll /= 3
    performance['train'].append(train_ll/train_N)
    performance['valid'].append(valid_ll/valid_N)
    performance['test'].append(test_ll/test_N)
    print("[{}] train LL {} valid LL {} test LL {}".format(
        epoch_count,
        train_ll/ train_N,
        valid_ll/ valid_N,
        test_ll/ test_N))
    
    einet.train()
    idx_batches = torch.randperm(train_N, device = device).split(batch_size)

    for idx in idx_batches:
        batch_x = train_x[idx,:]
        log_likelihood = 0

        for i in range(batch_x.shape[-1]):
            batch_x_dim = utils.fft2d(batch_x[:,:,:,i], use_pair, width, height)
            outputs = einet.forward(batch_x_dim)
            ll_sample = EinsumNetwork.log_likelihoods(outputs)
            log_likelihood += ll_sample.sum()
        log_likelihood /= 3
        log_likelihood.backward()
        einet.em_process_batch()
    einet.em_update()

model_dir = '../models/einet/cifar/'
samples_dir = '../samples/demo_cifar/'

utils.mkdir_p(model_dir)
utils.mkdir_p(samples_dir)

graph_file = os.path.join(model_dir, 'einet.pc')
Graph.write_gpickle(graph, graph_file)
print('Saved PC graph to {}'.format(graph_file))
model_file = os.path.join(model_dir, 'einet.mdl')
torch.save(einet, model_file)
print('Saved model to {}'.format(model_file))

del einet

einet = torch.load(model_file)
print('Loaded model from {}'.format(model_file))

train_ll = 0
valid_ll = 0
test_ll  = 0

for i in range(train_x.shape[-1]):
    train_ll += EinsumNetwork.eval_loglikelihood_batched(einet, utils.fft2d(train_x[:,:,:,i], use_pair, width, height), batch_size = batch_size)      
    valid_ll += EinsumNetwork.eval_loglikelihood_batched(einet, utils.fft2d(valid_x[:,:,:,i], use_pair, width, height), batch_size = batch_size)
    test_ll += EinsumNetwork.eval_loglikelihood_batched(einet, utils.fft2d(test_x[:,:,:,i], use_pair, width, height), batch_size = batch_size)
print()
print("Log-likelihoods after saving -- train LL {} valid LL {} test LL {}".format(
    train_ll / (3*train_N),
    valid_ll / (3*valid_N),
    test_ll / (3*test_N)))

samples = einet.sample(num_samples = 25)
samples = samples.reshape((-1, 32, 34))
samples = samples[:,:,:17] + samples[:,:,17:]*1j
samples[:,:,10:] = 0
samples[:, 10:20,:] = 0
samples = torch.fft.irfft2(samples, norm = 'forward').cpu().numpy()
samples[samples>0.5] = 0.5
samples[samples<-0.5] = -0.5
utils.save_image_stack(samples, 5, 5, os.path.join(samples_dir, 'samples.png'), margin_gray_val = 0.)


