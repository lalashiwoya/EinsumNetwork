import os
import numpy as np
import torch
from EinsumNetwork import Graph, EinsumNetwork
import datasets
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

demo_text = """
This demo loads (fashion) mnist and quickly trains an EiNet for some epochs. 

There are some parameters to play with, as for example which exponential family you want 
to use, which classes you want to pick, and structural parameters. Then an EiNet is trained, 
the log-likelihoods reported, some (conditional and unconditional) samples are produced, and
approximate MPE reconstructions are generated. 
"""
print(demo_text)

############################################################################
fashion_mnist = False

# exponential_family = EinsumNetwork.BinomialArray
# exponential_family = EinsumNetwork.CategoricalArray
# exponential_family = EinsumNetwork.NormalArray
exponential_family = EinsumNetwork.MultivariateNormalArray

classes = [7]
# classes = list(range(10))
# classes = [2, 3, 5, 7]
# classes = None

K = 1

# TODO: seems like i have exchanged width and height about everywhere, even in data loader!
width = 28
height = 28
fft_components = height // 2 + 1
input_size = width * fft_components

# structure = 'poon-domingos'
structure = 'binary-trees'

# 'poon-domingos'
pd_num_pieces = [4]
# pd_num_pieces = [7]
# pd_num_pieces = [7, 28]

# 'binary-trees'
depth = 4
num_repetitions = 7
split_vars_on_repetition = False

num_epochs = 10
batch_size = 100
online_em_frequency = 1
online_em_stepsize = 0.05
############################################################################

exponential_family_args = None
if exponential_family == EinsumNetwork.BinomialArray:
    exponential_family_args = {'N': 255}
if exponential_family == EinsumNetwork.CategoricalArray:
    exponential_family_args = {'K': 256}
if exponential_family == EinsumNetwork.NormalArray:
    exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}
if exponential_family == EinsumNetwork.MultivariateNormalArray:
    exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}

# get data
if fashion_mnist:
    train_x_raw, train_labels, test_x_raw, test_labels = datasets.load_fashion_mnist(width, height)
else:
    train_x_raw, train_labels, test_x_raw, test_labels = datasets.load_mnist(width, height)

# TODO: Rework this section
train_x = torch.fft.rfft(torch.tensor(train_x_raw.reshape((-1, width, height))), norm='forward')
test_x = torch.fft.rfft(torch.tensor(test_x_raw.reshape((-1, width, height))), norm='forward')

train_x = train_x.reshape((-1, train_x.shape[1] * train_x.shape[2]))
test_x = test_x.reshape((-1, test_x.shape[1] * test_x.shape[2]))

train_x = torch.stack([train_x.real, train_x.imag], dim=-1)
test_x = torch.stack([test_x.real, test_x.imag], dim=-1)

# validation split
valid_x = train_x[-10000:, :]
train_x = train_x[:-10000, :]
valid_labels = train_labels[-10000:]
train_labels = train_labels[:-10000]

# pick the selected classes
if classes is not None:
    train_x = train_x[np.any(np.stack([train_labels == c for c in classes], 1), 1), :]
    valid_x = valid_x[np.any(np.stack([valid_labels == c for c in classes], 1), 1), :]
    test_x = test_x[np.any(np.stack([test_labels == c for c in classes], 1), 1), :]
    test_x_raw = test_x_raw[np.any(np.stack([test_labels == c for c in classes], 1), 1), :]

train_x = train_x.to(torch.device(device))
valid_x = valid_x.to(torch.device(device))
test_x = test_x.to(torch.device(device))


# Make EinsumNetwork
######################################
if structure == 'poon-domingos':
    pd_delta = [[height / d, width / d] for d in pd_num_pieces]
    graph = Graph.poon_domingos_structure(shape=(height, width), delta=pd_delta)
elif structure == 'binary-trees':
    graph = Graph.random_binary_trees(num_var=input_size, depth=depth, num_repetitions=num_repetitions,
                                      split_vars_on_repetition=split_vars_on_repetition)
else:
    raise AssertionError("Unknown Structure")

args = EinsumNetwork.Args(
        num_var=input_size,
        num_dims=2,
        num_classes=1,
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

# Train
######################################

train_N = train_x.shape[0]
valid_N = valid_x.shape[0]
test_N = test_x.shape[0]

lls = []
import datetime as dt

a = dt.datetime.now()

for epoch_count in range(num_epochs):

    ##### evaluate
    einet.eval()
    train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
    valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
    test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)
    lls.append((train_ll / train_N, valid_ll / valid_N, test_ll / test_N))
    print("[{}]   train LL {}   valid LL {}   test LL {}".format(
        epoch_count,
        train_ll / train_N,
        valid_ll / valid_N,
        test_ll / test_N))
    einet.train()
    #####

    idx_batches = torch.randperm(train_N, device=device).split(batch_size)

    total_ll = 0.0
    for idx in idx_batches:
        batch_x = train_x[idx, :]
        outputs = einet.forward(batch_x)
        ll_sample = EinsumNetwork.log_likelihoods(outputs)
        log_likelihood = ll_sample.sum()
        log_likelihood.backward()

        einet.em_process_batch()
        total_ll += log_likelihood.detach().item()

    einet.em_update()

print((dt.datetime.now()-a).total_seconds())

import matplotlib.pyplot as plt

x = list(range(num_epochs))
plt.plot(x, [ll[0] for ll in lls], label='train')
plt.plot(x, [ll[1] for ll in lls], label='val')
plt.plot(x, [ll[2] for ll in lls], label='test')
plt.legend()
plt.show()

x = list(range(1, num_epochs))
plt.plot(x, [ll[0] for i, ll in enumerate(lls) if i > 0], label='train')
plt.plot(x, [ll[1] for i, ll in enumerate(lls) if i > 0], label='val')
plt.plot(x, [ll[2] for i, ll in enumerate(lls) if i > 0], label='test')
plt.legend()
plt.show()

if fashion_mnist:
    model_dir = '../models/einet/demo_fashion_mnist/'
    samples_dir = '../samples/demo_fashion_mnist/'
else:
    model_dir = '../models/einet/demo_mnist/'
    samples_dir = '../samples/demo_mnist/'
utils.mkdir_p(model_dir)
utils.mkdir_p(samples_dir)

#####################
# draw some samples #
#####################

samples = einet.sample(num_samples=25, ifft=True).cpu()
utils.save_image_stack(samples, 5, 5, os.path.join(samples_dir, "samples.png"), margin_gray_val=0.)

# Draw conditional samples for reconstruction
image_scope = np.array(range(height * width)).reshape(height, width)
marginalize_idx = list(image_scope[0:round(height/2), :].reshape(-1))
keep_idx = [i for i in range(width*height) if i not in marginalize_idx]
einet.set_marginalization_idx(marginalize_idx)

num_samples = 10
samples = None
for k in range(num_samples):
    if samples is None:
        samples = einet.sample(x=test_x[0:25, :], ifft=True).cpu().numpy()
    else:
        samples += einet.sample(x=test_x[0:25, :], ifft=True).cpu().numpy()
samples /= num_samples
samples = samples.squeeze()

samples = samples.reshape((-1, width, height))
utils.save_image_stack(samples, 5, 5, os.path.join(samples_dir, "sample_reconstruction.png"), margin_gray_val=0.)

# ground truth
ground_truth = test_x_raw[0:25, :]
ground_truth = ground_truth.reshape((-1, width, height))
utils.save_image_stack(ground_truth, 5, 5, os.path.join(samples_dir, "ground_truth.png"), margin_gray_val=0.)

###############################
# perform mpe reconstructions #
###############################

mpe = einet.mpe(ifft=True).cpu().numpy()
mpe = mpe.reshape((1, width, height))
utils.save_image_stack(mpe, 1, 1, os.path.join(samples_dir, "mpe.png"), margin_gray_val=0.)

# Draw conditional samples for reconstruction
image_scope = np.array(range(height * width)).reshape(height, width)
marginalize_idx = list(image_scope[0:round(height/2), :].reshape(-1))
keep_idx = [i for i in range(width*height) if i not in marginalize_idx]
einet.set_marginalization_idx(marginalize_idx)

mpe_reconstruction = einet.mpe(x=test_x[0:25, :], ifft=True).cpu().numpy()
mpe_reconstruction = mpe_reconstruction.squeeze()
mpe_reconstruction = mpe_reconstruction.reshape((-1, width, height))
utils.save_image_stack(mpe_reconstruction, 5, 5, os.path.join(samples_dir, "mpe_reconstruction.png"), margin_gray_val=0.)

print()
print('Saved samples to {}'.format(samples_dir))

####################
# save and re-load #
####################

# evaluate log-likelihoods
einet.eval()
train_ll_before = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
valid_ll_before = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
test_ll_before = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)

# save model
graph_file = os.path.join(model_dir, "einet.pc")
Graph.write_gpickle(graph, graph_file)
print("Saved PC graph to {}".format(graph_file))
model_file = os.path.join(model_dir, "einet.mdl")
torch.save(einet, model_file)
print("Saved model to {}".format(model_file))

del einet

# reload model
einet = torch.load(model_file)
print("Loaded model from {}".format(model_file))

# evaluate log-likelihoods on re-loaded model
train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=batch_size)
valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=batch_size)
test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=batch_size)
print()
print("Log-likelihoods before saving --- train LL {}   valid LL {}   test LL {}".format(
        train_ll / train_N,
        valid_ll / valid_N,
        test_ll / test_N))
print("Log-likelihoods after saving  --- train LL {}   valid LL {}   test LL {}".format(
        train_ll / train_N,
        valid_ll / valid_N,
        test_ll / test_N))
