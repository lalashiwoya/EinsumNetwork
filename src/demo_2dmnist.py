import os
import numpy as np
import torch
from EinsumNetwork import Graph, EinsumNetwork
import datasets
import utils
from collections import defaultdict
import matplotlib.pyplot as plt

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
exponential_family = EinsumNetwork.NormalArray

classes = [0,1,2,3,4]
test_classes = [5,6,7,8]
# classes = [2, 3, 5, 7]
# classes = None

K = 10

# structure = 'poon-domingos'
structure = 'binary-trees'

# 'poon-domingos'
pd_num_pieces = [4]
# pd_num_pieces = [7]
# pd_num_pieces = [7, 28]
width = 28
height = 28

use_pair = True
num_var = 28 * (15 if use_pair else 30)

# 'binary-trees'
depth = 3
num_repetitions = 20

num_epochs = 20
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

# get data
if fashion_mnist:
    train_x, train_labels, test_x, test_labels = datasets.load_fashion_mnist()
else:
    train_x, train_labels, test_x, test_labels = datasets.load_mnist()

train_x_f, _, _, _ = datasets.load_fashion_mnist()





if not exponential_family != EinsumNetwork.NormalArray:
    train_x /= 255.
    test_x /= 255.
    train_x -= .5
    test_x -= .5

# validation split
valid_x = train_x[-10000:, :]
train_x = train_x[:-10000, :]
valid_labels = train_labels[-10000:]
train_labels = train_labels[:-10000]

# pick the selected classes
if classes is not None:
    train_x = train_x[np.any(np.stack([train_labels == c for c in classes], 1), 1), :]
    valid_x = valid_x[np.any(np.stack([valid_labels == c for c in classes], 1), 1), :]
    test_x = test_x[np.any(np.stack([test_labels == c for c in test_classes], 1), 1), :]
    train_labels = np.array([c for c in train_labels if c in classes])
    valid_labels = np.array([c for c in valid_labels if c in classes])
    
    test_labels = np.array([c for c in test_labels if c in test_classes])

train_x = torch.from_numpy(train_x).to(torch.device(device))
valid_x = torch.from_numpy(valid_x).to(torch.device(device))
test_x = torch.from_numpy(test_x).to(torch.device(device))
test_x_f = train_x_f[:len(test_x),:]
test_x_f = torch.from_numpy(test_x_f).to(torch.device(device))
# Make EinsumNetwork
######################################
if structure == 'poon-domingos':
    pd_delta = [[height / d, width / d] for d in pd_num_pieces]
    graph = Graph.poon_domingos_structure(shape=(height, width), delta=pd_delta)
elif structure == 'binary-trees':
    #graph = Graph.random_binary_trees(num_var=train_x.shape[1], depth=depth, num_repetitions=num_repetitions)
    graph = Graph.random_binary_trees(num_var=num_var, depth=depth, num_repetitions=num_repetitions)
else:
    raise AssertionError("Unknown Structure")

args = EinsumNetwork.Args(
        num_var=num_var,#train_x.shape[1],
        num_dims=2 if use_pair else 1,
        num_classes=len(classes),
        num_sums=K,
        num_input_distributions=K,
        exponential_family=exponential_family,
        exponential_family_args=exponential_family_args,
        online_em_frequency=online_em_frequency,
        online_em_stepsize=online_em_stepsize)

einet = EinsumNetwork.EinsumNetwork(graph, args)
einet.initialize()
einet.to(device)



# Train
######################################

train_N = train_x.shape[0]
valid_N = valid_x.shape[0]
test_N = test_x.shape[0]
performance = defaultdict(list)

for epoch_count in range(num_epochs):

    ##### evaluate
    einet.eval()
    train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, utils.fft2d(train_x, use_pair, width, height), batch_size=batch_size)
    valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, utils.fft2d(valid_x, use_pair, width, height), batch_size=batch_size)
    test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, utils.fft2d(test_x, use_pair, width, height), batch_size=batch_size)
    test_ll_f = EinsumNetwork.eval_loglikelihood_batched(einet, utils.fft2d(test_x_f, use_pair, width, height), batch_size=batch_size)
    performance['train'].append(train_ll/train_N)
    performance['valid'].append(valid_ll / valid_N)
    performance['test'].append(test_ll/test_N)
    performance['test on fashonmnist'].append(test_ll_f / test_N)
    print("[{}]   train LL {}   valid LL {}   test LL {}  test LL on fashon mnist {}".format(
        epoch_count,
        train_ll / train_N,
        valid_ll / valid_N,
        test_ll / test_N,
        test_ll_f / test_N))
    einet.train()
    #####

    idx_batches = torch.randperm(train_N, device=device).split(batch_size)

    total_ll = 0.0
    for idx in idx_batches:
        # (100,784)
        batch_x = train_x[idx, :]
        batch_x = utils.fft2d(batch_x, use_pair, width, height)
        outputs = einet.forward(batch_x)
        ll_sample = EinsumNetwork.log_likelihoods(outputs)
        log_likelihood = ll_sample.sum()
        log_likelihood.backward()

        einet.em_process_batch()
        total_ll += log_likelihood.detach().item()

    einet.em_update()

print(performance['train'])

fig = plt.figure()
x = np.arange(1, num_epochs+1).astype(dtype=np.str)
ax = fig.add_subplot(111)
ax.plot(x,performance['test on fashonmnist'], label = 'test on fashion mnist')
plt.legend(loc=2)
plt.yscale('symlog')
ax2 = ax.twinx()
ax2.plot(x,performance['train'], label = 'train')
ax2.plot(x,performance['valid'], label = 'valid')
ax2.plot(x,performance['test'], label = 'test')

plt.legend(loc=1)

ax.set_xlabel('epoch')
ax.set_ylabel('Fashion mnist')

#ax.set_ylim([-40000000,-6000])
ax2.set_ylabel('mnist')
plt.yscale('symlog')
plt.savefig('log likelihood on different datasets.png')
if fashion_mnist:
    model_dir = '../models/einet/demo_2dfashion_mnist/'
    samples_dir = '../samples/demo_2dfashion_mnist/'
else:
    model_dir = '../models/einet/demo_2dmnist/'
    samples_dir = '../samples/demo_2dmnist/'

utils.mkdir_p(model_dir)
utils.mkdir_p(samples_dir)

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

#####################
# draw some samples #
#####################

#samples = einet.sample(num_samples=25).cpu().numpy()
#samples = samples.reshape((-1, 28, 28))
# irfft2 by @yu
samples = einet.sample(num_samples=25)
samples = samples.reshape((-1, 28, 30))
samples = samples[:,:,:15] + samples[:,:,15:] * 1j
print(samples.shape)
# filtering high freqs

samples[:,:,10:]=0
samples[:,10:20,:]=0
samples = torch.fft.irfft2(samples, norm="forward").cpu().numpy()
samples[samples>0.5]=0.5
samples[samples<-0.5]=-0.5



utils.save_image_stack(samples, 5, 5, os.path.join(samples_dir, "samples.png"), margin_gray_val=0.)
#0/0
# Draw conditional samples for reconstruction
image_scope = np.array(range(height *width)).reshape(height, width)
marginalize_idx = list(image_scope[0:round(height/2), :].reshape(-1))
keep_idx = [i for i in range(width*height) if i not in marginalize_idx]
einet.set_marginalization_idx(marginalize_idx)

num_samples = 10
samples = None

for k in range(num_samples):
    if samples is None:
        samples = einet.sample(x = utils.fft2d(test_x[0:25, :], use_pair, width, height)).cpu().numpy()
    else:
        samples += einet.sample(x = utils.fft2d(test_x[0:25, :], use_pair, width, height)).cpu().numpy()
samples /= num_samples
samples = samples.squeeze()
samples = samples.reshape(-1, 28, 15, 2)
samples = samples[:,:,:,0] + samples[:,:,:,1]*1j
samples = np.abs(np.fft.ifftn(samples))
print(samples.shape)
#samples = samples.reshape((-1, 28, 28))
utils.save_image_stack(samples, 5, 5, os.path.join(samples_dir, "sample_reconstruction.png"), margin_gray_val=0.)

# ground truth
ground_truth = test_x[0:25, :].cpu().numpy()
ground_truth = ground_truth.reshape((-1, 28, 28))
utils.save_image_stack(ground_truth, 5, 5, os.path.join(samples_dir, "ground_truth.png"), margin_gray_val=0.)

###############################
# perform mpe reconstructions #
###############################

mpe = einet.mpe().cpu().numpy()
mpe = mpe.reshape((-1, 28, 15, 2))
mpe = np.abs(np.fft.ifftn(mpe[:,:,:,0]+ mpe[:,:,:,1]*1j))
utils.save_image_stack(mpe, 1, 1, os.path.join(samples_dir, "mpe.png"), margin_gray_val=0.)

# Draw conditional samples for reconstruction
image_scope = np.array(range(height * width)).reshape(height, width)
marginalize_idx = list(image_scope[0:round(height/2), :].reshape(-1))
keep_idx = [i for i in range(width*height) if i not in marginalize_idx]
einet.set_marginalization_idx(marginalize_idx)

mpe_reconstruction = einet.mpe(x=utils.fft2d(test_x[0:25, :], use_pair, width, height)).cpu().numpy()
mpe_reconstruction = mpe_reconstruction.squeeze()
mpe_reconstruction = mpe_reconstruction.reshape((-1, 28, 15, 2))
mpe_reconstruction = mpe_reconstruction[:,:,:,0]+mpe_reconstruction[:,:,:,1]*1j
mpe_recontruction = np.fft.ifftn(mpe_reconstruction)
utils.save_image_stack(np.abs(mpe_reconstruction), 5, 5, os.path.join(samples_dir, "mpe_reconstruction.png"), margin_gray_val=0.)

print()
print('Saved samples to {}'.format(samples_dir))


# evaluate log-likelihoods
einet.eval()

train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, utils.fft2d(train_x, use_pair, width, height), batch_size=batch_size)
valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, utils.fft2d(valid_x, use_pair, width, height), batch_size=batch_size)
test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, utils.fft2d(test_x, use_pair, width, height), batch_size=batch_size)
print()

print("Log-likelihoods after saving  --- train LL {}   valid LL {}   test LL {}".format(
        train_ll / train_N,
        valid_ll / valid_N,
        test_ll / test_N))

