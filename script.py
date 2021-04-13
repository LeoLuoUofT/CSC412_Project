# shared constants
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import os
import json
import logging
from collections import defaultdict 

from import_midi import import_midi_from_folder

from project import *

# constant variables
PITCH_DIM = 61
INSTRUMENT_DIM = 1 # same as pitch pls?
VELOCITY_DIM = 1 

NUM_STYLES = 3
LATENT_DIM = 64 # ??

BATCH_SIZE = 1024 # bigger if possible, but might kill GPU

RNN_CELL = "gru"
RNN_CELL_NUMBER = LATENT_DIM # this is a lot

# all hyperparameters in here lol
# sorry, hope this is easy to read

use_cuda = True

encoder_dict = {
    "q_y_rnn": {"input_dim": PITCH_DIM + INSTRUMENT_DIM + VELOCITY_DIM,
                "output_dim": 3 * LATENT_DIM,
                "rnn_cell": RNN_CELL},
    "q_y_linear": {"input_dim": 3 * LATENT_DIM,
                   "output_dim": LATENT_DIM,
                   "output_activation_fn": "relu"},
    "q_y_gumbel": {"input_dim": LATENT_DIM,
                   "output_dim": NUM_STYLES},

    "q_z_rnn": {"input_dim": PITCH_DIM + INSTRUMENT_DIM + VELOCITY_DIM,
                "output_dim": 3 * LATENT_DIM,
                "rnn_cell": RNN_CELL},
    "q_z_linear": {"input_dim": 3 * LATENT_DIM + NUM_STYLES,
                   "output_dim": LATENT_DIM,
                   "output_activation_fn": "relu"},
    "q_z_gaussian": {"input_dim": LATENT_DIM,
                     "output_dim": LATENT_DIM},
    "cuda": use_cuda,
}

decoder_dict = {
    "input_dim": NUM_STYLES,
    "output_dim": LATENT_DIM,

    "p_x_linear": {"input_dim": LATENT_DIM,
                   "output_dim": 3 * LATENT_DIM,
                   "output_activation_fn": "tanh"}, # tanh better for rnn?
    "p_x_rnn": {"input_dim": 3 * LATENT_DIM,
                "output_dim": PITCH_DIM + INSTRUMENT_DIM + VELOCITY_DIM,
                "rnn_cell": RNN_CELL},
    "cuda": use_cuda,
}

argdict = {
    "learning_rate": 1e-4,
    "decay_epoch": 25,
    "lr_decay": 5e-1,

    # tune these
    "weight_style": 1,
    "weight_entropy": 0.5,
    "weight_sampling": 1,

    "weight_pitch": 1,
    "weight_velocity": 1,
    "weight_instrument": 1,

    "pitch_size": PITCH_DIM, #idk,
    "instrument_size": INSTRUMENT_DIM, # idk
    "velocity_size": VELOCITY_DIM,

    "init_temp": 1e-1,
    "decay_temp": 1e-1,
    "min_temp": 1e-5,
    "decay_temp_rate": 25, # every N epochs

    "cuda": use_cuda, # use GPU if possible

    "num_epochs": 100, # other paper uses 400, that's nuts

    "save_epoch": 5,
    "log_epoch": 5,

    "save_path": "./output/",
    "logdir" : "./logs/trial3",
    "encoder": encoder_dict,
    "decoder": decoder_dict,
}

data_folder = "./full_pickles/"

v, vt, _, _, _, _, i, it, _, _, \
    p, pt, c, ct, _, _ = import_midi_from_folder(data_folder, load_from_pkl = True) # test
print("data loaded from", data_folder)

# the original paper has some weird processing tricks
# we ignore those, i don't really understand them

# need to do some preprocessing on the data
labels = []
for i in range(len(p)):
    temp_labels = [c[i]] * p[i].shape[0]
    labels.extend(temp_labels)
    
labels_test = []
for i in range(len(pt)):
    temp_labels = [ct[i]] * p[i].shape[0]
    labels_test.extend(temp_labels)
    
# instruments are too difficult to parse
# just leave it as a random matrix
v = np.concatenate(v, axis=0)[:,:,np.newaxis]
vt = np.concatenate(vt, axis=0)[:,:,np.newaxis]

p = np.concatenate(p, axis=0)
pt = np.concatenate(pt, axis=0)[:,:,np.newaxis]

i = np.random.random(v.shape)
it = np.random.random(vt.shape)


train_data = music(p, i, v, labels)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)
print("data loader created")
test_data = music(pt, it, vt, labels_test)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


model = Model(argdict)
print("model created")

model.run(train_dataloader, test_dataloader)

