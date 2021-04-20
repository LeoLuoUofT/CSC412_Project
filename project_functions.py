# Contains all the classes and helper functions, drawn from Project.ipynb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import Dataset, DataLoader

import os
import json
import logging
from collections import defaultdict 

from pathlib import Path

import progressbar

def get_activation_function(id):
  if id == 'relu':
    return nn.ReLU()
  
  if id == 'sigmoid':
    return nn.Sigmoid()

  if id == 'tanh':
    return nn.Tanh()

  if id == 'none':
    return nn.Identity()

  else:
    raise ValueError

def get_batch_norm(bool, size):
  if bool:
    return nn.BatchNorm1d(size)

  else:
    return nn.Identity()

def get_rnn_cell(id):
  # returns constructor
  if id == 'gru':
    return nn.GRU

  if id == 'lstm':
    return nn.LSTM  

  if id == 'basic':
    return nn.RNN # why use this


class ff(nn.Module):
    def __init__(self, argdict):
      """
      argdict: contains all arguments 
      """
      super(ff, self).__init__()

      self.argdict = argdict

      input_dim = self.argdict["input_dim"]
      output_dim = self.argdict["output_dim"]
      if "layer_params" in self.argdict:
        layer_params = self.argdict["layer_params"]
        n_layers = len(layer_params)
      else:
        n_layers = 0  

      
      self.layers = []
      
      if n_layers != 0:
        self.layers.append(
          nn.Sequential(
            nn.Linear(input_dim, layer_params[0]["size"]),
            get_batch_norm(self.argdict[0]["batch_norm"], layer_params[0]["size"]),
            get_activation_function(layer_params[0]["activation_fn"])
          )
        )
      
        for i in range(n_layers-1):
          self.layers.append(
            nn.Sequential(
              nn.Linear(layer_params[i]["size"], layer_params[i+1]["size"]),
              get_batch_norm(self.argdict[i+1]["batch_norm"], layer_params[i+1]["size"]),
              get_activation_function(layer_params[i+1]["activation_fn"])
            )
        )
        
        self.layers.append(
            nn.Sequential(
              nn.Linear(layer_params[-1]["size"], output_dim),
              get_activation_function(self.argdict["output_activation_fn"])
            )
        )

      # 0 layer case, just pipe to output
      else:
        self.layers.append(
            nn.Sequential(
              nn.Linear(input_dim, output_dim),
              get_activation_function(self.argdict["output_activation_fn"])
            )
        )
      self.layers = torch.nn.ModuleList(self.layers)
    def forward(self, x):
        #print(x.is_cuda())
        for layer in self.layers:
            x = layer(x)

        return x

class rnn(nn.Module):
    def __init__(self, argdict):
      """
      """
      super(rnn, self).__init__()

      self.argdict = argdict

      # no embedding? assume already embedded

      input_dim = self.argdict["input_dim"]
      output_dim = self.argdict["output_dim"]
        
      # hidden default to zero

      rnn_cell_constructor = get_rnn_cell(self.argdict["rnn_cell"])

      self.rnn_layer = rnn_cell_constructor(input_dim, output_dim, batch_first = True)

    def forward(self, x):
      return self.rnn_layer(x)

# sampling
def gumbel_sampler(x, temperature):
    # softmax but with noise
    sampled = torch.rand(x.size())
    eps = 1e-10 # stability
    if x.is_cuda:
      sampled = sampled.cuda()
    noise = torch.log(-torch.log(sampled + eps) + eps) # loglog
    return F.softmax((x - noise) / temperature, dim=-1)

def gaussian_sampler(m, v):
    std = torch.sqrt(v + 1e-10)
    eps = torch.randn_like(std)
    z = m + eps * std
    return z

# losses
def cross_entropy(logits, labels):
    return F.cross_entropy(logits, labels)

def mse(pred, labels):
    loss = (pred - labels).pow(2)
    return loss.sum(-1).mean()

def entropy(logits, labels):
    # wrt logits
    log_q = F.log_softmax(logits, dim=-1)
    return -torch.mean(torch.sum(labels * log_q, dim=-1))

def log_normal(z, m, v):
    v_stable = v + 1e-10
    return -0.5 * torch.sum(torch.pow(z - m, 2)/v + torch.log(v), dim=-1) # ignore constant 

def gaussian_kl(sample, mu, var, mu_prior, var_prior):
    loss = log_normal(sample, mu, var) - log_normal(sample, mu_prior, var_prior)
    return loss.mean()


class softmax_with_gumbel(nn.Module):

  def __init__(self, argdict):
    super(softmax_with_gumbel, self).__init__()
    
    self.argdict = argdict

    input_dim = argdict["input_dim"]
    output_dim = argdict["output_dim"]
    self.layer = nn.Linear(input_dim, output_dim)
    self.activation = nn.Softmax(dim = -1)
  
  def forward(self, x, temperature = 1.0):
    x = self.layer(x)
    y = gumbel_sampler(x, temperature)
    return self.activation(x), y # logits, y

class gaussian(nn.Module):
  def __init__(self, argdict):
    super(gaussian, self).__init__()

    self.argdict = argdict
    input_dim = argdict["input_dim"]
    output_dim = argdict["output_dim"]

    self.mu_layer = nn.Linear(input_dim, output_dim)
    self.var_layer = nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.Softplus() # need softplus
    )

  def forward(self, x):
    mu = self.mu_layer(x)
    var = self.var_layer(x)
    z = gaussian_sampler(mu, var)
    return mu, var, z    

class encoder(nn.Module):
  def __init__(self, argdict):
    super(encoder, self).__init__()

    self.argdict = argdict
    
    # q(y|x)
    self.q_y_rnn = torch.nn.Sequential(
      rnn(argdict["q_y_rnn"]), # rnn component                                    
    ) # make sure to constrain arguments
    
    self.q_y_linear = ff(argdict["q_y_linear"])
    
    self.q_y = softmax_with_gumbel(argdict["q_y_gumbel"]) # separate for temperature parameter
    
    # q(z|y,x)
    self.q_z_recurrent = rnn(argdict["q_z_rnn"])
    
    self.q_z = torch.nn.Sequential( # rnn?? can remove if buggy
      ff(argdict["q_z_linear"]),
      gaussian(argdict["q_z_gaussian"])                                          
    ) # make sure to constrain arguments

    if argdict["cuda"]:
        self.q_y_linear = self.q_y_linear.cuda()
  
  def forward_fixed_y(self, x, y_fixed):
    # for style transfer
    x = self.q_z_recurrent(x)[0][:,-1,:]
    
    mu, var, z = self.q_z(torch.cat((x, y_fixed), dim=1))
    return_dict = {'mu': mu, 'var': var, 'z': z}
    return return_dict

  def forward(self, x, temperature = 1.0):
    pre_y = self.q_y_rnn(x)[0][:,-1,:]
    
    pre_y = self.q_y_linear(pre_y)
    pi, y = self.q_y(pre_y, temperature = temperature)
    
    x = self.q_z_recurrent(x)[0][:,-1,:]
    mu, var, z = self.q_z(torch.cat((x, y), dim=1))
    
    return_dict = {'pi': pi, 'y': y, 'mu': mu, 'var': var, 'z': z}
    return return_dict

class decoder(nn.Module):
  def __init__(self, argdict):
    super(decoder, self).__init__()

    self.argdict = argdict
    input_dim = self.argdict["input_dim"] # y_dim
    output_dim = self.argdict["output_dim"] # z_dim
    # make sure dims match when constructing args

    self.p_z_mu_nn = nn.Linear(input_dim, output_dim)
    self.p_z_var_nn = nn.Sequential(
      nn.Linear(input_dim, output_dim),
      nn.Softplus()
    )

    self.p_x = torch.nn.Sequential(
      ff(argdict["p_x_linear"]),
      rnn(argdict["p_x_rnn"]),
    ) # apply activations on output
    
    # technically the mean vector should not be a sequential, but this is too annoying to implement

  def forward(self, z, y, seq_len):
    z_mu = self.p_z_mu_nn(y)
    z_var = self.p_z_var_nn(y)
    
    z = z.view(z.shape[0], 1, z.shape[1])
    
    z_seq = z.repeat(1, seq_len, 1) # tiling for sequence
    
    x = self.p_x(z_seq)[0]

    return_dict = {'mu': z_mu, 'var': z_var, 'x': x}
    return return_dict

class GMVAE(nn.Module):
  def __init__(self, argdict):
    super(GMVAE, self).__init__()

    self.encoder = encoder(argdict["encoder"])
    self.decoder = decoder(argdict["decoder"])
    
    # weight initialization
    for m in self.modules():
      if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
      #elif type(m) == nn.RNN or type(m) == nn.GRU or type(m) == nn.LSTM:
      #  torch.nn.init.kaiming_normal_(m.weight) # RNN weighting
      # don't do this, too complicated

  def style_transfer(self, x, y_fixed):
    # y_fixed should be binary vector with the target style

    encoder_returns = self.encoder.forward_fixed_y(x, y_fixed)
    z = encoder_returns['z']
    decoder_returns = self.decoder(z, y_fixed)
    
    return_dict = {"encoder": encoder_returns, "decoder": decoder_returns}
    return return_dict

  def forward(self, x, temperature=1.0):
    # standard
    encoder_returns = self.encoder(x, temperature = temperature)
    z, y = encoder_returns['z'], encoder_returns['y']
    decoder_returns = self.decoder(z, y, x.shape[1])
    
    return_dict = {"encoder": encoder_returns, "decoder": decoder_returns}
    return return_dict

class Model():
  def __init__(self, argdict):
    # unpacking 50000000 args
    # lr control

    # lr variable
    self.init_learning_rate = argdict["learning_rate"]
    self.learning_rate = self.init_learning_rate

    # decay parameters
    self.decay_epoch = argdict["decay_epoch"]
    self.lr_decay = argdict["lr_decay"]

    # weighting for loss
    self.weight_style = argdict["weight_style"]
    self.weight_entropy = argdict["weight_entropy"]
    self.weight_sampling = argdict["weight_sampling"]

    # mix different audio??   
    self.weight_pitch = argdict["weight_pitch"]

    # self.weight_instrument = argdict["weight_instrument"]
    self.weight_velocity = argdict["weight_velocity"]

    # sizes, make sure it matches
    self.pitch_size = argdict["pitch_size"]
    self.instrument_size = argdict["instrument_size"]
    self.velocity_size = argdict["velocity_size"]

    # temperature for sampling for GMM, very annoying
    self.init_temp = argdict["init_temp"]
    self.decay_temp = argdict["decay_temp"]
    self.min_temp = argdict["min_temp"]
    self.decay_temp_rate = argdict["decay_temp_rate"]

    # temperature variable
    self.gumbel_temp = self.init_temp

    # epochs, etc
    self.num_epochs = argdict["num_epochs"]
    self.save_epoch = argdict["save_epoch"]
    self.log_epoch = argdict["log_epoch"]
    
    self.save_path = argdict["save_path"]
    self.logdir = argdict["logdir"]
    Path(self.logdir).mkdir(parents=True,exist_ok=True)
    
    if not os.path.exists(self.save_path):
        os.makedirs(self.save_path)

    self.model = GMVAE(argdict)
    self.use_cuda = argdict["cuda"]
    if self.use_cuda:
      self.model = self.model.cuda()
      #print(next(self.model.parameters()).is_cuda)

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

  def _elbo(self, pitch, instrument, velocity, style_label):
    
    x = torch.cat((pitch, instrument, velocity), dim=-1) # should be dim = 3
    if self.use_cuda:
        x = x.cuda()
        pitch = pitch.cuda()
        instrument = instrument.cuda()
        velocity = velocity.cuda()
        style_label = style_label.cuda()
        #print(x.is_cuda)
    return_dict = self.model(x, temperature = self.gumbel_temp)

    x_pred = return_dict["decoder"]["x"]
    pitch_pred, instrument_pred, velocity_pred = \
      torch.split(x_pred, [self.pitch_size, self.instrument_size, self.velocity_size], dim=-1)

    # renormalizing?
    velocity_pred = (velocity_pred + 1)/2 # is this correct? output of RNN should be in [0, 1]

    pitch_label = torch.argmax(pitch, dim=2).view(-1)
    pitch_pred = torch.reshape(pitch_pred, (-1, self.pitch_size))
    

    loss_pitch = cross_entropy(pitch_pred, pitch_label)
    # loss_instrument = cross_entropy(instrument_pred, torch.argmax(instrument, dim = -1))
    loss_velocity = mse(velocity, velocity_pred)

    style_logits = return_dict["encoder"]["pi"]
    y_pred = return_dict["encoder"]["y"]

    # style label may need copying
    loss_style = cross_entropy(style_logits, style_label.long()) # style label is not 1-hot?
    loss_entropy = entropy(style_logits, y_pred)

    z_pred = return_dict["encoder"]["z"]
    new_mu, new_var = return_dict["encoder"]["mu"], return_dict["encoder"]["var"]
    old_mu, old_var = return_dict["decoder"]["mu"], return_dict["decoder"]["var"]

    loss_kl = gaussian_kl(z_pred, new_mu, new_var, old_mu, old_var)

    loss_total = loss_pitch * self.weight_pitch + loss_velocity * self.weight_velocity + \
      loss_style * self.weight_style + loss_entropy * self.weight_entropy + \
      loss_kl * self.weight_sampling # + loss_instrument * self.weight_instrument

    stats_dict = {'kl': loss_kl, 'entropy': loss_entropy, 'ce_style': loss_style,
                  'ce_pitch': loss_pitch, #'ce_instrument': loss_instrument,
                  'mse_velocity': loss_velocity, 'total': loss_total}
    return stats_dict
    
  def _step(self, data, update=True):
    # get _elbo, optimize

    # data needs to be: pitch, instrument, velocity, style_label
    # pitch [N, T, X], instrument [N, T, Y], velocity [N, T, 1] ?
    # style label is just [N, Z]

    pitch, instrument, velocity, style = \
      data["pitch"], data["instrument"], data["velocity"], data["style"]
    stats_dict = self._elbo(pitch, instrument, velocity, style)

    final_loss = stats_dict["total"]
    
    if update:  
      self.optimizer.zero_grad()
      final_loss.backward()
      self.optimizer.step() # can use clipping, etc

    return stats_dict

  def train(self, data_loader):
    # iterate on data loader using step

    # decay lr and temp appropriately
    
    all_stats_dict = defaultdict(list)

    writer = SummaryWriter(log_dir=self.logdir)

    widgets = [
        progressbar.Percentage(),
        progressbar.Bar(),
        ' Adaptive ', progressbar.AdaptiveETA(),
        ', ',
        progressbar.Variable('epoch'),
        
    ]

    n = 0
    with progressbar.ProgressBar(max_value = self.num_epochs,widgets = widgets) as bar:
        for epoch in range(self.num_epochs):
            epoch_dict = defaultdict(list)

            for batch in data_loader:
                stats = self._step(batch)

                for key in stats:            
                    epoch_dict[key].append(stats[key].item()) # should be scalar
                    writer.add_scalar(key, stats[key].item(), n)
                    n += 1
            # aggregate stats
            for key in epoch_dict:
                epoch_dict[key] = np.mean(epoch_dict[key])
                all_stats_dict[key].append(epoch_dict[key])

            # do all epoch based updates
            if epoch % self.decay_epoch == 0:
                self.learning_rate *= self.lr_decay
                
            if epoch % self.decay_temp_rate == 0:
                self.gumbel_temp = max(self.gumbel_temp * self.decay_temp, self.min_temp)

            # save and log
            if epoch % self.save_epoch == 0:
                torch.save(self.model.state_dict(), self.save_path + "model_{}".format(epoch))
            
            if epoch % self.log_epoch == 0:
                with open(self.save_path + "stats_{}".format(epoch), 'w') as f:
                    json.dump(all_stats_dict, f)

                for key in epoch_dict:
                    logging.info(key + " : {}".format(epoch_dict[key]))

            bar.update(epoch, epoch=epoch)

  def test(self, data_loader):
    # iterate
    epoch_dict = defaultdict(list)

    for batch in data_loader:
      stats = self._step(batch, update=False)

      for key in stats:
        epoch_dict[key].append(stats[key].item) 
    
    for key in epoch_dict:
      epoch_dict[key] = np.mean(epoch_dict[key])
    
    return epoch_dict()
    

  def run(self, train_loader, test_loader):
    # train, test then plot outputs?? save model somehow
    print("Beginning run. Starting training")
    train_results = self.train(train_loader)
    print("training completed. Running testing.")
    test_results = self.test(test_loader)

    # plot outputs or something?
    # pass
    return train_results, test_results

  def transfer(self, data):
    # use the style transfer function in GMVAE

    # don't use data loader, just load data in directly
    pitch, instrument, velocity, style = \
      data["pitch"], data["instrument"], data["velocity"], data["style"]
    
    style = style.expand(list(style_label.shape)[0], list(pitch.shape)[1], list(style_label.shape)[1])
    x = torch.cat(pitch, instrument, velocity, dim=-1)
    if self.use_cuda:
      x= x.cuda()
    return_dict = self.model.style_transfer(x, style)
    
    x_pred = return_dict["decoder"]["x"]
    pitch_pred, instrument_pred, velocity_pred = \
      torch.split(x_pred, [self.pitch_size, self.instrument_size, self.velocity_size], dim=-1)

    # renormalizing?
    velocity_pred = (velocity_pred + 1)/2 # is this correct? output of RNN should be in [0, 1]
    
    return pitch_pred.numpy(), instrument_pred.numpy(), velocity_pred.numpy() # is this good?

  def load_weights(self, weights_path):
    state = torch.load(weights_path)
    self.model.load_state_dict(state)
    print("Weights successfully loaded from", weights_path)
    return True

class music(Dataset):
    """music"""
    def __init__(self, pitch_data, instrument_data, velocity_data, style_data):
        """
        # pitch [N, T, X], instrument [N, T, Y], velocity [N, T, 1] ?
        # style label is just [N, Z]
        """
        self.pitch_data = pitch_data
        self.instrument_data = instrument_data
        self.velocity_data = velocity_data
        self.style_data = style_data

    def __len__(self):
        return len(self.style_data)

    def __getitem__(self, idx):
        p = torch.tensor(self.pitch_data[idx], dtype=torch.float)
        i = torch.tensor(self.instrument_data[idx], dtype=torch.float)
        v = torch.tensor(self.velocity_data[idx], dtype=torch.float)
        s = torch.tensor(self.style_data[idx], dtype=torch.int)
            
        f = lambda z: torch.nn.utils.rnn.pad_sequence(z, batch_first=True)
        
        sample = {"pitch": f(p), "instrument": f(i), "velocity": f(v), "style": s}
        
        return sample