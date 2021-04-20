import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

import pretty_midi
import warnings
import os
import pickle

def train_network(model, train_loader, valid_loader, num_epochs=1, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda= lambda epoch: 0.95)

    log_path = os.path.join("logs", "classifiers", datetime.today().strftime("%b_%d_%I"))
    Path(log_path).mkdir(parents=True,exist_ok=True)
    writer = SummaryWriter(log_dir = log_path)

    model_folder = os.path.join("output", "classifier", datetime.today().strftime("%b_%d_%I"))
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    

    losses, train_acc, valid_acc, epochs = [], [], [], []
    n = 0
    for epoch in range(num_epochs):
        for batch in train_loader:

            labels = batch["style"]
            labels = labels.long()

            pitch = batch["pitch"]
            velocity = batch["velocity"]
            
            x = torch.cat((pitch, velocity), dim=-1)

            x = x.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            pred = model(x)

            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss/train", loss,n)
            n+=1
        losses.append(float(loss))

        epochs.append(epoch)
        t_acc = get_accuracy(model, train_loader)
        v_acc = get_accuracy(model, valid_loader)
        train_acc.append(t_acc)
        valid_acc.append(v_acc)
        print("Epoch {}, val acc {}".format(epoch, v_acc))
        writer.add_scalar("accuracy/train", t_acc,epoch)
        writer.add_scalar("accuracy/valid", v_acc,epoch)
        
        model_path = os.path.join(model_folder, "epoch" + str(epoch))
        torch.save(model.state_dict(), model_path)
        scheduler.step()

    

def get_accuracy(model, data_loader):

    """ Compute the accuracy of the `model` across a dataset `data`
    Example usage:
    >>> model = MyRNN() # to be defined
    >>> get_accuracy(model, valid) # the variable `valid` is from above
    """

    correct, total = 0, 0
    for batch in data_loader:
        labels = batch["style"]
        pitch = batch["pitch"]
        velocity = batch["velocity"]

        x = torch.cat((pitch, velocity), dim=-1)

        if torch.cuda.is_available():
            x = x.cuda()
            labels = labels.cuda()
        output = model(x)
        #print(output.squeeze()[0])
        
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.shape[0]
    return correct / total

# def transferred_acc(predicted_labels, old_labels, expected_transfer):


class MusicGRU(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(MusicGRU,self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size,hidden_size,batch_first=True)
        self.fc1 = nn.Linear(hidden_size*2, 50)
        self.fc2 = nn.Linear(50,num_classes) #input is twice the size because of the concatenated concat average and max pooling
        # self.fc3 = nn.Linear(25,num_classes)
    def forward(self,x):
        # x = self.emb(x)
        if torch.cuda.is_available():
            x = x.cuda()
        h0 = torch.zeros(1,x.size(0),self.hidden_size)
        out, __ = self.rnn(x)
        #out = self.fc(torch.max(out, dim=1)[0])
        out = torch.cat([torch.max(out, dim=1)[0], torch.mean(out, dim=1)], dim=1)
        out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        out = self.fc2(out)
        return out