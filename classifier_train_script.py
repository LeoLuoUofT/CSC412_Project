from classifier_functions import *
from project_functions import *
from import_midi import *

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

data_folder = "./full_pickles/"
BATCH_SIZE = 4096
NUM_EPOCHS = 100

v, vt, _, _, _, _, i, it, _, _, \
    p, pt, c, ct, _, _ = import_midi_from_folder(data_folder, load_from_pkl = True, class_list = ['Classic', 'Jazz', 'Pop']) # test
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
    temp_labels = [ct[i]] * pt[i].shape[0]
    labels_test.extend(temp_labels)

# print(labels_test.count(0)/len(labels_test))
# print(labels_test.count(1)/len(labels_test))
# print(labels_test.count(2)/len(labels_test))

# instruments are too difficult to parse
# just leave it as a random matrix
v = np.concatenate(v, axis=0)[:,:,np.newaxis]
vt = np.concatenate(vt, axis=0)[:,:,np.newaxis]

p = np.concatenate(p, axis=0)
pt = np.concatenate(pt, axis=0)


i = np.zeros(v.shape)
it = np.zeros(vt.shape)


train_data = music(p, i, v, labels)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print("data loader created")

test_data = music(pt, it, vt, labels_test)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = MusicGRU(62, 256, 3)
model = model.cuda()

# to train model
# print("beginning training")
# train_network(model, train_dataloader, test_dataloader, num_epochs = NUM_EPOCHS, lr=1e-2)

# to evaluate pretrained model
# state = torch.load("output/classifier/Apr_19_2021_03")
# model.load_state_dict(state)
# print("model loaded")

# acc = get_accuracy(model, test_dataloader)
# print(acc)
