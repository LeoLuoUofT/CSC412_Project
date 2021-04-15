from classifier_functions import *

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

# extract features to pickle
#classes = ["Classic", "Jazz", "Pop"]
classes = ["Jazz"]
class_to_idx = {classes[i]:i for i in range(len(classes))}

data_folder = "/home/brian/Data/CSC412_Project/full_data"
load_from_pkl = False # if set to false, extracting features will save the processed features to a pkl file 
pkl_file = "processed_data.pickle"
path_df = get_matched_midi(data_folder, classes)

data = extract_midi_features(path_df, class_to_idx, load_from_pkl, pickle_file = pkl_file)

X_train, X_test, y_train, y_test = train_test_split(data[:,:-2], data[:,-1], test_size=0.1, random_state=0,shuffle=True)

y_train = one_hot(y_train,len(classes))
y_test = one_hot(y_test, len(classes))

# Training model here
model = MLPClassifier(solver='adam', hidden_layer_sizes=(500, 200, 50), random_state=1)
print("beginning training")
model = model.fit(X,y)
print("model finished training.")
acc = get_accuracy(model, x_test, y_test)
print("accuracy of model: {}".format(acc))

with open("model_"+str(int(acc)), 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)