'''
This script will take in the test dataset and run 3 types of style transfers

    1. Jazz and Pop to Classical
    2. Classical and Pop to Jazz
    3. Classical and Jazz to Pop

Each style transfer will then be evaluated by the classifier.
It is hoped that the classifier will predict the transferred songs belong in their transferred style.
This accuracy will be compared against the classifier's accuracy on the true class.

'''
from import_midi import import_midi_from_folder
from project_functions import *
from classifier_functions import *

# constant variables
PITCH_DIM = 61
INSTRUMENT_DIM = 1 # same as pitch pls?
VELOCITY_DIM = 1 

NUM_STYLES = 3
LATENT_DIM = 64 # ??

BATCH_SIZE = 4096 # bigger if possible, but might kill GPU

RNN_CELL = "gru"
RNN_CELL_NUMBER = LATENT_DIM # this is a lot

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
gmvae_weights_path = "output/model_95"
classifier_weights_path = ""

class_list = ['Classic', 'Jazz', 'Pop']
style_to = 0


v, vt, _, _, _, _, i, it, _, _, \
    p, pt, c, ct, _, _ = import_midi_from_folder(data_folder, load_from_pkl = True, class_list = ['Classic', 'Jazz', 'Pop']) # test
print("data loaded from", data_folder)
    
labels_test = []
transfer_target_label = []
for i in range(len(pt)):
    temp_labels = [ct[i]] * p[i].shape[0]
    labels_test.extend(temp_labels)
    transfer_target_label.extend([style_to]*p[i].shape[0])

vt = np.concatenate(vt, axis=0)[:,:,np.newaxis]
pt = np.concatenate(pt, axis=0)
it = np.zeros(vt.shape)

test_data = music(pt, it, vt, transfer_target_label)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print("loaders created")

gmvae_model = Model(argdict)

gmvae_model.load_weights(gmvae_weights_path)
    
classifier = MusicGRU(62, 256, 3)
state = torch.load(classifier_weights_path)
classifier.model.load_state_dict(state)
classifier = classifier.cuda()

for batch in dataloader:
    transferred_pitch, __, transferred_vel = gmvae_model.transfer(batch)
    x = torch.cat((transferred_pitch, transferred_vel), dim=-1)
    x = x.cuda()

    output = classifier(x)
    pred = output.max(1,keepdim=True)[1]
predicted_labels = classifier