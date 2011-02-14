#TODO, sparse fonction remains to be implemented
from sparse.dA import main_train

def train_dA(state, channel):
    dataset = state.dataset
    save_dir = state.savedir
    n_hidden = state.nhidden
    tied_weights = state.tied_weights
    act_enc = state.act_enc
    act_dec = state.act_dec
    learning_rate = state.lr
    batch_size = state.batchsize
    epochs = state.epochs
    cost_type  = state.cost_type
    noise_type =  state.noise_type
    corruption_level = state.corruption_level
    normalized_data = state.normalize

    out = main_train(dataset, save_dir, n_hidden, tied_weights, act_enc,
        act_dec, learning_rate, batch_size, epochs, cost_type,
        noise_type, corruption_level, normalized_data)

    state.denoising_error = out[0]
    state.time_spent = out[1]

    return channel.COMPLETE
