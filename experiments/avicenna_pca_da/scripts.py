from framework.scripts.example_pca_da import main_train

def train_pca_da(state, channel):
  
    solution = state.solution
    batch_size = state.batch_size
    epochs = state.epochs
    sparse_penalty = state.sparse_penalty
    sparsityTarget = state.sparsityTarget
    sparsityTargetPenalty = state.sparsityTargetPenalty
                     

    out = main_train(epochs,batch_size,solution,sparse_penalty,sparsityTarget,sparsityTargetPenalty)

    state.alc_on_train = out[0]
    state.final_cost = out[1]
  

    return channel.COMPLETE

