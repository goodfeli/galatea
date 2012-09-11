import theano

class _ElemwiseNoGradient(theano.tensor.Elemwise):

    def connection_pattern(self, node):

        return [ [ False ] ]

    def grad(self, inputs, output_gradients):

        return [ theano.gradient.DisconnectedType()() ]

# Call this on a theano variable to make a copy of that variable
# No gradient passes through the copying operation
# This is equivalent to making my_copy = var.copy() and passing
# my_copy in as part of consider_constant to tensor.grad
# However, this version doesn't require as much long range
# communication between parts of the code
block_gradient = _ElemwiseNoGradient(theano.scalar.identity)
