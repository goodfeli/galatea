import theano
import theano.tensor as T
from theano.sandbox import linalg
import numpy, scipy
import scipy.optimize
from mlp_for_mackey_glass import MLP_Mackey_Glass
from early_stopping_trainer import EarlyStoppingTrainer
import os

floatX = theano.config.floatX

class Mackey_Glass:
    def __init__(self):
        
        self.a = 0.2
        self.b = 0.1
        self.tao = 17
        
        self.dataset = numpy.zeros((6000,))
        
        self.generate_datasets()
        

        self.trainset = [self.trainset_x, self.trainset_y]
        self.validset = [self.validset_x, self.validset_y]
        self.testset = [self.testset_x, self.testset_y]
        
    def generate_datasets(self):
        if os.path.exists('mackey_glass.npz'): 
                print 'loading data'
                self.dataset = numpy.load('mackey_glass.npz')['arr_0']
                                
        else:
            for i in xrange(self.dataset.shape[0]-1):
                if i <= self.tao:
                    self.dataset[i+1] = numpy.random.randn(1)
                
                else:
                    self.dataset[i+1] = ((1 - self.b) * self.dataset[i] +
                        self.a * self.dataset[i-self.tao] / ((1+(self.dataset[i-self.tao])**10)))
            numpy.savez('mackey_glass.npz', self.dataset)
                    
        self.trainset_x = numpy.zeros((500,4))
        self.trainset_y = numpy.zeros((500,1))
        for i in xrange(201,701):

            self.trainset_x[i-201] = self.dataset[[i, i-6, i-12, i-18]]
            self.trainset_y[i-201] = self.dataset[i+6]

        self.validset_x = numpy.zeros((500,4))
        self.validset_y = numpy.zeros((500,1))
        for i in xrange(1001,1501):
            self.validset_x[i-1001] = self.dataset[[i, i-6, i-12, i-18]]
            self.validset_y[i-1001] = self.dataset[i+6]

        
        self.testset_x = numpy.zeros((500,4))
        self.testset_y = numpy.zeros((500,1))
        for i in xrange(5001,5501):
            self.testset_x[i-5001] = self.dataset[[i, i-6, i-12, i-18]]
            self.testset_y[i-5001] = self.dataset[i+6]

class Regressor:
    def __init__(self, trainset, validset, testset, batch_size=10):
        self.learning_rate = 0.005
        self.batch_size = batch_size
        
        self.train_x = theano.shared(numpy.asarray(trainset[0], dtype=theano.config.floatX))
        self.train_y = theano.shared(numpy.asarray(trainset[1], dtype=theano.config.floatX))
        self.valid_x = theano.shared(numpy.asarray(validset[0], dtype=theano.config.floatX))
        self.valid_y = theano.shared(numpy.asarray(validset[1], dtype=theano.config.floatX))
        self.test_x = theano.shared(numpy.asarray(testset[0], dtype=theano.config.floatX))
        self.test_y = theano.shared(numpy.asarray(testset[1], dtype=theano.config.floatX))

        

        #self.build_theano_fn_sgd()
        self.build_theano_fn_ng_approx()
        #self.build_theano_fn_ng_exact()

        # compute number of minibatches for training, validation and testing
        self.n_train_batches = self.train_x.get_value(borrow=True).shape[0] / batch_size
        self.n_valid_batches = self.valid_x.get_value(borrow=True).shape[0] / batch_size
        self.n_test_batches = self.test_x.get_value(borrow=True).shape[0] / batch_size
        

        self.trainer = EarlyStoppingTrainer(self.n_train_batches, self.n_valid_batches, self.n_test_batches, n_epoch=2000)
        
    def build_theano_fn_ng_exact(self):
        rng = numpy.random.RandomState(1234)
        index = T.scalar(dtype='int32')
        x = T.matrix(name='x', dtype=floatX)
        y = T.matrix(name='y', dtype=floatX)
        minibatch_size = x.shape[0]

        self.model = MLP_Mackey_Glass(rng, input=x, n_in=4, n_hidden=10, n_out=1)
        
        self.Gs = theano.shared(numpy.eye(self.model.theta.get_value().shape[0], dtype=floatX))
                        
        outputs = T.flatten(self.model.predict)
        
        cost = T.mean(T.sqr(y - self.model.predict))

        # each param has a Jacobian
        J = theano.gradient.jacobian(outputs, self.model.theta) 
                
        # FIM estimated on minibatch
        G = T.dot(J.T, J) / minibatch_size

        G_inv = linalg.matrix_inverse(G)
        grad_theta = T.grad(cost, self.model.theta)
        
        updates = {}
        delta_theta = T.dot(G_inv, grad_theta)
        updates[self.model.theta] = self.model.theta - self.learning_rate * delta_theta

        self.delta_theta_fn = theano.function(
                   inputs=[index],
                   outputs=G_inv,
                   givens={
                      x: self.train_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.train_y[index * self.batch_size:(index + 1) * self.batch_size]},
                   on_unused_input='warn'
                   )
        self.G_fn = theano.function(
                   inputs=[index],
                   outputs=G,
                   givens={
                      x: self.train_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.train_y[index * self.batch_size:(index + 1) * self.batch_size]},
                   on_unused_input='warn'
                   )
        
        self.G_inv_fn = theano.function(
                   inputs=[index],
                   outputs=G_inv,
                   givens={
                      x: self.train_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.train_y[index * self.batch_size:(index + 1) * self.batch_size]},
                   on_unused_input='warn'
                   ) 
        self.J_fn = theano.function(
                   inputs=[index],
                   outputs=J,
                   givens={
                      x: self.train_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.train_y[index * self.batch_size:(index + 1) * self.batch_size]},
                   on_unused_input='warn'
                   )
        
        self.params_fn = theano.function(
                   inputs=[],
                   outputs=self.model.theta,
                   )
        
        self.grads_fn = theano.function(
                   inputs=[index],
                   outputs=grad_theta,
                   givens={
                      x: self.train_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.train_y[index * self.batch_size:(index + 1) * self.batch_size]})
        
        self.train_cost_fn = theano.function(
                   inputs=[index],
                   outputs=cost,
                   givens={
                      x: self.train_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.train_y[index * self.batch_size:(index + 1) * self.batch_size]})
        
        
        self.valid_cost_fn = theano.function(
                   inputs=[index],
                   outputs=cost,
                   givens={
                      x: self.valid_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.valid_y[index * self.batch_size:(index + 1) * self.batch_size]})

        self.test_cost_fn = theano.function(
                   inputs=[index],
                   outputs=cost,
                   givens={
                      x: self.test_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.test_y[index * self.batch_size:(index + 1) * self.batch_size]})
        
        self.learn_fn = theano.function(
                   inputs=[index],
                   outputs=cost,
                   updates=updates, 
                   givens={
                      x: self.train_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.train_y[index * self.batch_size:(index + 1) * self.batch_size]})
        
    def build_theano_fn_ng_approx(self):
        alpha = T.scalar(dtype=floatX)
        
        rng = numpy.random.RandomState(1234)
        index = T.scalar(dtype='int32')
        x = T.matrix(name='x', dtype=floatX)
        y = T.matrix(name='y', dtype=floatX)
        
        self.model = MLP_Mackey_Glass(rng, input=x, n_in=4, n_hidden=10, n_out=1)
        

        self.G_inv = theano.shared(numpy.eye(self.model.theta.get_value().shape[0], dtype=floatX))
        
        output = T.mean(self.model.predict)
        
        cost = T.mean(T.sqr(y - self.model.predict))

        grad_F = T.grad(output, self.model.theta)
            
        grad_theta = T.grad(cost, self.model.theta)
        
        updates = {}
        updates[self.model.theta] = self.model.theta - self.learning_rate * T.dot(self.G_inv, grad_theta)
        t = T.dot(self.G_inv, grad_F) 
        updates[self.G_inv] = (1 + alpha) * self.G_inv - alpha * T.dot(t, t.T) 

        self.G_inv_fn = theano.function(
                   inputs=[],
                   outputs=self.G_inv,
                   )
        
        self.params_fn = theano.function(
                   inputs=[],
                   outputs=self.model.theta,
                   )
        
        self.grads_fn = theano.function(
                   inputs=[index],
                   outputs=grad_theta,
                   givens={
                      x: self.train_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.train_y[index * self.batch_size:(index + 1) * self.batch_size]})
        

        self.train_cost_fn = theano.function(
                   inputs=[index],
                   outputs=cost,
                   givens={
                      x: self.train_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.train_y[index * self.batch_size:(index + 1) * self.batch_size]})
        
        
        self.valid_cost_fn = theano.function(
                   inputs=[index],
                   outputs=cost,
                   givens={
                      x: self.valid_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.valid_y[index * self.batch_size:(index + 1) * self.batch_size]})

        self.test_cost_fn = theano.function(
                   inputs=[index],
                   outputs=cost,
                   givens={
                      x: self.test_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.test_y[index * self.batch_size:(index + 1) * self.batch_size]})
        
        self.learn_fn = theano.function(
                   inputs=[index, alpha],
                   outputs=cost,
                   updates=updates, 
                   givens={
                      x: self.train_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.train_y[index * self.batch_size:(index + 1) * self.batch_size]})

        
    def build_theano_fn_sgd(self):
        index = T.scalar(dtype='int32')
        rng = numpy.random.RandomState(1234)
        x = T.matrix(name='x', dtype=floatX)
        y = T.matrix(name='y', dtype=floatX)
        
        self.model = MLP_Mackey_Glass(rng, input=x, n_in=4, n_hidden=10, n_out=1)

        predict_model = self.model.predict

        cost = T.mean(T.sqr(y - self.model.predict))

        grad_theta = T.grad(cost, self.model.theta)
        updates = {}
        updates[self.model.theta] = self.model.theta - self.learning_rate * grad_theta
        
        self.grads_fn = theano.function(
                   inputs=[index],
                   outputs=grad_theta,
                   givens={
                      x: self.train_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.train_y[index * self.batch_size:(index + 1) * self.batch_size]})
        
        self.train_cost_fn = theano.function(
                   inputs=[index],
                   outputs=cost,
                   givens={
                      x: self.train_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.train_y[index * self.batch_size:(index + 1) * self.batch_size]})
        
        
        self.valid_cost_fn = theano.function(
                   inputs=[index],
                   outputs=cost,
                   givens={
                      x: self.valid_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.valid_y[index * self.batch_size:(index + 1) * self.batch_size]})

        self.test_cost_fn = theano.function(
                   inputs=[index],
                   outputs=cost,
                   givens={
                      x: self.test_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.test_y[index * self.batch_size:(index + 1) * self.batch_size]})
        
        self.learn_fn = theano.function(
                   inputs=[index],
                   outputs=cost,
                   updates=updates, 
                   givens={
                      x: self.train_x[index * self.batch_size:(index + 1) * self.batch_size],
                      y: self.train_y[index * self.batch_size:(index + 1) * self.batch_size]})

    def cost_scipy(self, params):
        params = self.reshape_for_scipy(params, method='tighten')
        for i, param in enumerate(params):
            self.model.params[i].set_value(param)
        cost = self.train_cost_fn(self.index)
        return cost
    
    def grads_scipy(self, params):
        
        params = self.reshape_for_scipy(params, method='tighten')
        for i, param in enumerate(params):
            self.model.params[i].set_value(param)
        g = self.grads_fn(self.index)
        g = self.reshape_for_scipy(g, method='flatten')
        return g
        
    def learn_fn_scipy(self, index):
        self.index = index
        
        params = [param.get_value() for param in self.model.params]
        x0 = self.reshape_for_scipy(params, method='flatten')
        xopt = scipy.optimize.fmin_cg(
                                f=self.cost_scipy,
                                x0=x0,
                                fprime=self.grads_scipy
        )
                
        opt = self.reshape_for_scipy(xopt, method='tighten')
        for i, param in enumerate(opt):
            self.model.params[i].set_value(param)
        cost = self.train_cost_fn(self.index)

        return cost
        
    def reshape_for_scipy(self, inputs, method='flatten'):
        """
        flatten or unflatten params
        """
        flatten = lambda x:reduce(lambda x,y:x+y,map(flatten,x)) if isinstance(x[0],list) else x
        

        params_shapes = [param.get_value().shape for param in self.model.params]
        params_flatten_range = []
        for shape in params_shapes:
            if len(shape) == 2:
                params_flatten_range.append(shape[0] * shape[1])
            if len(shape) == 1:
                params_flatten_range.append(shape[0])
        params_indices = [0]
        for i in params_flatten_range:
            params_indices.append(params_indices[-1] + i)
            
        if method == 'flatten':
            rval = [list(inp.flatten()) for inp in inputs]
            return numpy.array(flatten(rval))
        if method == 'tighten':
            rval = [inputs[params_indices[i]: params_indices[i+1]] for i in range(len(params_indices)-1)]
            rval = [numpy.array(val).reshape(params_shapes[i]) for i, val in enumerate(rval)]
            return rval
        
if __name__ == '__main__':
    
    dataset = Mackey_Glass()
    trainset = dataset.trainset
    validset = dataset.validset
    testset = dataset.testset
    
    regressor = Regressor(trainset, validset, testset, 1)
    regressor.trainer.train(regressor, learn_fn_scipy=False)