"""
You can do natural gradient by using a few conjugate gradient
steps to approximately invert the covariance matrix.

Razvan says the right thing to do is use the uncentered covariance.

Let f:X->R^m be a function giving the cost at each example.

If you use conjugate gradient to solve
Ad = grad_theta mean f
you get
d = A^-1 grad_theta mean f
which gives you the search direction you want.
(note: only run it for a few steps and get an
approximate direction)


set d_0 to grad_theta mean f(x)
conjugate gradient algorithm to solve Ad = g

r_0 = g - A d_0
p_0 = r_0
rns = r^T r
for k=0:N
    Ap = A p
    alpha =  rns/(p^T Ap)
    d = d + alpha p
    r = r - alpha Ap
    if r is small:
        break
    rns_new = r^T r
    beta = rns_new / rns
    p = r + beta*p
    rns = rns_new


For A we use the uncentered covariance: (grad_theta f) (grad_theta f)^T
so
A p = (1/m) Lop(f(x), theta, Rop(f(x), theta, p))
(TODO: check this)

For fixed p, the covariance matrix is additive over examples, so this
can be computed by adding it up over minibatches to save memory.


"""


class NG_Wrapper(object):

    def setup(self, cost, model, dataset):


class NaturalGradient(object):

    def __init__(self, f, params, param_constrainers = None,):
        """
            f: A theano expression for the objective to be minimized.
               Should be written entirely in terms of shared variables,
               so that the other methods can work in terms of a load_batch
               callable that preps the input
               NOTE: since this.
            params: a list of theano shared variables defining the parameters
                to be optimized.
        """
        raise NotImplementedError()

    def minimize(self, load_batch, num_batches):
        self.compute_direction(load_batch, num_batches)
        self.line_search(load_batch, num_batches)

    def line_search(self, load_batch, num_batches):
        self.store_params()
        best_obj = self.compute_objective(load_batch, num_batches)
        best_alpha = 0.
        print '\t',0.,':',best_obj
        for alpha in [0.5, 1., 1.5]:
            self.goto_alpha(alpha)
            obj = self.compute_objective(load_batch, num_batches)
            if obj < best_obj:
                best_obj = obj
            print '\t',alpha,':',obj


