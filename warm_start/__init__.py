import numpy as np
import warnings

from pylearn2.space import NullSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.utils import function
from pylearn2.utils import grad
from pylearn2.utils import safe_zip
from pylearn2.utils import serial

class WarmStart(TrainExtension):

    def __init__(self, num_basis_vectors, num_points, scale, max_jump_norm = 1.,
            method = 'gradient', fitting_cost = 'mse', include_root = False,
            num_applications = -1, psd = False, use_solver = False, reps=1):
        self.__dict__.update(locals())
        del self.self
        self.batch_size = 1000
        self.rng = np.random.RandomState([2014, 5, 8, 2])

    def setup(self, model, dataset, algorithm):
        """
        Train calls this immediately upon instantiation,
        before any monitoring is done.

        This subclass uses it to warm-start the parameters.

        Parameters
        ----------
        model : pylearn2.models.Model
            The model object being trained.

        dataset : pylearn2.datasets.Dataset
            The dataset object being trained.

        algorithm : pylearn2.training_algorithms.TrainingAlgorithm
            The object representing the training algorithm being
            used to train the model.
            *This must be a TrainingAlgorithm that has a `cost`
            attribute that is a pylearn2 `Cost`, such as `SGD`
            or `BGD`.*
        """

        if self.num_applications == 0:
            return
        self.num_applications -= 1

        for i in xrange(self.reps):
            self.setup_impl(model, dataset, algorithm)

    def setup_impl(self, model, dataset, algorithm):
        cost = algorithm.cost

        root = model.get_param_vector()

        dim = root.size

        rng = self.rng


        points = rng.randn(self.num_points, self.num_basis_vectors)
        points = points.astype(root.dtype)
        points *= self.scale

        if self.include_root:
            points[0, :] = 0.

        if not hasattr(self, 'cost_fn'):
            # Cargo cult all the Pascal bullshit needed to evaluate the fucking cost function now
            # =======================================
            data_specs = cost.get_data_specs(model)
            mapping = DataSpecsMapping(data_specs)
            space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
            source_tuple = mapping.flatten(data_specs[1], return_tuple=True)

            # Build a flat tuple of Theano Variables, one for each space.
            # We want that so that if the same space/source is specified
            # more than once in data_specs, only one Theano Variable
            # is generated for it, and the corresponding value is passed
            # only once to the compiled Theano function.
            theano_args = []
            for space, source in safe_zip(space_tuple, source_tuple):
                name = '%s[%s]' % (self.__class__.__name__, source)
                arg = space.make_theano_batch(name=name,
                                              batch_size=self.batch_size)
                theano_args.append(arg)
            theano_args = tuple(theano_args)

            # Methods of `cost` need args to be passed in a format compatible
            # with data_specs
            nested_args = mapping.nest(theano_args)
            fixed_var_descr = cost.get_fixed_var_descr(model, nested_args)
            self.on_load_batch = fixed_var_descr.on_load_batch

            cost_value = cost.expr(model, nested_args,
                                        ** fixed_var_descr.fixed_vars)
            # End cargo culting
            # ======================

            print "Compiling cost function..."
            cost_fn = function(theano_args, cost_value)
            self.cost_fn = cost_fn
        else:
            cost_fn = self.cost_fn

        cost_values = np.zeros(self.num_points)


        data = list(dataset.get_batch_design(self.batch_size,
            include_labels=True))
        from pylearn2.utils.one_hot import one_hot
        data[1] = one_hot(data[1])


        if self.method == 'gaussian':
            basis = rng.randn(dim, self.num_basis_vectors).astype(root.dtype)
        elif self.method == 'element':
            basis = np.zeros((dim, self.num_basis_vectors)).astype(root.dtype)
            for i in xrange(self.num_basis_vectors):
                basis[rng.randint(dim), i] = 1.
        elif self.method == 'gradient':
            if not hasattr(self, 'grad_fn'):
                self.grad_fn = function(theano_args, grad(cost_value, model.get_params()))
            grad_fn = self.grad_fn

            basis = np.zeros((dim, self.num_basis_vectors)).astype(root.dtype)
            for i in xrange(self.num_basis_vectors):
                ipt = list(dataset.get_batch_design(1, include_labels=True))
                label = ipt[1]
                assert label.size == 1
                label = label[0]
                one_hot = np.zeros((1, 10,),dtype='float32')
                one_hot[0, label] = 1
                ipt[1] = one_hot
                g = grad_fn(*ipt)
                basis[:,i] = np.concatenate([e.reshape(e.size) for e in g], axis=0)
        else:
            assert False

        basis /= np.sqrt(np.square(basis).sum(axis=0))

        # Orthogonalize basis
        for i in xrange(self.num_basis_vectors):
            v = basis[:,i ].copy()
            for j in xrange(i - 1):
                u = basis[:, j].copy()
                v -= np.dot(u, v) * u
            norm = np.sqrt(np.square(v).sum())
            assert norm > 1e-4
            v /= norm
            basis[:,i] = v


        for i in xrange(self.num_points):
            print "Evaluating cost at point ", i

            point = points[i, :]
            full_point = root + np.dot(basis, point)
            model.set_param_vector(full_point)

            cost_values[i] = cost_fn(*data)
            print cost_values[i]


        from pylearn2.utils import sharedX
        import theano.tensor as T

        print "!!!!!!!! FITTING THE QUADRATIC FUNCTION !!!!!!!!!!!!!!!!!!!"

        if not hasattr(self, 'fit_quad'):
            points = sharedX(points)
            #from theano import config
            #config.compute_test_value = 'raise'
            cost_values = sharedX(cost_values)
            A = sharedX(np.zeros((self.num_basis_vectors, self.num_basis_vectors)))
            if self.psd:
                mat = T.dot(A.T, A)
            else:
                mat = A
            b = sharedX(np.zeros(self.num_basis_vectors))
            c = sharedX(0.)
            half_quad = T.dot(points, mat)
            quad = (points * half_quad).sum(axis=1)
            lin = T.dot(points, b)
            pred = quad + lin + c

            from pylearn2.optimization.batch_gradient_descent import BatchGradientDescent

            mse = T.square(pred - cost_values).mean()
            mae = abs(pred - cost_values).mean()

            obj = locals()[self.fitting_cost]

            fit_quad = BatchGradientDescent(obj, params = [A, b, c],
                    max_iter = self.num_basis_vectors ** 2,
                    verbose = 3, tol = None,
                    init_alpha = None, min_init_alpha = 1e-7,
                    reset_alpha = False, conjugate = True,
                    reset_conjugate = False,
                    line_search_mode = 'exhaustive')
            self.fit_quad = fit_quad
            self.A = A
            self.b = b
            self.c = c
            self.points = points
            self.cost_values = cost_values
        else:
            self.A.set_value(.001 * np.identity(self.A.get_value().shape[0], dtype=self.A.dtype))
            self.b.set_value(self.b.get_value() * 0.)
            self.c.set_value(self.c.get_value() * 0.)
            self.points.set_value(points)
            self.cost_values.set_value(cost_values.astype(self.cost_values.dtype))

        self.fit_quad.minimize()

        print "!!!!!!!!!!!!! FINDING ITS MINIMUM !!!!!!!!!!!!!!!!!!!!!!!!!!!"

        if self.use_solver:
            if self.psd:
                Av = self.A.get_value()
                mat_v = np.dot(Av.T, Av)
            else:
                mat_v = self.A.get_value()
            bv = self.b.get_value()

            # minimize for x^T A x + b^T x + c
            # -> solve 2 A x + b = 0
            # Ax = - b / 2

            print "********** mat_v", mat_v.min(), mat_v.max()
            x, ignored_residuals, ignored_rank, ignored_singular_values = np.linalg.lstsq(mat_v, - 0.5 * bv)
            print "********** soln: ", x.min(), x.mean(), x.max()
            print "********** SVs: ", ignored_singular_values.min(), ignored_singular_values.max()
            assert x.ndim == 1, x.shape
            prod = np.dot(basis, x)
            norm = np.sqrt(np.square(prod).sum())
            print "*************** Moving params by ",norm
            vector = root + prod
            model.set_param_vector(vector)

        else: # use minimizer
            if not hasattr(self, 'fit_params'):
                self.vector = sharedX(points.get_value().mean(axis=0))
                vector = self.vector
                obj = T.dot(T.dot(mat, vector), vector) + T.dot(b, vector)

                def constrain(d):
                    assert vector in d
                    n = d[vector]
                    norm = T.sqrt(T.square(n).sum())
                    desired_norm = T.clip(norm, 0., self.max_jump_norm)
                    d[vector] = n * desired_norm / norm

                self.fit_params = BatchGradientDescent(obj, params=[vector],
                    max_iter = self.num_basis_vectors,
                    verbose = 3, tol=None,
                    param_constrainers = [constrain],
                    init_alpha = None, min_init_alpha = 1e-3,
                    reset_alpha=False, conjugate=True, reset_conjugate=False,
                    line_search_mode='exhaustive')
            else:
                self.vector.set_value(points.mean(axis=0).astype(self.vector.dtype))

            self.fit_params.minimize()

            model.set_param_vector(root + np.dot(basis , self.vector.get_value()))

class Jumpy(WarmStart):

    def on_monitor(self, model, dataset, algorithm):
        self.setup(model, dataset, algorithm)


class FindBasis(TrainExtension):

    def __init__(self, soln_path, save_path):
        self.__dict__.update(locals())
        del self.self

        soln = serial.load(soln_path)
        self.soln = soln.get_param_vector()

    def setup(self, model, dataset, algorithm):
        self.origin = model.get_param_vector()
        direction = self.soln - self.origin
        del self.soln
        norm = np.sqrt(np.square(direction).sum())
        self.unit = direction / norm
        model.monitor.add_channel(
                    name="soln_coord",
                    ipt=None,
                    val=0.,
                    data_specs=(NullSpace(), ''),
                    dataset=model.monitor._datasets[0])
        model.monitor.add_channel(
                    name="residual_norm",
                    ipt=None,
                    val=0.,
                    data_specs=(NullSpace(), ''),
                    dataset=model.monitor._datasets[0])
        self.biggest_residual = -1.

    def on_monitor(self, model, dataset, algorithm):
        cur = model.get_param_vector()
        cur -= self.origin
        coord = np.dot(cur, self.unit)
        proj = self.unit * coord
        residual = cur - proj
        residual_norm = np.sqrt(np.square(residual).sum())
        assert model.monitor.channels['soln_coord'].val_record[-1] == 0.
        model.monitor.channels['soln_coord'].val_record[-1] = coord
        assert model.monitor.channels['residual_norm'].val_record[-1] == 0.
        model.monitor.channels['residual_norm'].val_record[-1] = residual_norm
        if residual_norm > self.biggest_residual:
            self.biggest_residual = residual_norm
            serial.save(self.save_path, model)


class FindBasis2(TrainExtension):

    def __init__(self, soln_path, save_path, black_sheep_path):
        self.__dict__.update(locals())
        del self.self

        soln = serial.load(soln_path)
        self.soln = soln.get_param_vector()

        black_sheep = serial.load(black_sheep_path)
        self.black_sheep = black_sheep.get_param_vector()

    def setup(self, model, dataset, algorithm):
        self.origin = model.get_param_vector()
        direction = self.soln - self.origin
        norm = np.sqrt(np.square(direction).sum())
        self.unit = direction / norm
        sheep_coord = np.dot(self.black_sheep - self.origin, self.unit)
        sheep_proj = self.unit * sheep_coord
        direction = self.black_sheep - self.origin - sheep_proj
        norm = np.sqrt(np.square(direction).sum())
        self.unit2 = direction / norm
        model.monitor.add_channel(
                    name="soln_coord",
                    ipt=None,
                    val=0.,
                    data_specs=(NullSpace(), ''),
                    dataset=model.monitor._datasets[0])
        model.monitor.add_channel(
                    name="soln_dist",
                    ipt=None,
                    val=0.,
                    data_specs=(NullSpace(), ''),
                    dataset=model.monitor._datasets[0])
        model.monitor.add_channel(
                    name="sheep_coord",
                    ipt=None,
                    val=0.,
                    data_specs=(NullSpace(), ''),
                    dataset=model.monitor._datasets[0])
        model.monitor.add_channel(
                    name="residual_norm",
                    ipt=None,
                    val=0.,
                    data_specs=(NullSpace(), ''),
                    dataset=model.monitor._datasets[0])
        self.biggest_residual = -1.

    def on_monitor(self, model, dataset, algorithm):
        cur = model.get_param_vector()
        dist = np.sqrt(np.square(cur - self.soln).sum())
        cur -= self.origin
        coord = np.dot(cur, self.unit)
        proj = self.unit * coord
        residual = cur - proj
        sheep_coord = np.dot(residual, self.unit2)
        proj = self.unit2 * sheep_coord
        residual -= proj
        residual_norm = np.sqrt(np.square(residual).sum())
        assert model.monitor.channels['soln_coord'].val_record[-1] == 0.
        model.monitor.channels['soln_coord'].val_record[-1] = coord
        assert model.monitor.channels['soln_dist'].val_record[-1] == 0.
        model.monitor.channels['soln_dist'].val_record[-1] = dist
        assert model.monitor.channels['sheep_coord'].val_record[-1] == 0.
        model.monitor.channels['sheep_coord'].val_record[-1] = sheep_coord
        assert model.monitor.channels['residual_norm'].val_record[-1] == 0.
        model.monitor.channels['residual_norm'].val_record[-1] = residual_norm
        if residual_norm > self.biggest_residual:
            self.biggest_residual = residual_norm
            serial.save(self.save_path, model)

class Booster(TrainExtension):
    def __init__(self, scales):
        self.__dict__.update(locals())
        del self.self
        self.batch_size = 2000

    def setup(self, model, dataset, algorithm):
        self.origin = model.get_param_vector()

        cost = algorithm.cost
        # Cargo cult all the Pascal bullshit needed to evaluate the fucking cost function now
        # =======================================
        data_specs = cost.get_data_specs(model)
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)

        # Build a flat tuple of Theano Variables, one for each space.
        # We want that so that if the same space/source is specified
        # more than once in data_specs, only one Theano Variable
        # is generated for it, and the corresponding value is passed
        # only once to the compiled Theano function.
        theano_args = []
        for space, source in safe_zip(space_tuple, source_tuple):
            name = '%s[%s]' % (self.__class__.__name__, source)
            arg = space.make_theano_batch(name=name,
                                          batch_size=self.batch_size)
            theano_args.append(arg)
        theano_args = tuple(theano_args)

        # Methods of `cost` need args to be passed in a format compatible
        # with data_specs
        nested_args = mapping.nest(theano_args)
        fixed_var_descr = cost.get_fixed_var_descr(model, nested_args)
        self.on_load_batch = fixed_var_descr.on_load_batch

        cost_value = cost.expr(model, nested_args,
                                    ** fixed_var_descr.fixed_vars)
        # End cargo culting
        # ======================

        print "Compiling cost function..."
        cost_fn = function(theano_args, cost_value)
        self.cost_fn = cost_fn

    def on_monitor(self, model, dataset, algorithm):
        d = model.get_param_vector() - self.origin

        data = list(dataset.get_batch_design(self.batch_size,
            include_labels=True))
        from pylearn2.utils.one_hot import one_hot
        data[1] = one_hot(data[1])

        cost_values = []
        for scale in self.scales:
            print "Evaluating cost at scale ", scale

            model.set_param_vector(self.origin + scale * d)

            cost_values.append(self.cost_fn(*data))

        print 'Scales searched: ',self.scales
        print 'Cost values: ', cost_values

        best_scale = self.scales[cost_values.index(min(cost_values))]
        print "best_scale: ", best_scale
        model.set_param_vector(self.origin + best_scale * d)
