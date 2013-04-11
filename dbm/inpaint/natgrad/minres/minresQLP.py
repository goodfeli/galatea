"""
Evaluates min-length solution to symmetric (possibly singular) Ax=b or
min||Ax-b|| using the minresQLP.

See also BICG, BICGSTAB, BICGSTABL, CGS, GMRES, LSQR, PCG, QMR, SYMMLQ,
TFQMR, CHOLINC, FUNCTION_HANDLE.
Also MINRES, SYMMLQ, LSQR, CGLS downloadable from
  http://www.stanford.edu/group/SOL/software.html

REFERENCES:
S.-C. Choi, C. C. Paige, and M. A. Saunders,
MINRES-QLP: A Krylov subspace method for indefinite or singular symmetric
systems, SIAM Journal of Scientific Computing, submitted on March 7, 2010.

S.-C. Choi's PhD Dissertation, Stanford University, 2006:
  http://www.stanford.edu/group/SOL/dissertations.html

The current code is a translation of the matlab code from
http://www.stanford.edu/group/SOL/software.html into python using numpy and
Theano.

Contact: Razvan Pascanu (r.pascanu@gmai.com)

License: 3-clause BSD
"""
import numpy
import theano
import theano.tensor as TT
import theano.sandbox.cuda


messages = [' beta2 = 0.  b and x are eigenvectors                   ',  # -1
            ' beta1 = 0.  The exact solution is  x = 0               ',  # 0
            ' A solution to Ax = b found, given rtol                 ',  # 1
            ' Min-length solution for singular LS problem, given rtol',  # 2
            ' A solution to Ax = b found, given eps                  ',  # 3
            ' Min-length solution for singular LS problem, given eps ',  # 4
            ' x has converged to an eigenvector                      ',  # 5
            ' xnorm has exceeded maxxnorm                            ',  # 6
            ' Acond has exceeded Acondlim                            ',  # 7
            ' The iteration limit was reached                        ',  # 8
            ' Least-squares problem but no converged solution yet    ']  # 9


def make_array(ndarray):
    if theano.sandbox.cuda.cuda_available:
        return theano.sandbox.cuda.CudaNdarray(ndarray)
    else:
        return ndarray


def tocuda(var):
    if theano.sandbox.cuda.cuda_available:
        return theano.sandbox.cuda.CudaNdarray(
            numpy.array(var, dtype='float32'))
    else:
        return var


class MinresQLP(theano.gof.Op):
    def __init__(self,
                 compute_Gv,
                 param_shapes,
                 Ms=None,
                 rtol=1e-6,
                 maxit=100,
                 damp=0,
                 maxxnorm=1e7,
                 Acondlim=1e15,
                 TranCond=1e7,
                 mode=None,
                 gpu=0,
                 profile=0):
        """
        Theano Op for evaluating minres QLP algorithm.
        """
        self.gpu = gpu
        floatX = theano.config.floatX
        self.r1s = [theano.shared(numpy.zeros(shp, dtype=floatX),
                                  name='r1_%d' % k)
                    for k, shp in enumerate(param_shapes)]
        self.r2s = [theano.shared(numpy.zeros(shp, dtype=floatX),
                                  name='r2_%d' % k)
                    for k, shp in enumerate(param_shapes)]
        self.r3s = [theano.shared(numpy.zeros(shp, dtype=floatX),
                                  name='r3_%d' % k)
                    for k, shp in enumerate(param_shapes)]
        self.vs = [theano.shared(numpy.zeros(shp, dtype=floatX),
                                 name='v_%d' % k)
                   for k, shp in enumerate(param_shapes)]
        self.xs = [theano.shared(numpy.zeros(shp, dtype=floatX),
                                  name='x_%d' % k)
                    for k, shp in enumerate(param_shapes)]
        self.xl2s = [theano.shared(numpy.zeros(shp, dtype=floatX),
                                  name='xl2_%d' % k)
                    for k, shp in enumerate(param_shapes)]
        self.ws = [theano.shared(numpy.zeros(shp, dtype=floatX),
                                  name='w_%d' % k)
                    for k, shp in enumerate(param_shapes)]
        self.wls = [theano.shared(numpy.zeros(shp, dtype=floatX),
                                  name='wl_%d' % k)
                    for k, shp in enumerate(param_shapes)]
        self.wl2s = [theano.shared(numpy.zeros(shp, dtype=floatX),
                                  name='wl2_%d' % k)
                    for k, shp in enumerate(param_shapes)]
        ## Params
        self.damp = damp
        self.maxxnorm = maxxnorm
        self.Acondlim = Acondlim
        self.TranCond = TranCond
        self.maxit = maxit
        self.rtol = rtol
        self.Ms = Ms
        if theano.sandbox.cuda.cuda_available and \
           'gpu' in theano.config.device :
            self.zero = theano.sandbox.cuda.CudaNdarray(
                numpy.array(0, dtype='float32'))
        else:
            self.zero = 0
        ## Variables
        self.beta = TT.scalar('beta')
        self.betal = TT.scalar('betal')
        self.alfa = TT.scalar('alfa')
        self.eplnn = TT.scalar('eplnn')
        self.dlta_QLP = TT.scalar('dlta_QLP')
        self.gama_tmp = TT.scalar('gama_tmp')
        self.tau = TT.scalar('tau')
        self.gamal3 = TT.scalar('gamal3')
        self.veplnl2 = TT.scalar('veplnl2')
        self.etal = TT.scalar('etal')
        self.gamal_QLP = TT.scalar('gamal_QLP')
        self.vepln_QLP = TT.scalar('vepln_QLP')
        self.gama_QLP = TT.scalar('gama_QLP')
        self.ul_QLP = TT.scalar('ul_QLP')
        self.u_QLP = TT.scalar('u_QLP')
        self.cr1 = TT.scalar('cr1')
        self.sr1 = TT.scalar('sr1')
        self.cr2 = TT.scalar('cr2')
        self.sr2 = TT.scalar('sr2')
        self.ul2 = TT.scalar('ul2')
        self.ul = TT.scalar('ul')
        self.u = TT.scalar('u')
        rvals = compute_Gv(*self.vs)
        self.Gvs = rvals[0]
        if not isinstance(rvals[1], list):
            self.updates = rvals[1].items()
        else:
            self.updates = rvals[1]
        self.mode = mode
        self.profile = profile

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(self) == type(other)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, *args):
        return theano.gof.Apply(self, args, [x.type() for x in args]
                               + [TT.scalar(), TT.scalar(), TT.scalar(),
                                TT.scalar(), TT.scalar()])

    def sym_givens2(self, a, b):
        if b == 0:
            if a == 0:
                c = 1
            else:
                c = numpy.sign(a)
            s = 0
            d = abs(a)
        elif a == 0:
            c = 0
            s = numpy.sign(b)
            d = abs(b)
        elif abs(b) > abs(a):
            t = a / b
            s = numpy.sign(b) / numpy.sqrt(1 + t ** 2)
            c = s * t
            d = b / s  # computationally better than d = a / c since |c| <= |s|
        else:
            t = b / a
            c = numpy.sign(a) / numpy.sqrt(1 + t ** 2)
            s = c * t
            d = a / c  # computationally better than d = b / s since |s| <= |c|
        return c, s, d

    def compile_all(self):
        if self.Ms:
            r3s = [r2 / m for r2, m in zip(self.r2s, self.Ms)]
        else:
            r3s = [r2 for r2 in self.r2s]
        beta1 = TT.sqrt(sum((r2 * r3).sum() for r2, r3 in zip(self.r2s, r3s)))
        self.compute_beta1 = theano.function(
            [],
            beta1,
            updates=[(or3, nr3) for or3, nr3 in zip(self.r3s, r3s)],
            mode=self.mode,
            profile=self.profile,
            allow_input_downcast=True,
            name='compute_beta1')

        self.update_vs = theano.function(
            [self.beta],
            [],
            updates=[(v, r3 / self.beta) for
                     v, r3 in zip(self.vs, self.r3s)],
            mode=self.mode,
            profile=self.profile,
            allow_input_downcast=True,
            name='update_vs')

        r3s = [gv + self.damp * v for gv, v in zip(self.Gvs, self.vs)]
        alfa = sum(TT.sum(r3 * v) for r3, v in zip(r3s, self.vs))
        r3s = [r3 - (alfa / self.beta) * r2 for r3, r2 in zip(r3s, self.r2s)]

        updates1 = [(r1, r2) for r1, r2 in zip(self.r1s, self.r2s)]
        updates2 = [(r2, r3) for r2, r3 in zip(self.r2s, r3s)]

        if not self.Ms:
            betan = TT.sqrt(sum((r3 ** 2).sum() for r3 in r3s))
            updates3 = [(or3, nr3) for or3, nr3 in zip(self.r3s, r3s)]
        else:
            nr3s = [r3 / m for r3, m in zip(r3s, self.Ms)]
            betan = TT.sqrt(sum((r3 * nr3).sum()
                                for nr3, r3 in zip(r3s, nr3s)))
            updates3 = [(or3, nr3) for or3, nr3 in zip(self.r3s, nr3s)]

        self.step0 = theano.function(
            [self.beta],
            [alfa, betan],
            updates=self.updates + updates1 + updates2 + updates3,
            mode=self.mode,
            profile=self.profile,
            allow_input_downcast=True,
            name='step0')

        self.eval_x = theano.function(
            [self.alfa],
            [],
            updates=[(x, x / self.alfa) for x in self.xs],
            mode=self.mode,
            profile=self.profile,
            allow_input_downcast=True,
            name='eval_x')

        r3s = [gv + self.damp * v for gv, v in zip(self.Gvs, self.vs)]
        r3s = [r3 - (self.beta / self.betal) * r1
               for r3, r1 in zip(r3s, self.r1s)]
        alfa = sum(TT.sum(r3 * v) for r3, v in zip(r3s, self.vs))
        r3s = [r3 - (alfa / self.beta) * r2 for r3, r2 in zip(r3s, self.r2s)]
        updates1 = [(r1, r2) for r1, r2 in zip(self.r1s, self.r2s)]
        updates2 = [(r2, r3) for r2, r3 in zip(self.r2s, r3s)]

        if not self.Ms:
            betan = TT.sqrt(sum((r3 ** 2).sum() for r3 in r3s))
            updates3 = [(or3, nr3) for or3, nr3 in zip(self.r3s, r3s)]
        else:
            nr3s = [r3 / m for r3, m in zip(r3s, self.Ms)]
            betan = TT.sqrt(sum((r3 * nr3).sum()
                                for nr3, r3 in zip(r3s, nr3s)))
            updates3 = [(or3, nr3) for or3, nr3 in zip(self.r3s, nr3s)]
        self.step1 = theano.function(
            [self.beta, self.betal],
            [alfa, betan],
            updates=self.updates + updates1 + updates2 + updates3,
            mode=self.mode,
            profile=self.profile,
            allow_input_downcast=True,
            name='step1')
        updates1 = [(wl2, wl) for wl2, wl in zip(self.wl2s, self.wls)]
        updates2 = [(wl, w) for wl, w in zip(self.wls, self.ws)]
        updates3 = [(w, (v - self.eplnn * wl -
                         self.dlta_QLP * w) / self.gama_tmp)
                    for w, v, wl in zip(self.ws, self.vs, self.wls)]

        self.update_ws = theano.function(
            [self.eplnn, self.dlta_QLP, self.gama_tmp],
            [],
            updates=updates1 + updates2 + updates3,
            mode=self.mode,
            profile=self.profile,
            name='update_ws')

        updates = [(x, x + self.tau * w) for x, w in zip(self.xs, self.ws)]
        self.update_xs_case0 = theano.function(
            [self.tau],
            [],
            updates=updates,
            mode=self.mode,
            profile=self.profile,
            allow_input_downcast=True,
            name='update_xs_case0')

        updates = [(wl2, self.gamal3 * wl2 + self.veplnl2 * wl + self.etal * w)
                   for wl2, wl, w in zip(self.wl2s, self.wls, self.ws)]
        self.update_QLP1_iter_g3 = theano.function(
            [self.gamal3, self.veplnl2, self.etal],
            [],
            updates=updates,
            mode=self.mode,
            profile=self.profile,
            allow_input_downcast=True,
            name='update_QLP1_iter_g3')

        updates = [(wl, self.gamal_QLP * wl + self.vepln_QLP * w)
                    for wl, w in zip(self.wls, self.ws)]
        self.update_QLP1_iter_g2 = theano.function(
            [self.gamal_QLP, self.vepln_QLP],
            [],
            updates=updates,
            mode=self.mode,
            profile=self.profile,
            allow_input_downcast=True,
            name='update_QLP1_iter_g2')

        updates1 = [(w, self.gama_QLP * w) for w in self.ws]
        updates2 = [(xl2, x - wl * self.ul_QLP - w * self.u_QLP)
                    for xl2, x, wl, w in zip(self.xl2s,
                                             self.xs,
                                             self.wls,
                                             self.ws)]

        self.update_QLP1 = theano.function(
            [self.gama_QLP, self.ul_QLP, self.u_QLP],
            [],
            updates=updates1 + updates2,
            mode=self.mode,
            profile=self.profile,
            allow_input_downcast=True,
            name='update_QLP1')

        nw_wl2 = [wl for wl in self.wls]
        nw_wl = [v * self.sr1 for v in self.vs]
        nw_w = [-v * self.cr1 for v in self.vs]
        nw_xl2 = [xl2 + wl2 * self.ul2 for xl2, wl2 in zip(self.xl2s, nw_wl2)]
        nw_xs = [xl2 + wl * self.ul + w * self.u
                  for xl2, wl, w in zip(nw_xl2, nw_wl, nw_w)]
        updates = zip(self.wl2s + self.wls + self.ws + self.xl2s + self.xs,
                      nw_wl2 + nw_wl + nw_w + nw_xl2 + nw_xs)
        self.update_iter1 = theano.function(
            [self.sr1, self.cr1, self.ul2, self.ul, self.u],
            [],
            updates=updates,
            mode=self.mode,
            profile=self.profile,
            allow_input_downcast=True,
            name='update_iter1')

        nw_wl2 = [wl for wl in self.wls]
        nw_wl = [w * self.cr1 + v * self.sr1 for w, v in zip(self.ws, self.vs)]
        nw_w = [w * self.sr1 - v * self.cr1 for w, v in zip(self.ws, self.vs)]
        nw_xl2 = [xl2 + wl2 * self.ul2 for xl2, wl2 in zip(self.xl2s, nw_wl2)]
        nw_xs = [xl2 + wl * self.ul + w * self.u
                 for xl2, wl, w in zip(nw_xl2, nw_wl, nw_w)]
        updates = zip(self.wl2s + self.wls + self.ws + self.xl2s + self.xs,
                      nw_wl2 + nw_wl + nw_w + nw_xl2 + nw_xs)
        self.update_iter2 = theano.function(
            [self.sr1, self.cr1, self.ul2, self.ul, self.u],
            [],
            updates=updates,
            mode=self.mode,
            profile=self.profile,
            allow_input_downcast=True,
            name='update_iter2')

        nw_wl2 = [wl for wl in self.wls]
        nw_wl = [w for w in self.ws]
        nw_w = [wl2 * self.sr2 - v * self.cr2
                for wl2, v in zip(nw_wl2, self.vs)]
        nw_wl2 = [wl2 * self.cr2 + v * self.sr2
                  for wl2, v in zip(nw_wl2, self.vs)]
        nw_vs = [wl * self.cr1 + w * self.sr1 for wl, w in zip(nw_wl, nw_w)]
        nw_w = [wl * self.sr1 - w * self.cr1 for wl, w in zip(nw_wl, nw_w)]
        nw_wl = [v for v in nw_vs]
        nw_xl2 = [xl2 + wl2 * self.ul2 for xl2, wl2 in zip(self.xl2s, nw_wl2)]
        nw_xs = [xl2 + wl * self.ul + w * self.u
                  for xl2, wl, w in zip(nw_xl2, nw_wl, nw_w)]
        updates = zip(self.wl2s + self.wls + self.ws + self.xl2s + self.xs,
                      nw_wl2 + nw_wl + nw_w + nw_xl2 + nw_xs)
        inps = [self.sr1,
                self.cr1,
                self.sr2,
                self.cr2,
                self.ul2,
                self.ul,
                self.u]

        self.update_iter_g2 = theano.function(
            inps,
            [],
            updates=updates,
            mode=self.mode,
            profile=self.profile,
            allow_input_downcast=True,
            name='update_iter2')

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]
        node_input_compute = [compute_map[r] for r in node.inputs]
        node_output_compute = [compute_map[r] for r in node.outputs]
        # Compile inner functions
        self.compile_all()
        p = self.execute
        # default arguments are stored in the closure of `rval`

        def rval(p=p, i=node_input_storage, o=node_output_storage, n=node):
            r = p(n, [x[0] for x in i], o)
            for o in node.outputs:
                compute_map[o][0] = True
            return r
        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        rval.perform = p
        rval.lazy = False
        return rval

    def execute(self, node, inputs, outputs):
        realmin = 1e-38
        eps = 1e-16
        for r2, inp in zip(self.r2s, inputs):
            r2.set_value(inp, borrow=True)
        beta1 = self.compute_beta1()

        ## Initialize other quantities
        flag0 = -2
        flag = -2
        iters = 0
        QLPiter = 0
        beta = 0
        tau = 0
        taul = 0
        phi = beta1
        betan = beta1
        gmin = 0
        cs = -1
        sn = 0
        cr1 = -1
        sr1 = 0
        cr2 = -1
        sr2 = 0
        dltan = 0
        eplnn = 0
        gama = 0
        gamal = 0
        gamal2 = 0
        eta = 0
        etal = 0
        etal2 = 0
        vepln = 0
        veplnl = 0
        veplnl2 = 0
        ul3 = 0
        ul2 = 0
        ul = 0
        u = 0
        rnorm = betan
        xnorm = 0
        xl2norm = 0
        Axnorm = 0
        Anorm = 0
        Acond = 1
        relres = rnorm / (beta1 + 1e-50)
        for x, w, wl in zip(self.xs, self.ws, self.wls):
            x.container.storage[0][:] = self.zero
            w.container.storage[0][:] = self.zero
            wl.container.storage[0][:] = self.zero

        if beta1 == 0:
            flag = 0
        while flag == flag0 and iters < self.maxit:
            iters = iters + 1
            betal = beta
            beta = betan
            self.update_vs(beta)
            if iters > 1:
                alfa, betan = self.step1(beta, betal)
            else:
                alfa, betan = self.step0(beta)
            if not self.Ms:
                if iters == 1:
                    if betan == 0:
                        if alfa == 0:
                            flag = 0
                            break
                        else:
                            flag = -1
                            self.eval_x(alfa)
                            break
            pnorm = numpy.sqrt(betal ** 2 + alfa ** 2 + betan ** 2)
            dbar = dltan
            dlta = cs * dbar + sn * alfa
            epln = eplnn
            gbar = sn * dbar - cs * alfa
            eplnn = sn * betan
            dltan = -cs * betan
            dlta_QLP = dlta
            gamal3 = gamal2
            gamal2 = gamal
            gamal = gama
            cs, sn, gama = self.sym_givens2(gbar, betan)
            gama_tmp = gama
            taul2 = taul
            taul = tau
            tau = cs * phi
            Axnorm = numpy.sqrt(Axnorm ** 2 + tau ** 2)
            phi = sn * phi

            if iters > 2:
                veplnl2 = veplnl
                etal2 = etal
                etal = eta
                dlta_tmp = sr2 * vepln - cr2 * dlta
                veplnl = cr2 * vepln + sr2 * dlta
                dlta = dlta_tmp
                eta = sr2 * gama
                gama = -cr2 * gama

            if iters > 1:
                cr1, sr1, gamal = self.sym_givens2(gamal, dlta)
                vepln = sr1 * gama
                gama = -cr1 * gama

            xnorml = xnorm
            ul4 = ul3
            ul3 = ul2
            if iters > 2:
                ul2 = (taul2 - etal2 * ul4 - veplnl2 * ul3) / gamal2
            if iters > 1:
                ul = (taul - etal * ul3 - veplnl * ul2) / gamal

            xnorm_tmp = numpy.sqrt(xl2norm ** 2 + ul2 ** 2 + ul ** 2)
            if abs(gama) > realmin and xnorm_tmp < self.maxxnorm:
                u = (tau - eta * ul2 - vepln * ul) / gama
                if numpy.sqrt(xnorm_tmp ** 2 + u ** 2) > self.maxxnorm:
                    u = 0
                    flag = 6
            else:
                u = 0
                flag = 9
            xl2norm = numpy.sqrt(xl2norm ** 2 + ul2 ** 2)
            xnorm = numpy.sqrt(xl2norm ** 2 + ul ** 2 + u ** 2)

            if (Acond < self.TranCond) and flag != flag0 and QLPiter == 0:
                self.update_ws(epln, dlta_QLP, gama_tmp)
                if xnorm < self.maxxnorm:
                    self.update_xs_case0(tau)
                else:
                    flag = 6
            else:
                QLPiter = QLPiter + 1
                if QLPiter == 1:
                    for xl2 in self.xl2s:
                        xl2.container.storage[0][:] = self.zero
                    if  iters > 1:  # construct w_{k-3}, w_{k-2}, w_{k-1}
                        if iters > 3:
                            self.udpate_QLP1_iter_g3(gamal, veplnl2, etal)
                        if iters > 2:
                            self.update_QLP1_iter_g2(gamal_QLP, vepln_QLP)
                        self.update_QLP1(gama_QLP, ul_QLP, u_QLP)

                if iters == 1:
                    self.update_iter1(sr1, cr1, ul2, ul, u)
                elif iters == 2:
                    self.update_iter2(sr1, cr1, ul2, ul, u)
                else:
                    self.update_iter_g2(sr1, cr1, sr2, cr2, ul2, ul, u)

            ## Compute the next right plane rotation P{k-1,k+1}
            gamal_tmp = gamal
            cr2, sr2, gamal = self.sym_givens2(gamal, eplnn)

            gamal_QLP = gamal_tmp
            vepln_QLP = vepln
            gama_QLP = gama
            ul_QLP = ul
            u_QLP = u

            ## Estimate various norms
            abs_gama = abs(gama)
            Anorml = Anorm
            Anorm = numpy.max([Anorm, pnorm, gamal, abs_gama])
            if iters == 1:
                gmin = gama
                gminl = gmin
            elif iters > 1:
                gminl2 = gminl
                gminl = gmin
                gmin = numpy.min([gminl2, gamal, abs_gama])
            Acondl = Acond
            Acond = Anorm / gmin
            rnorml = rnorm
            relresl = relres
            if flag != 9:
                rnorm = phi
            relres = rnorm / (Anorm * xnorm + beta1)
            rootl = numpy.sqrt(gbar ** 2 + dltan ** 2)
            Arnorml = rnorml * rootl
            relAresl = rootl / Anorm

            ## See if any of the stopping criteria are satisfied.
            epsx = Anorm * xnorm * eps
            if (flag == flag0) or (flag == 9):
                t1 = 1 + relres
                t2 = 1 + relAresl
                if iters >= self.maxit:
                    flag = 8
                if Acond >= self.Acondlim:
                    flag = 7
                if xnorm >= self.maxxnorm:
                    flag = 6
                if epsx >= beta1:
                    flag = 5
                if t2 <= 1:
                    flag = 4
                if t1 <= 1:
                    flag = 3
                if relAresl <= self.rtol:
                    flag = 2
                if relres <= self.rtol:
                    flag = 1

            if flag == 2 or flag == 4 or flag == 6 or flag == 7:
                iters = iters - 1
                Acond = Acondl
                rnorm = rnorml
                relres = relresl
        for buf, x in zip(outputs, self.xs):
            if self.gpu:
                buf[0] = x.container.storage[0]
            else:
                buf[0] = numpy.array(x.container.storage[0])
        npos = len(self.xs)
        outputs[npos + 0][0] = numpy.array(flag, dtype=theano.config.floatX)
        outputs[npos + 1][0] = numpy.array(iters, dtype=theano.config.floatX)
        outputs[npos + 2][0] = numpy.array(relres, dtype=theano.config.floatX)
        outputs[npos + 3][0] = numpy.array(Anorm, dtype=theano.config.floatX)
        outputs[npos + 4][0] = numpy.array(Acond, dtype=theano.config.floatX)


def minresQLP(compute_Av,
              bs,
              param_shapes,
              rtol=numpy.float32(1e-6),
              maxit=20,
              Ms=None,
              damp=numpy.float32(0),
              maxxnorm=numpy.float32(1e15),
              Acondlim=numpy.float32(1e16),
              TranCond=numpy.float32(1e7),
              mode=None,
              profile=0):
    """
     minresQLP: min-length solution to symmetric (possibly singular) Ax=b
     or min||Ax-b||.

     DESCRIPTION:
         minres attempts to find the solution x to the system of linear
         equations A*x = b or least squares problem min||Ax-b||.
         The n-by-n coefficient matrix A must be symmetric (but need not
         be positive definite or invertible).
         The right-hand-side column vector b must have length n.

         In practice, one is required only to provide means for computing
         A*x for some random x vector. Also x and b can be provided as a
         list of tensors, where the meaning is that x or b can be
         constructed by flattening and concatenating all tensors in the
         given lists. Note that if one choses this options both the output
         of the function `compute_Av` and `b` has to be provided in this
         form.

     INPUTS:
        :param compute_Av: callable returning the symbolic expression for
            `Av`. `v` can be represented as a list, where `v` becomes the
            concatenation of all the tensors in the list after flattening
            each one
        :param bs: list of Theano variables or variable. We are looking to
            compute `A^-1 \dot b`, where `b` is the concatenation of all
            tensors in `bs` after flattening if `bs` is a list.
        :param param_shapes: list or int depicting the shape of `bs` (or of
            each tensor in `bs`)
        :param rtol: Optional, real, specifies the tolerance of the method.
            Default is 1e-6
        :param maxit: Optional, positive integer, specifies the maximum number
            of iterations. Default is 20
        :param Ms: List of theano expression of same shape as `bs`. The
            method uses these to precondition with diag(Ms)
        :param damp: Optional, variable or number.  Default is 0.
            Effectively solve the system (A + damp I) * x = b.
        :param maxxnorm:
            real positive, maximum bound on NORM(x). Default is 1e14.
        :param Acondlim:
            real positive, maximum bound on COND(A). Default is 1e15.
        :param TranCond:
            real scalar >= 1.
            If TranCond>1,      a switch is made from MINRES iterations to
                                MINRES-QLP iterationsd when ACOND >= TRANCOND.
            If TranCond=1,      all iterations will be MINRES-QLP iterations.
            If TranCond=Acondlim, all iterations will be conventional MINRES
                                iterations (which are slightly cheaper).
            Default TranCond=1e7.


     OUTPUTS:
        x       tensor or list of tensors (if `bs` was provided as a list)
        flag    integer, convergence flag
               -1  beta2 = 0.  If M = I, b and x are eigenvectors.
                0 beta1 = 0.  The exact solution is  x = 0.
                1 A solution to (poss. singular) Ax = b found, given rtol.
                2 Pseudoinverse solution for singular LS problem, given rtol.
                3 A solution to (poss. singular) Ax = b found, given eps.
                4 Pseudoinverse solution for singular LS problem, given eps.
                5 x has converged to an eigenvector.
                6 xnorm has exceeded maxxnorm.
                7 Acond has exceeded Acondlim.
                8 The iteration limit was reached.
                9 It is a least squares problem but no converged solution yet.
        iter    integer, iteration number at which x was computed:
            0 <= iter <= maxit.
        relres  real positive, the relative residual is defined as
                     NORM(b-A*x)/(NORM(A) * NORM(x) + NORM(b)),
                computed recurrently here.  If flag is 1 or 3,  relres <= TOL.
        Anorm   real positive, estimate of matrix 2-norm of A.
        Acond   real positive, estimate of condition number of A with
                respect to 2-norm.


     REFERENCES:
        Sou-Cheng Choi's PhD Dissertation, Stanford University, 2006.
             http://www.stanford.edu/group/SOL/software.html


    """
    if not isinstance(bs, (tuple, list)):
        bs = [bs]
        param_shapes = [param_shapes]
        if Ms:
            Ms = [Ms]
        return_as_list = False
    else:
        bs = list(bs)
        return_as_list = True

    minres_qlp = MinresQLP(
        compute_Av,
        param_shapes=param_shapes,
        Ms=Ms,
        rtol=rtol,
        maxit=maxit,
        damp=damp,
        maxxnorm=maxxnorm,
        Acondlim=Acondlim,
        TranCond=TranCond,
        mode=mode,
        profile=profile)
    rvals = minres_qlp(*bs)
    sol = rvals[:len(bs)]
    if not return_as_list:
        sol = sol[0]
    return [sol] + rvals[len(bs):]


def test_1():
    n = 100
    on = numpy.ones((n, 1), dtype='float32')
    A = numpy.zeros((n, n), dtype='float32')
    for k in xrange(n):
        A[k, k] = 4.
        if k > 0:
            A[k - 1, k] = -2.
            A[k, k - 1] = -2.
    b = A.sum(axis=1)
    rtol = numpy.float32(1e-10)
    maxit = 50
    M = numpy.ones((n,), dtype='float32') * 4.
    tA = theano.shared(A.astype('float32'))
    tb = theano.shared(b.astype('float32'))
    tM = theano.shared(M.astype('float32'))

    compute_Gv = lambda x: ([TT.dot(tA, x)], {})

    sol, flag, iters, relres, Anorm, Acond = minresQLP(
        compute_Gv,
        tb,
        param_shapes=(n,),
        Ms=tM,
        rtol=rtol,
        maxit=maxit)

    mqlp = theano.function([], [sol, flag, iters, relres, Anorm, Acond])
    sol, flag, iters, relres, Anorm, Acond = mqlp()
    sol = numpy.array(sol)
    print 'flag', flag
    print messages[int(flag + 1)]
    print 'iters', iters
    print 'relres', relres
    print 'Anorm', Anorm
    print 'Acond', Acond
    print 'error', numpy.sqrt(numpy.sum((numpy.dot(sol, A) - b) ** 2))
    print 'Solution', sol
    print


def test_2():
    h = 1
    a = -10
    b = -a
    n = 2 * b // h + 1
    A = numpy.zeros((n, n), dtype='float32')
    A = numpy.zeros((n, n), dtype='float32')
    v = a
    for k in xrange(n):
        A[k, k] = v
        v += h
    b = numpy.ones((n,), dtype='float32')
    rtol = numpy.float32(1e-6)
    maxxnorm = 1e8
    maxit = 500
    tA = theano.shared(A.astype('float32'))
    tb = theano.shared(b.astype('float32'))
    compute_Gv = lambda x: ([TT.dot(tA, x)], {})

    sol, flag, iters, relres, Anorm, Acond = minresQLP(
        compute_Gv,
        tb,
        param_shapes=(n,),
        rtol=rtol,
        maxit=maxit)

    mqlp = theano.function([], [sol, flag, iters, relres, Anorm, Acond])
    sol, flag, iters, relres, Anorm, Acond = mqlp()
    sol = numpy.array(sol)
    print 'flag', flag
    print messages[int(flag + 1)]
    print 'iters', iters
    print 'relres', relres
    print 'Anorm', Anorm
    print 'Acond', Acond
    print 'error', numpy.sqrt(numpy.sum((numpy.dot(sol, A) - b) ** 2))
    print 'Solution', sol
    print


def test_3():
    n = 10
    rtol = numpy.float32(1e-10)
    rng = numpy.random.RandomState(23)
    A = rng.uniform(size=(n, n))
    A[:, n - 3:] = 0.
    A[n - 3:] = 0.
    b = rng.uniform(size=(n,))

    maxit = 5
    tA = theano.shared(A.astype('float32'))
    tb = theano.shared(b.astype('float32'))
    compute_Gv = lambda x: ([TT.dot(tA, x)], {})

    sol, flag, iters, relres, Anorm, Acond = minresQLP(
        compute_Gv,
        tb,
        param_shapes=(n,),
        rtol=rtol,
        maxit=maxit)

    mqlp = theano.function([], [sol, flag, iters, relres, Anorm, Acond])
    sol, flag, iters, relres, Anorm, Acond = mqlp()
    sol = numpy.array(sol)
    print 'flag', flag
    print messages[int(flag + 1)]
    print 'iters', iters
    print 'relres', relres
    print 'Anorm', Anorm
    print 'Acond', Acond
    print 'error', numpy.sqrt(numpy.sum((numpy.dot(sol, A) - b) ** 2))
    print 'Solution', sol
    print

if theano.sandbox.cuda.cuda_available:
    from theano.gof import local_optimizer
    from theano.sandbox.cuda.opt import register_opt
    from theano.sandbox.cuda.basic_ops import gpu_from_host, host_from_gpu
    from theano.sandbox.cuda.type import CudaNdarrayType

    @register_opt()
    @local_optimizer([])
    def local_gpu_minres(node):
        if isinstance(node.op, MinresQLP):
            sw = False
            for inp in node.inputs:
                if inp.owner and inp.owner.op == host_from_gpu:
                    sw = True
            if sw:
                inps = node.inputs
                nw_inps = []
                for inp in inps:
                    if not isinstance(inp.type, CudaNdarrayType):
                        nw_inps.append(gpu_from_host(inp))
                    else:
                        nw_inps.append(inp)
                new_op = node.op
                new_op.gpu = 1
                _new_outs = node.op(*nw_inps)
                new_outs = []
                for out in _new_outs:
                    if isinstance(out.type, CudaNdarrayType):
                        new_outs.append(host_from_gpu(out))
                    else:
                        new_outs.append(out)
                return new_outs
            else:
                return False


if __name__ == '__main__':
    test_1()
    test_2()
    test_3()
