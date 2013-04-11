"""
Evaluates the solution to symmetric (possibly singular) Ax=b or
min||Ax-b|| using the minres.

See also BICG, BICGSTAB, BICGSTABL, CGS, GMRES, LSQR, PCG, QMR, SYMMLQ,
TFQMR, CHOLINC, FUNCTION_HANDLE.
Also MINRESQLP, SYMMLQ, LSQR, CGLS downloadable from
  http://www.stanford.edu/group/SOL/software.html

REFERENCES:
S.-C. Choi's PhD Dissertation, Stanford University, 2006:
  http://www.stanford.edu/group/SOL/dissertations.html

The current code is a translation of the matlab code from
http://www.stanford.edu/group/SOL/software.html into python using numpy and
Theano.

Contact: Razvan Pascanu (r.pascanu@gmai.com)

License: 3-clause BSD
"""
import theano
import theano.tensor as TT
from theano.ifelse import ifelse
from theano.sandbox.scan import scan
import numpy

messages = [' beta1 = 0.  The exact solution is  x = 0.                    ',
            ' A solution to (poss. singular) Ax = b found, given rtol.     ',
            ' A least-squares solution was found, given rtol.              ',
            ' A solution to (poss. singular) Ax = b found, given eps.      ',
            ' A least-squares solution was found, given eps.               ',
            ' x has converged to an eigenvector.                           ',
            ' xnorm has exceeded maxxnorm.                                 ',
            ' Acond has exceeded Acondlim.                                 ',
            ' The iteration limit was reached.                             ',
            ' A least-squares solution for singular LS problem, given eps. ',
            ' A least-squares solution for singular LS problem, given rtol.',
            ' A null vector obtained, given rtol.                          ',
            ' Numbers are too small to continue computation                ']


def norm(xs, ys=None):
    """
    Compute the norm between xs and ys. If ys is not provided, computes the
    norm between xs and xs.
    Note : xs is a list of Tensors
    """
    if ys is None:
        ys = [x for x in xs]
    return TT.sqrt(sum((x * y).sum() for x, y in zip(xs, ys)))


def sqnorm(xs, ys=None):
    """
    Compute the square norm between xs and ys. If ys is not provided, computes
    the norm between xs and xs.
    Note : xs is a list of Tensors
    """
    if ys is None:
        ys = [x for x in xs]
    return sum((x * y).sum() for x, y in zip(xs, ys))


def symGivens2(a, b):
    """
    Stable Symmetric Givens rotation plus reflection
    %  INPUTS:
    %    a      first element of a two-vector  [a; b]
    %    b      second element of a two-vector [a; b]
    %
    %  OUTPUTS:
    %    c  cosine(theta), where theta is the implicit angle of
    %       rotation (counter-clockwise) in a plane-rotation
    %    s  sine(theta)
    %    d  two-norm of [a; b]
    %  DESCRIPTION:
    %     Stable symmetric Givens rotation that gives c and s
    %     such that
    %        [ c  s ][a] = [d],
    %        [ s -c ][b]   [0]
    %     where d = two norm of vector [a, b],
    %        c = a / sqrt(a^2 + b^2) = a / d,
    %        s = b / sqrt(a^2 + b^2) = b / d.
    %     The implementation guards against overlow in computing
    %     sqrt(a^2 + b^2).
    %
    %  SEE ALSO:
    %     (1) Algorithm 4.9, stable *unsymmetric* Givens
    %     rotations in
    %     Golub and van Loan's book Matrix Computations, 3rd
    %     edition.
    %     (2) MATLAB's function PLANEROT.
    """
    c = TT.switch(
        TT.eq(b, numpy.float32(0.)),
        TT.switch(TT.eq(a, numpy.float32(0.)),
                  TT.constant(numpy.float32(1.)),
                  TT.sgn(a)),
        TT.switch(
            TT.eq(a, numpy.float32(0.)),
            TT.constant(numpy.float32(0.)),
            TT.switch(TT.gt(abs(b), abs(a)),
                      (a / b) * TT.sgn(b) / TT.sqrt(numpy.float32(1.) +
                                                    (a / b) ** 2),
                      TT.sgn(a) / TT.sqrt(numpy.float32(1.) + (b / a) ** 2))))
    s = TT.switch(
        TT.eq(b, numpy.float32(0.)),
        TT.constant(numpy.float32(0.)),
        TT.switch(
            TT.eq(a, numpy.float32(0.)),
            TT.sgn(b),
            TT.switch(TT.gt(abs(b), abs(a)),
                      TT.sgn(b) / TT.sqrt(numpy.float32(1.) + (a / b) ** 2),
                      (b / a) * TT.sgn(a) / TT.sqrt(numpy.float32(1.) +
                                              (b / a) ** 2))))

    d = TT.switch(
        TT.eq(b, numpy.float32(0.)),
        abs(a),
        TT.switch(
            TT.eq(a, numpy.float32(0.)),
            abs(b),
            TT.switch(TT.gt(abs(b), abs(a)),
                      b / (TT.sgn(b) / TT.sqrt(numpy.float32(1.) +
                                               (a / b) ** 2)),
                      a / (TT.sgn(a) / TT.sqrt(numpy.float32(1.) +
                                               (b / a) ** 2)))))
    return c, s, d


def minres(compute_Av,
           bs,
           rtol=numpy.float32(1e-6),
           maxit=20,
           Ms=None,
           damp=numpy.float32(0.),
           maxxnorm=numpy.float32(1e15),
           Acondlim=numpy.float32(1e16),
           mode=None,
           profile=0):
    """
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
        relAres real positive, the relative-NORM(Ar) := NORM(Ar) / NORM(A) ---
                computed recurrently here. If flag is 2 or 4, relAres <= TOL.
        Anorm   real positive, estimate of matrix 2-norm of A.
        Acond   real positive, estimate of condition number of A with
                respect to 2-norm.
        xnorm   non-negative positive, recurrently computed NORM(x)
        Axnorm  non-negative positive, recurrently computed NORM(A * x).


     REFERENCES:
        Sou-Cheng Choi's PhD Dissertation, Stanford University, 2006.
             http://www.stanford.edu/group/SOL/software.html

    """

    if not isinstance(bs, (tuple, list)):
        bs = [bs]
        return_as_list = False
    else:
        bs = list(bs)
        return_as_list = True

    eps = numpy.float32(1e-23)

    # Initialise
    flag = theano.shared(numpy.float32(0.))
    beta1 = norm(bs)

    #------------------------------------------------------------------
    # Set up p and v for the first Lanczos vector v1.
    # p  =  beta1 P' v1,  where  P = C**(-1).
    # v is really P' v1.
    #------------------------------------------------------------------
    r3s = [b for b in bs]
    r2s = [b for b in bs]
    r1s = [b for b in bs]
    if Ms is not None:
        r3s = [b / m for b, m in zip(bs, Ms)]
        beta1 = norm(r3s, bs)
    #------------------------------------------------------------------
    ## Initialize other quantities.
    # Note that Anorm has been initialized by IsOpSym6.
    # ------------------------------------------------------------------
    bnorm = beta1
    n_params = len(bs)

    def loop(niter,
             beta,
             betan,
             phi,
             Acond,
             cs,
             dbarn,
             eplnn,
             rnorm,
             sn,
             Tnorm,
             rnorml,
             xnorm,
             Dnorm,
             gamma,
             pnorm,
             gammal,
             Axnorm,
             relrnorm,
             relArnorml,
             Anorm,
             flag,
             *args):
        #-----------------------------------------------------------------
        ## Obtain quantities for the next Lanczos vector vk+1, k = 1, 2,...
        # The general iteration is similar to the case k = 1 with v0 = 0:
        #
        #   p1      = Operator * v1  -  beta1 * v0,
        #   alpha1  = v1'p1,
        #   q2      = p2  -  alpha1 * v1,
        #   beta2^2 = q2'q2,
        #   v2      = (1/beta2) q2.
        #
        # Again, p = betak P vk,  where  P = C**(-1).
        # .... more description needed.
        #-----------------------------------------------------------------
        xs = args[0 * n_params: 1 * n_params]
        r1s = args[1 * n_params: 2 * n_params]
        r2s = args[2 * n_params: 3 * n_params]
        r3s = args[3 * n_params: 4 * n_params]
        dls = args[4 * n_params: 5 * n_params]
        ds = args[5 * n_params: 6 * n_params]
        betal = beta
        beta = betan
        vs = [r3 / beta for r3 in r3s]
        r3s, upds = compute_Av(*vs)
        r3s = [r3 + damp * v for r3, v in zip(r3s, vs)]
        r3s = [TT.switch(TT.ge(niter, numpy.float64(1.)),
                         r3 - (beta / betal) * r1,
                         r3) for r3, r1 in zip(r3s, r1s)]

        alpha = sqnorm(r3s, vs)
        r3s = [r3 - (alpha / beta) * r2 for r3, r2 in zip(r3s, r2s)]
        r1s = [r2 for r2 in r2s]
        r2s = [r3 for r3 in r3s]
        if Ms is not None:
            r3s = [r3 / M for r3, M in zip(r3s, Ms)]
            betan = norm(r2s, r3s)
        else:
            betan = norm(r3s)
        pnorml = pnorm
        pnorm = TT.switch(TT.eq(niter, numpy.float32(0.)),
                          TT.sqrt(TT.sqr(alpha) + TT.sqr(betan)),
                          TT.sqrt(TT.sqr(alpha) + TT.sqr(betan) +
                                  TT.sqr(beta)))

        #-----------------------------------------------------------------
        ## Apply previous rotation Qk-1 to get
        #   [dlta_k epln_{k+1}] = [cs  sn][dbar_k    0      ]
        #   [gbar_k  dbar_{k+1} ]   [sn -cs][alpha_k beta_{k+1}].
        #-----------------------------------------------------------------
        dbar = dbarn
        epln = eplnn
        dlta = cs * dbar + sn * alpha
        gbar = sn * dbar - cs * alpha

        eplnn = sn * betan
        dbarn = -cs * betan

        ## Compute the current plane rotation Qk
        gammal2 = gammal
        gammal = gamma
        cs, sn, gamma = symGivens2(gbar, betan)
        tau = cs * phi
        phi = sn * phi
        Axnorm = TT.sqrt(TT.sqr(Axnorm) + TT.sqr(tau))
        # Update d

        dl2s = [dl for dl in dls]
        dls = [d for d in ds]
        ds = [TT.switch(TT.neq(gamma, numpy.float32(0.)),
                        (v - epln * dl2 - dlta * dl) / gamma,
                        v)
              for v, dl2, dl in zip(vs, dl2s, dls)]
        d_norm = TT.switch(TT.neq(gamma, numpy.float32(0.)),
                           norm(ds),
                           TT.constant((numpy.float32(numpy.inf))))

        # Update x except if it will become too big
        xnorml = xnorm
        dl2s = [x for x in xs]
        xs = [x + tau * d for x, d in zip(xs, ds)]

        xnorm = norm(xs)
        xs = [TT.switch(TT.ge(xnorm, maxxnorm),
                        dl2,
                        x) for dl2, x in zip(dl2s, xs)]

        flag = TT.switch(TT.ge(xnorm, maxxnorm),
                         numpy.float32(6.), flag)
        # Estimate various norms
        rnorml = rnorm  # ||r_{k-1}||
        Anorml = Anorm
        Acondl = Acond
        relrnorml = relrnorm
        flag_no_6 = TT.neq(flag, numpy.float32(6.))
        Dnorm = TT.switch(flag_no_6,
                          TT.sqrt(TT.sqr(Dnorm) + TT.sqr(d_norm)),
                          Dnorm)
        xnorm = TT.switch(flag_no_6, norm(xs), xnorm)
        rnorm = TT.switch(flag_no_6, phi, rnorm)
        relrnorm = TT.switch(flag_no_6,
                             rnorm / (Anorm * xnorm + bnorm),
                             relrnorm)
        Tnorm = TT.switch(flag_no_6,
                          TT.switch(TT.eq(niter, numpy.float32(0.)),
                                    TT.sqrt(TT.sqr(alpha) + TT.sqr(betan)),
                                    TT.sqrt(TT.sqr(Tnorm) +
                                            TT.sqr(beta) +
                                            TT.sqr(alpha) +
                                            TT.sqr(betan))),
                          Tnorm)
        Anorm = TT.maximum(Anorm, pnorm)
        Acond = Anorm * Dnorm
        rootl = TT.sqrt(TT.sqr(gbar) + TT.sqr(dbarn))
        Anorml = rnorml * rootl
        relArnorml = rootl / Anorm

        #---------------------------------------------------------------
        # See if any of the stopping criteria are satisfied.
        # In rare cases, flag is already -1 from above (Abar = const*I).
        #---------------------------------------------------------------
        epsx = Anorm * xnorm * eps
        epsr = Anorm * xnorm * rtol
        #Test for singular Hk (hence singular A)
        # or x is already an LS solution (so again A must be singular).
        t1 = numpy.float32(1) + relrnorm
        t2 = numpy.float32(1) + relArnorml
        flag = TT.switch(
            TT.bitwise_or(TT.eq(flag, numpy.float32(0.)),
                          TT.eq(flag, numpy.float32(6.))),
                      TT.switch(TT.le(t1, numpy.float32(1.)),
                                numpy.float32(3.),
                      TT.switch(TT.le(t2, numpy.float32(1.)),
                                numpy.float32(4.),
                      TT.switch(TT.le(relrnorm, rtol),
                                numpy.float32(1.),
                      TT.switch(TT.le(Anorm, numpy.float32(1e-20)),
                                numpy.float32(12),
                      TT.switch(TT.le(relArnorml, rtol),
                                numpy.float32(10.),
                      TT.switch(TT.ge(epsx, beta1),
                                numpy.float32(5.),
                      TT.switch(TT.ge(xnorm, maxxnorm),
                                numpy.float32(6.),
                      TT.switch(TT.ge(niter, TT.cast(maxit, 'float32')),
                                numpy.float32(8.),
                                flag)))))))),
            flag)

        flag = TT.switch(TT.lt(Axnorm, rtol * Anorm * xnorm),
                               numpy.float32(11.), flag)
        return [
            niter + numpy.float32(1.),
            beta,
            betan,
            phi,
            Acond,
            cs,
            dbarn,
            eplnn,
            rnorm,
            sn,
            Tnorm,
            rnorml,
            xnorm,
            Dnorm,
            gamma,
            pnorm,
            gammal,
            Axnorm,
            relrnorm,
            relArnorml,
            Anorm,
            flag] + xs + r1s + r2s + r3s + dls + ds, upds, \
                theano.scan_module.scan_utils.until(TT.neq(flag, 0))

    states = []
    # 0 niter
    states.append(TT.constant(numpy.float32([0])))
    # 1 beta
    states.append(TT.constant(numpy.float32([0])))
    # 2 betan
    states.append(TT.unbroadcast(TT.shape_padleft(beta1), 0))
    # 3 phi
    states.append(TT.unbroadcast(TT.shape_padleft(beta1), 0))
    # 4 Acond
    states.append(TT.constant(numpy.float32([1])))
    # 5 cs
    states.append(TT.constant(numpy.float32([-1])))
    # 6 dbarn
    states.append(TT.constant(numpy.float32([0])))
    # 7 eplnn
    states.append(TT.constant(numpy.float32([0])))
    # 8 rnorm
    states.append(TT.unbroadcast(TT.shape_padleft(beta1), 0))
    # 9 sn
    states.append(TT.constant(numpy.float32([0])))
    # 10 Tnorm
    states.append(TT.constant(numpy.float32([0])))
    # 11 rnorml
    states.append(TT.unbroadcast(TT.shape_padleft(beta1), 0))
    # 12 xnorm
    states.append(TT.constant(numpy.float32([0])))
    # 13 Dnorm
    states.append(TT.constant(numpy.float32([0])))
    # 14 gamma
    states.append(TT.constant(numpy.float32([0])))
    # 15 pnorm
    states.append(TT.constant(numpy.float32([0])))
    # 16 gammal
    states.append(TT.constant(numpy.float32([0])))
    # 17 Axnorm
    states.append(TT.constant(numpy.float32([0])))
    # 18 relrnorm
    states.append(TT.constant(numpy.float32([1])))
    # 19 relArnorml
    states.append(TT.constant(numpy.float32([1])))
    # 20 Anorm
    states.append(TT.constant(numpy.float32([0])))
    # 21 flag
    states.append(TT.constant(numpy.float32([0])))
    xs = [TT.unbroadcast(TT.shape_padleft(TT.zeros_like(b)), 0) for b in bs]
    ds = [TT.unbroadcast(TT.shape_padleft(TT.zeros_like(b)), 0) for b in bs]
    dls = [TT.unbroadcast(TT.shape_padleft(TT.zeros_like(b)), 0) for b in bs]
    r1s = [TT.unbroadcast(TT.shape_padleft(r1), 0) for r1 in r1s]
    r2s = [TT.unbroadcast(TT.shape_padleft(r2), 0) for r2 in r2s]
    r3s = [TT.unbroadcast(TT.shape_padleft(r3), 0) for r3 in r3s]

    rvals, lupds = scan(loop,
                    states=states + xs + r1s + r2s + r3s + dls + ds,
                    n_steps=maxit + numpy.int32(1),
                    name='minres',
                    profile=profile,
                    mode=mode)

    niters = TT.cast(rvals[0][0], 'int32')
    flag = TT.cast(rvals[21][0], 'int32')
    relres = rvals[18][0]
    relAres = rvals[19][0]
    Anorm = rvals[20][0]
    Acond = rvals[4][0]
    xnorm = rvals[12][0]
    Axnorm = rvals[17][0]
    sol = [x[0] for x in rvals[22: 22 + n_params]]
    return (sol, flag, niters, relres, relAres,
            Anorm, Acond, xnorm, Axnorm, lupds)


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
    compute_Av = lambda x: ([TT.dot(tA, x)], {})
    xs, flag, iters, relres, relAres, Anorm, Acond, xnorm, Axnorm, _ = \
            minres(compute_Av, [tb],
                   rtol=rtol,
                   maxit=maxit,
                   Ms=[tM],
                   profile=0)

    func = theano.function([],
                           xs + [flag, iters, relres, relAres, Anorm, Acond,
                                 xnorm, Axnorm],
                          name='func',
                          profile=0)
    rvals = func()
    print 'flag', rvals[1]
    print messages[int(rvals[1])]
    print 'iters', rvals[2]
    print 'relres', rvals[3]
    print 'relAres', rvals[4]
    print 'Anorm', rvals[5]
    print 'Acond', rvals[6]
    print 'xnorm', rvals[7]
    print 'Axnorm', rvals[8]
    print 'error', numpy.sqrt(numpy.sum((
        numpy.dot(rvals[0], A) - b) ** 2))
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
    maxit = 50
    tA = theano.shared(A.astype('float32'))
    tb = theano.shared(b.astype('float32'))
    compute_Av = lambda x: ([TT.dot(tA, x)], {})
    xs, flag, iters, relres, relAres, Anorm, Acond, xnorm, Axnorm, _ = \
            minres(compute_Av, [tb], rtol=rtol, maxit=maxit,
                   maxxnorm=maxxnorm, profile=0)

    func = theano.function([],
                           xs + [flag, iters, relres, relAres, Anorm, Acond,
                                 xnorm, Axnorm],
                          name='func',
                          profile=0)
    rvals = func()
    print 'flag', rvals[1]
    print messages[int(rvals[1])]
    print 'iters', rvals[2]
    print 'relres', rvals[3]
    print 'relAres', rvals[4]
    print 'Anorm', rvals[5]
    print 'Acond', rvals[6]
    print 'xnorm', rvals[7]
    print 'Axnorm', rvals[8]
    print rvals[0 ]


if __name__ == '__main__':
    test_1()
    test_2()
