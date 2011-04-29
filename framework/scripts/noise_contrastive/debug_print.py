from theano import function
from theano.printing import Print
import theano.tensor as T

x = T.scalar()

x = Print('x')(x)

two_x = 2. * x

two_x = Print('two_x')(two_x)

func = function([x], two_x)

func(0.0)
