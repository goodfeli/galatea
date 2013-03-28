import scipy
from scikits.learn import datasets
import numpy
class Iris:
    def __init__(self):
        import pdb; pdb.set_trace()

        self.x = datasets.load_iris()['data']
        self.y = datasets.load_iris()['target']

        indices = []

if __name__ == '__main__':
    
    dataset = Iris()
    import pdb; pdb.set_trace()

