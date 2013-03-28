from pylearn2.utils import serial

path = 'test_comp.pkl'
d = serial.load(path)

x = d['x']
ys = d['ys']
ys = zip(*ys)
model_paths = d['model_paths']
model_paths = [ path.split('/')[-1].replace('.pkl','') for path in model_paths]


from matplotlib import pyplot
pyplot.hold(True)

for y, path in zip(ys, model_paths):
    pyplot.plot(x, y, label = path)
pyplot.legend()

pyplot.xlabel('probability of dropping an input unit')
pyplot.ylabel('inpainting cost (negative log likelihood)')

print 'showing'
pyplot.show()
