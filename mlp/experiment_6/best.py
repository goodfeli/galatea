results = open('results.dat', 'r')
results = results.readlines()
mn = 1.1
mni = -1
for i, result in enumerate(results):
    result = result.split(' ')[0]
    if result == 'P':
        continue
    result = float(result)
    if result < mn:
        mn = result
        mni = i
print 'experiment %d (line %d): %f' % (mni, mni+1, mn)
