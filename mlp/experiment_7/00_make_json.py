from pylearn2.config import yaml_parse

params = yaml_parse.load_path('params.yaml')
assert isinstance(params, list)

out = open('config.json', 'w')

print >>out, "{"

for i, param in enumerate(params):
    name = param['name']
    print >>out, '\t"' + name + '": {'
    print >>out, '\t\t"name": "%s",' % name
    t = param['type']
    print >>out, '\t\t"type": "%s",' % t
    if t == 'float':
        mn = param["min"]
        mx = param["max"]
        if 'transform' in param:
            transform = param['transform']
            mn = transform.inverse(mn)
            mx = transform.inverse(mx)
        print >>out, '\t\t"min": %f,' % param["min"]
        print >>out, '\t\t"max": %f,' % param["max"]
    elif t == 'int':
        if 'transform' in param:
            transform = param['transform']
            mn = transform.inverse(mn)
            mx = transform.inverse(mx)
        print >>out, '\t\t"min": %d,' % mn
        print >>out, '\t\t"max": %d,' % mx
    else:
        raise NotImplementedError()
    assert 'size' not in param
    print >>out, '\t\t"size": 1'
    if i == len(params) -1:
        print >>out, '\t}'
    else:
        print >>out, '\t},'
print >>out, '}'
out.close()
