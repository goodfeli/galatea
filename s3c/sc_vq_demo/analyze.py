import sys

d = sys.argv[1]

import os

names = os.listdir(d)

C_to_res = {}

for name in names:

    path = d + '/' + name

    fold_str, C_txt = name.split('_')

    fold = int(fold_str)

    C_str = C_txt.replace('.txt','')

    C = float(C_str)

    f = open(path,'r')

    l = f.readlines()

    f.close()

    l ,= l

    l.replace('\n','')

    acc = float(l)

    if C not in C_to_res:
        C_to_res[C] = []

    C_to_res[C].append(acc)

num_folds = 5

for key in C_to_res:

    value = C_to_res[key]

    if len(value) != num_folds:
        print key, 'has only ',len(value)
        assert False

    C_to_res[key] = sum(value) / float(num_folds)

ranked_C = sorted(C_to_res.keys(), key = lambda idx : C_to_res[idx] )

for C in ranked_C:
    print C, C_to_res[C]
