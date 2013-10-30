#usage: python analyze.py <directory created by launch_workers>
#prints out validation error for each value of C, with best printed last

import sys

ignore, target = sys.argv

import os
files = os.listdir(target)


C_to_results = {}

for name in files:

    if name.find('log') != -1:
        continue

    path = target + '/' + name

    f = open(path,'r')

    lines = f.readlines()

    f.close()

    header = 'C\tfold\tvalidation accuracy\n'

    assert lines[0] == header
    assert len(lines) == 2

    C_str, fold_str, acc_str = lines[1].split()

    C = float(C_str)
    fold = int(fold_str)
    acc = float(acc_str)


    if C not in C_to_results:
        C_to_results[C] = []

    C_to_results[C].append(acc)

num_folds = len(C_to_results.values()[0])

best_acc = -1.

for C in C_to_results:
    results = C_to_results[C]

    assert len(results) == num_folds

    result = sum(results) / float(num_folds)

    print '%(C)f: %(result)f' % locals()

    if result > best_acc:
        best_acc = result
        best_C = C

print '%d-fold valid acc: ' % num_folds,best_acc
print 'best C: ',best_C

