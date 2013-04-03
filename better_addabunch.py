import os
from pylearn2.utils.shell import run_shell_command
suffixes = ['.py', '.m', '.sh', '.yaml', 'Makefile', '.c', '.cpp', '.pdf', '.lyx']

def recurse_add(d):
    print "Exploring "+d
    files = os.listdir(d)
    for f in files:
        if f.startswith('.'):
            continue
        if f == 'LOGS':
            continue
        full_path = d + '/' + f
        if os.path.isdir(full_path):
            recurse_add(full_path)
        elif any(f.endswith(suffix) for suffix in suffixes):
            run_shell_command("git add "+full_path)

recurse_add('.')
