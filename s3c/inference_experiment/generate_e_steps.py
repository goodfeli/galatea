from pylearn2.utils import serial

def emit_cg(directory, n, eta_h):
    h_schedule = [ eta_h ] * n

    for steps in [1,2,3]:
        s_schedule = [ steps ] * n

        filepath = directory + '/s'+str(steps)+'.yaml'

        f = open(filepath,'w')

        f.write("""!obj:galatea.s3c.s3c.E_Step_CG_Scan {
                        "h_new_coeff_schedule" : %s,
                        "s_max_iters" : %s,
               }
               """ % (str(h_schedule),str(s_schedule)))

        f.close()

def emit_heuristic(directory, n, eta_h):
    h_schedule = [ eta_h ] * n

    for s_eta in [.25, .5, .75]:
        s_schedule = [ s_eta ] * n

        filepath = directory + '/s'+str(s_eta) + '.yaml'

        f = open(filepath,'w')

        f.write("""!obj:galatea.s3c.s3c.E_Step_Scan {
                        "h_new_coeff_schedule" : %s,
                        "s_new_coeff_schedule" : %s,
                        "clip_reflections" : 1,
               }
               """ % (str(h_schedule), str(s_schedule)))

        f.close()

def emit_eta_h(method, directory, n, eta_h):
    directory = directory + '/eta_h_'+str(eta_h)

    serial.mkdir(directory)

    if method == 'cg':
        emit_cg(directory, n, eta_h)
    else:
        assert method == 'heuristic'
        emit_heuristic(directory, n, eta_h)

def emit_n(method, directory, n):
    directory = directory + '/n'+str(n)

    for eta_h in [.25, .5, .75]:
        emit_eta_h(method, directory, n, eta_h)

def emit_method(method):
    directory = 'e_steps/'+method

    for n in [5,10,15,20]:
        emit_n(method, directory, n)

for method in ['cg','heuristic']:
    emit_method(method)


