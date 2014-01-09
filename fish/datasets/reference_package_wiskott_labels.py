#! /usr/bin/env python

"
Note: this file was used only for dataset creation. It is provided here only as a reference
"
assert False, 'This file is only a reference and should probably not be run or imported.'



import os
import numpy as N
import SkyNet
from scipy import io

base = SkyNet.get_wiskott_path() + '/'

sets = os.listdir(base)
sets = [ x for x in sets if x != 'zips' ]

for set in sets:
    print 'making labels for '+set
    setdir = base + set

    is_fish = (set.find('fish') != -1)


    print '\treading labels'

    configs = io.loadmat(setdir+'/configs/configs.mat')
    print '\tformatting labels'

    x = configs['x'].T
    y = configs['y'].T
    phi_y = configs['phi_y'][0,:]
    oid = configs['id'][0,:]

    m = x.shape[0]

    assert m == y.shape[0]
    assert m == phi_y.shape[0]
    assert m == oid.shape[0]

    labels_mats = [x,y]

    phi_y_sin = N.sin(phi_y)
    phi_y_cos = N.cos(phi_y)

    if is_fish:
        num_id = 25
    else:
        num_id = 10
        phi_z = configs['phi_z'][0,:]
        phi_z_sin = N.sin(phi_z)
        phi_z_cos = N.cos(phi_z)
        phi_z_sin_mat = N.zeros((m,num_id))
        phi_z_cos_mat = N.zeros((m,num_id))
        assert m == phi_z.shape[0]

    id_mat = N.zeros((m,num_id))
    phi_y_sin_mat = N.zeros((m,num_id))
    phi_y_cos_mat = N.zeros((m,num_id))

    for i in xrange(num_id):
        mask = oid == i
        id_mat[:,i] = mask
        phi_y_sin_mat[:,i] = phi_y_sin
        phi_y_cos_mat[:,i] = phi_y_cos
        if not is_fish:
            phi_z_sin_mat[:,i] = phi_z_sin
            phi_z_cos_mat[:,i] = phi_z_cos
        ""
    ""

    labels_mats.append(id_mat)
    labels_mats.append(phi_y_sin_mat)
    labels_mats.append(phi_y_cos_mat)
    if not is_fish:
        labels_mats.append(phi_z_sin_mat)
        labels_mats.append(phi_z_cos_mat)

    labels_mat = N.concatenate(labels_mats, axis=1)



    viewsdir = setdir + '/views'

    read_pos = 0

    seqdirs = [seqdir for seqdir in sorted(os.listdir(viewsdir)) if seqdir.endswith('.seqdir') ]
    for seqdir in seqdirs:
        #print '\tsaving labels for '+seqdir
        fullseqdir = viewsdir + '/' + seqdir

        dtype = 'uint8'
        path = fullseqdir
        filepaths = [ path + '/' + frame for frame in sorted(os.listdir(path)) if frame.endswith('.png') ]
        cur_m = len(filepaths)
        assert cur_m > 0


        cur_mat = labels_mat[read_pos:read_pos+cur_m,:]
        read_pos += cur_m

        outfile = fullseqdir.replace('.seqdir','.labels.npy')
        #print '\t\tsaving as '+outfile
        N.save(outfile,cur_mat)
    assert read_pos == m
