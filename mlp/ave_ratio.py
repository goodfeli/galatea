from pylearn2.utils import serial
maxout = serial.load('expdir/grad_var.pkl')
rect = serial.load('expdir/grad_var_rect.pkl')

mm = maxout.monitor
rm = rect.monitor
mc = mm.channels
rc = rm.channels

m_last = 'subtrain_y_softmax_Wgrad_stdev'
m_first = 'subtrain_h0_h0_Wgrad_stdev'
m_last = mc[m_last]
m_first = mc[m_first]

r_last = 'sub_train_y_softmax_Wgrad_stdev'
r_first = 'sub_train_h0_h0_Wgrad_stdev'
r_last = rc[r_last]
r_first = rc[r_first]

m_last = m_last.val_record
m_first = m_first.val_record

r_last = r_last.val_record
r_first = r_first.val_record

first_ratios = [(m/r) ** 2. for m, r in zip(m_first, r_first)]
print sum(first_ratios)/float(len(first_ratios))

first_ratios = [(m/r) ** 2. for m, r in zip(m_last, r_last)]
print sum(first_ratios)/float(len(first_ratios))

