import numpy as np

train_size = np.asarray([50,100,200,400,1000])
#for each train size m, sample m labeled examples per class
#from cifar10 train set, train svm on just these examples, then
#evaluate on test set. Do this repeatedly, and use the average
#as the model's accuracy, the standard dev as its confidence interval


#sparse coding, from Adam's ICML 2011 paper
sc_acc = np.asarray([ .5375, .58, .62, .665, .7075 ])
sc_confidence = np.asarray([ .0075, .0075, .0075, .0075, .00375] )
#Thresh (alpha = 0.25), from Adam's ICML 2011 paper
t_acc = np.asarray([  .465, .5475, .5875, .6456, .7025 ] )
t_confidence = np.asarray([ .015, .0075, .005, .00725, .004 ] )
#spike and slab coding, from G_interm exp_h 3x3
s3c_acc = np.asarray([0.539056, 0.597475, 0.643667, 0.680640, 0.723340])
s3c_confidence = np.asarray([0.009359, 0.006416, 0.004602, 0.002851, 0.002314])

sc_rel_acc = (sc_acc - t_acc) #/ t_confidence
s3c_rel_acc = (s3c_acc - t_acc) #/ t_confidence

import matplotlib.pyplot as plt

plt.hold(True)

plt.plot(train_size,s3c_rel_acc, color="green", label="S3C")
#plt.plot(train_size,s3c_acc-s3c_confidence, color="green", linestyle="dashed")
#plt.plot(train_size,s3c_acc+s3c_confidence, color="green", linestyle="dashed")

plt.plot(train_size,sc_rel_acc, color="blue", label="SC")
#plt.plot(train_size,sc_acc-sc_confidence, color="blue", linestyle="dashed")
#plt.plot(train_size,sc_acc+sc_confidence, color="blue", linestyle="dashed")

#plt.plot(train_size,t_acc, color='red', label='Thresh')
#plt.plot(train_size,t_acc-t_confidence,color='red',linestyle='dashed')
#plt.plot(train_size,t_acc+t_confidence,color='red',linestyle='dashed')

plt.xlabel('Labeled Training Examples Per Class',fontsize=18)
plt.ylabel('Test Set Accuracy',fontsize=18)

plt.title('CIFAR-10 Learning Curve', fontsize=20)

plt.legend(loc = 4)

plt.show()

