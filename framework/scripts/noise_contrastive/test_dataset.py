import matplotlib.pyplot as plt
from framework.datasets.cos_dataset import CosDataset

d = CosDataset()

x = d.get_batch_design(100)

plt.scatter(x[:,0],x[:,1])
plt.show()
