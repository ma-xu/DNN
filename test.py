import numpy as np
X,Y = np.meshgrid(np.arange(4),np.arange((5)))

pool_list=[1,2,4]
print(14/3)
for pool_num, num_pool_regions in enumerate(pool_list):
    print("pool_num is : ", pool_num)
    print("num_pool_regions is : ", num_pool_regions)