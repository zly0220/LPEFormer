import pandas as pd
import os

data_root = "/home/user/zhangliyuan/bobo_workspace/trans-ppi/data_full"

for i in range(1,6):
    csv_train = os.path.join(data_root,"{}/cv_test_{}.csv".format(i,i))
    data_frame = pd.read_csv(csv_train)
    total = data_frame.iloc[:,-1].to_numpy()
    num_total = len(total)
    num_pos = total.sum()
    num_neg = num_total - num_pos
    print(num_neg/num_total)
    
    