from preprocessing import *
from predictor import BPNetwork
from predictor import *
from left_time import Model
from left_time import * 

data4 = np.loadtxt('data4_normalized.txt', dtype=np.float32)
namelist = get_user_ids(dataset4)
model1 = BPNetwork(366, 70, 2)
model2 = Model(366, 300, 1)
model1 = torch.load('model1.pth')
model2 = torch.load('model2.pth')
pred1 = model1(torch.from_numpy(data4))
pred2 = model2(torch.from_numpy(data4))
pred1 = pred1[:, 0] > 0.5 # lost uesrs
data5 = pd.DataFrame()
length = pred1.shape[0]
last_users = []
final_time = []
for i in range(length):
    if pred1[i]:
        last_users.append(namelist[i])
        final_time.append(pred2[i].detach().item())

print(final_time)
data5['user_id'] = last_users
data5['final_left_time'] = final_time
print(data5)
data5.to_csv('data/5.csv')
