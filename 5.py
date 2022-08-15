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
options = []
for i in range(length):
    if pred1[i]:
        last_users.append(namelist[i])
        time = pred2[i].detach().item() * 180
        user = data4[i].reshape([3, 122])
        user = user[0] + user[1] + user[2]
        tick = 0
        for j in range(122):
            if user[121 - j] / max(user) >= 0.03:
                tick = j
                break
        time = time - 3 * tick
        final_time.append(time)
        if time <= 30:
            options.append('A')
        elif time <= 90:
            options.append('B')
        elif time <= 180:
            options.append('C')
        else:
            options.append('D')

print(final_time)
data5['user_id'] = last_users
data5['final_left_time'] = options
print(data5)
data5.to_csv('data/5.csv')
