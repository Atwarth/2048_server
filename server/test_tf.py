import math
import torch
from torch import nn
from collections import namedtuple, deque
import random

# class DQN(nn.Module):
    # def __init__(self):
        # super().__init__()
        # self.fn = nn.Flatten()
        
        
        
        # self.conv1 = nn.Conv2d(4,16,2,2)
        # #self.bn1 = nn.BatchNorm2d(16)
        # self.relu = nn.ReLU()
        # self.conv2 = nn.Conv2d(16,64,2,2)
        # #self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(64,256,1,1)
        
        # #self.bn3 = nn.BatchNorm2d(124)
        
        # self.ln = nn.Linear(16,256)
        # self.ln1 = nn.Linear(256,512)
        # self.ln2 = nn.Linear(512,1024)
        # self.ln3 = nn.Linear(1024,4)
        # self.ln4 = nn.Linear(2048,4)
        
        
        # # self.ln = nn.Linear(4352,4)
        # # self.cv1 = nn.Conv2d(4,128,kernel_size = (1,14), stride= 1)
        # # self.cv2 = nn.Conv2d(4,128,kernel_size = (1,14), stride= 1)

        # # self.cv_l11 = nn.Conv2d(4,3,kernel_size = (1,1), stride= 1)
        # # self.cv_l12 = nn.Conv2d(4,4,kernel_size = (2,1), stride= 1)

        # # self.cv_l21 = nn.Conv2d(3,3,kernel_size = (2,1), stride= 1)
        # # self.cv_l22 = nn.Conv2d(3,2,kernel_size = (1,1), stride= 1)
        
        
    # def forward(self,x):

        
        # # f1 = self.cv1(x).reshape(1,4,3,128)
        # # f1_1 = self.relu(self.cv_l11(f1))
        # # f1_2 = self.relu(self.cv_l12(f1))


        # # f2 = self.cv2(x).reshape(1,3,4,128)
        # # f2_1 = self.relu(self.cv_l21(f2))
        # # f2_2 = self.relu(self.cv_l22(f2))
        # # l1 = self.relu(f2_1.view((-1,1)).squeeze())
        # # l2 = self.relu(f2_2.view((-1,1)).squeeze())
        # # l3 = self.relu(f1_2.view((-1,1)).squeeze())
        # # l4 = self.relu(f1_1.view((-1,1)).squeeze())
        # # l = torch.cat((l1,l2,l3,l4))
        # # x = self.ln(l)
        
        

        
        # #x = self.bn1(x)
        
        
        

        # #x = self.bn2(x)
        

        
        # #x = self.bn3(x)
        
        # x = x.view((-1,1)).squeeze().reshape(1,16)
        # x = self.ln(x).reshape(4,4,4,4)
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.conv3(x)
        # x = self.relu(x)
        # x = x.view((-1,1))
        # x = self.ln3(x)
        # #self.relu(self.ln4(x))
        # #x = self.fn(x).view((-1,1)).reshape(256,1).squeeze()
        # #x = self.ln1(x)
        
        

        # return x
        
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = nn.Flatten()
        self.conv1 = nn.Conv2d(1,16,kernel_size=(1,1))
        self.l1 = nn.Linear(32,256)
        self.l2 = nn.Linear(256,4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16,32,2,2)
        self.conv3 = nn.Conv2d(32,32,2,2)
        
    def forward(self,x):
        #print(x.size())
        x = x.unsqueeze(1)
        #print(x.size())
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        #print(x)
        x = x.view(x.size(0),x.size(1))
        #print(x.size())
        return self.l2(self.l1(x))
        
# class DQN(nn.Module):
    # def __init__(self):
        # super().__init__()
        # self.fn = nn.Flatten()
        # self.conv1 = nn.Conv2d(4,16,1,1)
        # self.relu = nn.ReLU()
        # self.conv2 = nn.Conv2d(16,32,1,1)
        # self.conv3 = nn.Conv2d(32,64,1,1)
        # self.ll = nn.Linear(256,4)
        
    # def forward(self,x):

        # x = x.unsqueeze(0).reshape(12,4,4,1)
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))
        # x = self.ll(x.view(x.size(0),-1))
        # print(x.size())
        # return x
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)  
 
policy_net = DQN()
memory = ReplayMemory(1000)

Transition = namedtuple('Transition',
                         ('state', 'action',  'reward','next_state'))
                         
memory.push(torch.tensor([[[0,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([3]),torch.tensor([[[0,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))
memory.push(torch.tensor([[[0,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([1]),torch.tensor([[[0,2,0,0],[2,2,0,0],[0,0,0,0],[0,0,0,0]]]))
memory.push(torch.tensor([[[0,2,0,0],[0,2,0,0],[0,4,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([0]),torch.tensor([[[0,2,0,0],[0,2,0,0],[0,4,0,0],[0,0,0,0]]]))

memory.push(torch.tensor([[[0,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,4,0]]]),torch.tensor([[random.randrange(4)]],dtype=torch.long),torch.tensor([5]),torch.tensor([[[0,2,0,0],[0,2,0,0],[0,0,0,0],[0,4,0,0]]]))

memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([3]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))
memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([3]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))

memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([3]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))

memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([3]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))

memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([2]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))

memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([2]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))

memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([1]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))

memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([0]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))

memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([6]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))

memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([1]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))

memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([6]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))

memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([0]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))

memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([1]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))
memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([3]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))
memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([4]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))
memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([0]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))
memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([4]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))
memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([1]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))
memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([0]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))
memory.push(torch.tensor([[[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]),torch.tensor([[random.randrange(4)]], dtype=torch.long),torch.tensor([2]),torch.tensor([[[2,2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,0]]]))


transitions = memory.sample(12)
batch = Transition(*zip(*transitions))

state_batch = torch.cat(batch.state).float()
#print(state_batch.size())
action_batch = torch.cat(batch.action).float()
reward_batch = torch.cat(batch.reward).float()
# print(state_batch.size())
# #print(state_batch.unsqueeze(0).size())
# GAMMA  = 0.9
# state_action_values = policy_net(state_batch).gather(1, action_batch.long())

# next_state_values = torch.zeros(12)

# non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          # batch.next_state)),  dtype=torch.bool).long()

# non_final_next_states = torch.cat([s for s in batch.next_state
                                                # if s is not None]).float()

# next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1)[0].detach()
# expected_state_action_values = (next_state_values * GAMMA) + reward_batch
# #print(expected_state_action_values)
# #print(state_action_values)

# criterion = nn.SmoothL1Loss()
# loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
# #print(state_action_values)
# #print(expected_state_action_values)
# print(loss.item())
# #optimizer.zero_grad()
# loss.backward()
    
# #optimizer.step()
# # EPS_START = 0.9
# # EPS_END = 0.05
# # EPS_DECAY = 200

# # steps_done = 1000

# # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        # # math.exp(-1. * steps_done / EPS_DECAY)

# # print(eps_threshold)
tof = torch.Tensor([[0,2,0,4],[0,2,0,0],[0,0,0,0],[0,0,0,0]])

#print(policy_net(state_batch).gather(1, action_batch.long()))
#print(policy_net(tof.unsqueeze(0)))

m = torch.zeros(16,4,4)
m[:] = tof
print(m)
