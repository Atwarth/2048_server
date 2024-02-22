import torch
import torch.nn as nn
import numpy  as np
from collections import namedtuple, deque
import random
import logic
from pathlib import Path
from tqdm import tqdm
from itertools import count
import math


dest = r"C:\Users\Jorge Eliecer\Desktop\server\static\logs\logs.txt"
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "2048_tdr.pth"

MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

device = "cuda" if torch.cuda.is_available() else "cpu"


GAMMA = 0.9



        
class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        self.soft = nn.Softmax(dim=1)
        self.fn = nn.Flatten()
        self.conv_test = nn.Conv2d(16,128,kernel_size=(2,2),stride=(1,1))
        self.conv_test2 = nn.Conv2d(128,128,kernel_size=(2,2),stride=(1,1))
        self.L_s = nn.Linear(512,1)
        
        # self.conv1 = nn.Conv2d(16,128,kernel_size=(1,2),stride=(1,1))
        # self.conv2 = nn.Conv2d(16,128,kernel_size=(2,1),stride=(1,1))

        # self.conv_11 = nn.Conv2d(128,128,kernel_size=(1,2), stride=(1,1))
        # self.conv_12 = nn.Conv2d(128,128,kernel_size=(2,1), stride=(1,1))
        
        # self.conv_21 = nn.Conv2d(128,128,kernel_size=(1,2), stride=(1,1))
        # self.conv_22 = nn.Conv2d(128,128,kernel_size=(2,1), stride=(1,1))


        # self.L_ff = nn.Linear(7424,2048)
        # self.L_ff2 = nn.Linear(2048,1)
        # self.L_ff3 = nn.Linear(256,1)
        self.relu = nn.ReLU()
            

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        
        
        
    def forward(self,x):
        
        

        f1 = self.bn1(self.relu(self.conv_test(x)))
        f1 = self.bn2(self.relu(self.conv_test2(f1)))
        f1 = self.fn(f1)
        F = self.L_s(f1)
        # f1_1 = self.bn2(self.relu(self.conv_11(f1)))
        # f1_2 = self.bn3(self.relu(self.conv_12(f1)))
        
        # f1n = self.fn(f1)
        # fl1 = self.fn(f1_2)
        # fl2 = self.fn(f1_1)

        
        # f2 = self.bn1(self.relu(self.conv2(x)))
        # f2_1 = self.bn2(self.relu(self.conv_21(f1)))
        # f2_2 = self.bn3(self.relu(self.conv_22(f1)))
        
        # f2n = self.fn(f2)
        # f21 = self.fn(f2_1)
        # f22 = self.fn(f2_2)

        #F = torch.cat((f1n,f2n,fl1,fl2,f21,f22),1)
        #F = self.relu(self.L_ff(F))
        #F = self.L_ff2(F)
        #F = self.bn4(self.relu(self.L_ff3(F)))
       
        return F

policy_dqn = DQN().to(device)
target_dqn = DQN().to(device)
target_dqn.load_state_dict(policy_dqn.state_dict())
target_dqn.eval()

    
steps_done = 0       
def board_16(board):
    board = torch.as_tensor(board).to(device)
    p = 2
    z = torch.zeros(16,4,4)
    for i in range(4):
        for j in range(4):
            if board[i][j] == 0:
                z[0][i][j] = 1
                
    for i in range(1,len(z)):
        z[i] = board/p
        for j in range(4):
            for k in range(4):
                if z[i][j][k]!=1:
                    z[i][j][k] = 0
        p +=2
    return z.float().to(device)



def select_action(state):
    actions = [logic.move_left,logic.move_right,logic.move_up,logic.move_down]
    actions_state = [action(state)[0] for action in actions]
    dq = [policy_dqn(board_16(actions_state[0]).unsqueeze(0)).squeeze().cpu().detach().numpy(),policy_dqn(board_16(actions_state[1]).unsqueeze(0)).squeeze().cpu().detach().numpy(),policy_dqn(board_16(actions_state[2]).unsqueeze(0)).squeeze().cpu().detach().numpy()
                                ,policy_dqn(board_16(actions_state[3]).unsqueeze(0)).squeeze().cpu().detach().numpy()]
    #dq = policy_dqn(board_16(actions_state[]))
    argmx = np.flatnonzero(dq == np.max(dq))
    #print(argmx)
    if len(argmx)>1:
        argmx = random.choice([0,1,2,3])
    else:
        argmx = random.choice(argmx)
    
    #print(state)
    #print(argmx)
    if argmx == 5:
        next_state, _, reward = actions[random.randint(0,3)](state)
    else:
        next_state, _, reward = actions[argmx](state)
        
    next_state = torch.as_tensor(next_state).to(device).float()
    
    return next_state, _, reward 
            
    
def step(state):
    
    next_state_p,_,reward = select_action(state)
    done = logic.get_current_state(next_state_p)
    next_state_p = torch.as_tensor(next_state_p).to(device).float()
    reward = torch.as_tensor([reward]).to(device)
    if not done:
        next_state = torch.as_tensor(logic.add_new_2(next_state_p)).to(device).float()
        return next_state, reward, done, next_state_p
    else:
        return next_state_p, reward, done, next_state_p

    
Transition = namedtuple('Transition',
                         ('state', 'reward','next_state'))
        
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

memory = ReplayMemory(6000)

optimizer = torch.optim.RMSprop(policy_dqn.parameters())




N = 10000
losses = []
total_epochs = 0
total_penalties = 0
t = []
rws = []
batch_size = 32
#C = 10
t_reward = []
criterion = nn.SmoothL1Loss()


with open(dest,'w',buffering=1) as f:
    for epoch in tqdm(range(N)):

        state_t = torch.as_tensor(logic.start_game()).float().to(device)
        
        total_reward = 0
        steps = 0
        total_loss = []
        penalties = 0
        epochs = 0
        for _ in count():
            
            steps += 1
            
            state_next, reward,terminated, state_p = step(state_t)
            
                
            state_next_tensor = board_16(state_next).to(device).unsqueeze(0).float()
            state_p = board_16(state_p).to(device).unsqueeze(0).float()

            rd = reward[0].cpu().numpy()
            total_reward += rd
            memory.push(state_p,reward,state_next_tensor)
            
            if rd == 0:
                penalties+=1
                
            if terminated:
                #if len(memory) >= batch_size:
                
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))
                state_batch = torch.cat(batch.state).float()
                reward_batch = torch.cat(batch.reward).float()
                n_state_batch = torch.cat(batch.next_state).float()
                state_action_values = policy_dqn(state_batch)
                
                next_state_values = policy_dqn(n_state_batch).detach().squeeze()

                expected_state_action_values = (next_state_values * GAMMA) + reward_batch

                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                
                #for param in policy_dqn.parameters():
                #    param.grad.data.clamp_(-1, 1)
                    
                optimizer.step()
                total_loss.append(loss.item()/batch_size)
                break
           

            #print(state_next)
            
            


            state_t = state_next
            
            
            
            
        rws.append(total_reward)
        t.append(epoch+1)
        total_penalties += penalties
        total_epochs += steps
         
        f.write(f"epoch: {epoch+1}  [#]steps: {steps} [#] total_reward: {int(total_reward)} [#] penalties: {penalties} [#] avg_total_loss: {np.mean(total_loss):.3f} ")
        f.write(f"\n")
        f.write(f"{state_next}")
        f.write(f"\n")

     
torch.save(obj=policy_dqn.state_dict(), # only saving the state_dict() only saves the models learned parameters
                   f=MODEL_SAVE_PATH) 
                           
        