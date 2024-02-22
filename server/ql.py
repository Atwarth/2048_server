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
import time

dest = r"C:\Users\Jorge Eliecer\Desktop\server\static\logs\logs.txt"

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
#MODEL_NAME_load = "2048_mc.pth"
MODEL_NAME = "2048_bn.pth"
#MODEL_SAVE_load = MODEL_PATH / MODEL_NAME_load
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

device = "cpu" if torch.cuda.is_available() else "cpu"
#print(device)
alpha = 0.1
GAMMA = 0.9
#epsilon = 0.6
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

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

        # return self.ll(x.view(x.size(0),-1))
        
# class DQN(nn.Module):
    # def __init__(self):
        # super().__init__()
        # self.fn = nn.Flatten()
        # self.conv1 = nn.Conv2d(4,16,kernel_size=(2,1))
        # self.lin = nn.Linear(96,256)
        # self.lin2 = nn.Linear(256,4)
        # self.relu = nn.ReLU()
        # self.conv2 = nn.Conv2d(16,32,1,1)
        # self.conv3 = nn.Conv2d(32,32,1,1)
        
        # self.bn1 = nn.BatchNorm2d(16)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(32)
        
        
        
    # def forward(self,x):
        # # if x.size(0)>=12:
        # #print(x.size())
        # x = x.permute(1,2,0).unsqueeze(0)
        # x = self.relu(self.conv1(x))
        # x = self.bn1(x)
        # x = self.relu(self.conv2(x))
        # x = self.bn2(x)
        # x = self.relu(self.conv3(x))
        # x = self.bn3(x)
        # x = x.view(x.size(3),x.size(1)*x.size(2))
        # return self.lin2(self.relu(self.lin(x)))
        # # else:
            # # x = x.unsqueeze(0).reshape(1,4,4,1)
            # # x = self.relu(self.conv1(x))
            # # x = self.bn1(x)
            # # x = self.relu(self.conv2(x))
            # # x = self.bn2(x)
            # # x = self.relu(self.conv3(x))
            # # x = self.bn3(x)
            # # x = self.ll(x.view(x.size(0),-1))
            # # return x.reshape(1,4)
            
            # #x = self.ln(x).reshape(4,4,4,4)
            
class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        self.soft = nn.Softmax(dim=1)
        self.fn = nn.Flatten()
        
        self.conv_test = nn.Conv2d(16,128,kernel_size=(2,2),stride=(1,1))
        self.conv_test2 = nn.Conv2d(128,128,kernel_size=(2,2),stride=(1,1))
        self.L_s = nn.Linear(512,256)
        self.L_s2 = nn.Linear(256,4)
        
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
        F = self.L_s2(self.L_s(f1))
        
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
       
        return self.soft(F)
        
policy_dqn = DQN().to(device)
target_dqn = DQN().to(device)

target_dqn.load_state_dict(policy_dqn.state_dict())
target_dqn.eval()

# def board_16(X):
    # power_mat = torch.zeros(16,4,4).float()
    # for i in range(4):
        # for j in range(4):
            # if(X[i][j]==0):
                # power_mat[0][i][j][0] = 1.0
            # else:
                # power = int(math.log(X[i][j],2))
                # power_mat[0][i][j][power] = 1.0
    # return power_mat   
    
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
        #z[i] = board/p
        for j in range(4):
            for k in range(4):
                if board[j][k]/p==1:
                    z[i][j][k] = 1
                else:
                    z[i][j][k] = 0
        p *=2
    return z.float().to(device)

def select_action(state):
    global steps_done
    #sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.uniform(0,1) < eps_threshold:
        return torch.as_tensor([[random.randrange(4)]]).to(device)
    else:
        #state = torch.as_tensor(change_values(state)).float().to(device)
        with torch.no_grad():
            state = torch.as_tensor(board_16(state)).to(device).float().unsqueeze(0)
        #print(state.size())
            #arg = torch.argmax(policy_dqn(state))
            #print(arg)
            return  policy_dqn(state).max(1)[1].view(1, 1)#
def select_action_2(state):
    global steps_done
    #sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if eps_threshold < eps_threshold:
        return torch.as_tensor([[random.randrange(4)]]).to(device)
    else:
        #state = torch.as_tensor(change_values(state)).float().to(device)
        with torch.no_grad():
            state = torch.as_tensor(board_16(state)).to(device).float().unsqueeze(0)
        #print(state.size())
            #arg = torch.argmax(policy_dqn(state))
            #print(arg)
            return  policy_dqn(state).max(1)[1].view(1, 1)#    
def step(state):
    actis = [logic.move_left,logic.move_right,logic.move_up,logic.move_down]
    action = select_action(state)
    
    
    next_state,_,reward = actis[action](state)
    #print(next_state)
    #print(action)
    next_state = logic.add_new_2(next_state)
    terminated = torch.as_tensor([logic.get_current_state(next_state)]).float()
    #next_state = torch.Tensor(next_state).to(device).float().requires_grad_()
    next_state = torch.as_tensor(next_state).to(device)
    #print(next_state.size())
    reward = torch.as_tensor([reward]).to(device)
    #print(reward)
    return next_state, reward, terminated, action

def step_2(state):
    actis = [logic.move_left,logic.move_right,logic.move_up,logic.move_down]
    action = select_action_2(state)
    
    
    next_state,_,reward = actis[action](state)
    #print(next_state)
    #print(action)
    next_state = logic.add_new_2(next_state)
    terminated = logic.get_current_state(next_state)
    #next_state = torch.Tensor(next_state).to(device).float().requires_grad_()
    next_state = torch.as_tensor(next_state).to(device)
    #print(next_state.size())
    reward = torch.as_tensor([reward]).to(device)
    #print(reward)
    return next_state, reward, terminated, action
    
Transition = namedtuple('Transition',
                         ('state', 'action',  'reward','next_state', 'done'))
        
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
        
def main():
    #policy_dqn.load_state_dict(torch.load(MODEL_SAVE_load))
        
    memory = ReplayMemory(10000)

    optimizer = torch.optim.RMSprop(policy_dqn.parameters())
    #board = torch.Tensor(logic.start_game()).unsqueeze(dim=-1).unsqueeze(dim=-1).float()
    #board = torch.from_numpy(board)

    N = 5000
    
    
    total_penalties = 0
    #t = []
    rws = []
    batch_size = 128
    #C = 10
    #t_reward = []
    criterion = nn.MSELoss()
    TAU = 0.005

    with open(dest,'w',buffering=1) as f:
        for epoch in tqdm(range(N)):
            state_number = logic.start_game()
            state_t = torch.as_tensor(state_number).float().to(device)
            
            total_reward = 0
            steps = 0
            total_loss = [0]
            penalties = 0
            epochs = 0
            
            for _ in count():

                steps += 1
                
                state_next, reward,terminated, action = step(state_t)
                state_next_tensor_16 = board_16(state_next).to(device).unsqueeze(0).float()
                #print(state_next_tensor_16.size())
                state_tensor_16 = board_16(state_t).to(device).unsqueeze(0).float()
                #print(state_tensor_16.size())
                rd = reward[0].cpu().numpy()
                total_reward += rd

                if reward[0] == 0:
                    penalties+=1
                
                
                
                
                memory.push(state_tensor_16,action,reward,state_next_tensor_16, terminated)
                state_t = state_next
                if len(memory)>= batch_size:
                    #print(state_next)
                    transitions = memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))
                    
                    
                    action_batch = torch.cat(batch.action).float()
                    n_state_batch = torch.cat(batch.next_state).float()
                    reward_batch = torch.cat(batch.reward).float()
                    state_batch = torch.cat(batch.state).float()
                    done_batch = torch.cat(batch.done).float()
                    #print(state_batch.size())
                    state_action_values = policy_dqn(state_batch).gather(1, action_batch.long()).float()
                    #print(state_batch)
                    next_state_values = target_dqn(n_state_batch).max(1)[0].detach().squeeze()
            
                    expected_state_action_values = ((1-done_batch)*(next_state_values * GAMMA) + reward_batch).unsqueeze(1)
                    #print(expected_state_action_values)
                    #print(state_action_values)
                    #.unsqueeze(1)
                    #print(done_batch)
                    loss = criterion(state_action_values, expected_state_action_values)
                    optimizer.zero_grad()
                    loss.backward()
                    #print(loss)
                    #for param in policy_dqn.parameters():
                    #    param.grad.data.clamp_(-1, 1)
                    torch.nn.utils.clip_grad_value_(policy_dqn.parameters(), 1.0)
                    optimizer.step()
                    total_loss.append(loss.item()/batch_size)
                    
                target_net_state_dict = target_dqn.state_dict()
                policy_net_state_dict = policy_dqn.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_dqn.load_state_dict(target_net_state_dict)

                
                
                if terminated[0]:

                    break
                    
            rws.append(total_reward)
            #t.append(epoch+1)
            total_penalties += penalties
            
            #s = np.array(state_next)
             
            f.write(f"epoch: {epoch+1}  [#]steps: {steps} [#] total_reward: {int(total_reward)} [#] penalties: {penalties} [#] avg_total_loss: {np.mean(total_loss)} ")
            f.write(f"\n")
            f.write(f"{state_next}")
            f.write(f"\n")

         
    torch.save(obj=policy_dqn.state_dict(), # only saving the state_dict() only saves the models learned parameters
                       f=MODEL_SAVE_PATH) 
                   
def test():
    
    policy_dqn.load_state_dict(torch.load(MODEL_SAVE_load))
    state_number = logic.start_game()
    state_t = torch.as_tensor(state_number).float().to(device)
    
    total_reward = 0
    steps = 0
    total_loss = []
    penalties = 0
    epochs = 0
    
    for _ in count():

        steps += 1
        
        state_next, reward,terminated, action = step_2(state_t)
        #state_next_tensor_16 = board_16(state_next).to(device).unsqueeze(0).float()
        #print(state_next_tensor_16.size())
        #state_tensor_16 = board_16(state_t).to(device).unsqueeze(0).float()
        #print(state_tensor_16.size())
        rd = reward[0].cpu().numpy()
        total_reward += rd

        #if reward == 0:
        #    penalties+=1
        if terminated:

            break
        state_t = state_next
        print(f"{state_t}\n")
        time.sleep(1)
if __name__ == "__main__":
    main()