# 2048.py
 
# importing the logic.py file
# where we have written all the
# logic functions used.
import logic
import numpy as np
import copy
# Driver code
#if __name__ == '__main__':
     
# calling start_game function
# to initialize the matrix
    #mat = logic.start_game()
    #main(mat)
def dic(x1):
    d_o = {}
    for i in x1:
        for j in i:
            if j!=0 and j not in d_o:
                d_o[j] = 0
            if j in d_o:
                d_o[j]+=1
    return d_o
    
def rewards(old,mat):
    rewards = 0
    #n_2, n_4, n_8, n_16, n_32, n_64, n_128, n_256= 0
    #if np.count_nonzero(old) >= np.count_nonzero(mat):
    d_o = dict(sorted(dic(old).items()))
    d_n = dict(sorted(dic(mat).items()))
    print(d_o)
    print(d_n)
    for i in d_o:
        if i not in d_n:
            rewards+=i
    for i,j in zip(d_n,d_o):
        if i==j and (d_n[i]>d_o[j]):
            rewards+=i
    
    for i in d_n:        
        if i not in d_o:
            rewards+=i
    return rewards
def string_m(m):
    s = ""
    for i in range(len(m)):
        for j in range(len(m[0])):
            s+=str(m[i][j])
    return s
def env_reset(mat):
    m = logic.start_game()
    return m
def game(mat):
    reward = 0
    logic.add_new_2(mat)
    print(mat)
    while(True):
        print(reward)
        # taking the user input
        # for next step
        x = input("Press the command : ")
        #print(mat)
        # we have to move up
        old = copy.deepcopy(mat)
        o_s  = string_m(old)
        #print(np.count_nonzero(old))
        if(x == 'W' or x == 'w'):
            
            # call the move_up function
            mat, flag = logic.move_up(mat)
            m_s = string_m(mat)
            #print(flag)
            
            # get the current state and print it
            status = logic.get_current_state(mat)
            #print(status)
            if flag:
                reward += rewards(old,mat) 
                
            
            # if game not ove then continue
            # and add a new two
            if(status == True):
                break
            if o_s!=m_s:
                logic.add_new_2(mat)
            
                
        # the above process will be followed
        # in case of each type of move
        # below
     
        # to move down
        elif(x == 'S' or x == 's'):
            mat, flag = logic.move_down(mat)
            status = logic.get_current_state(mat)
            m_s = string_m(mat)
            #if flag:
            #    reward += rewards(old,mat) 
            
            if flag:
                reward += rewards(old,mat) 
                
            if status == True:
                break
            if o_s!=m_s:
                logic.add_new_2(mat)
            
     
        # to move left
        elif(x == 'A' or x == 'a'):
            mat, flag = logic.move_left(mat)
            status = logic.get_current_state(mat)
            m_s = string_m(mat)
            if flag:
                reward += rewards(old,mat) 
            if status == True:
                break
            if o_s!=m_s:
                logic.add_new_2(mat)
            #else:
            #    logic.add_new_2(mat)
        # to move right
        elif(x == 'D' or x == 'd'):
            mat, flag = logic.move_right(mat)
            status = logic.get_current_state(mat)
            m_s = string_m(mat)
            if flag:
                reward += rewards(old,mat) 
                
            if status == True:
                break
            if o_s!=m_s:
                logic.add_new_2(mat)  
            
        else:
            print("Invalid Key Pressed")
     
        # print the matrix after each
        # move.
        mat = np.array(mat)
        
        print(mat)
        #print(np.count_nonzero(mat))