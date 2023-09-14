from typing import Dict, List, Tuple

from numpy.typing import NDArray
import arcle
import gymnasium as gym
import time
import numpy as np
import pickle
from arcle.loaders import ARCLoader, Loader, MiniARCLoader

loader = ARCLoader()
miniloader = MiniARCLoader()
def findbyname(name):
    for i, aa in enumerate(loader.data):
        if aa[4]['id'] == name:
            return i
    for i, aa in enumerate(miniloader.data):
        if aa[4]['id'] == name:
            return i

def action_convert(action_entry):
    _, action, data, grid = action_entry
    sel = np.zeros((30,30), dtype=np.bool_)
    op = 0
    match action:
        case "CopyFromInput":
            op = 31
        case "ResizeGrid":
            op = 33
            h, w = data[0]
            sel[:h,:w] = 1
        case "ResetGrid":
            op = 32
        case "Submit":
            op = 34
        case "Color":
            h, w = data[0]
            op = data[1]
            sel[h,w] = 1

        case "Fill":
            h0, w0 = data[0]
            h1, w1 = data[1]
            op = data[2]
            sel[h0:h1+1 , w0:w1+1] = 1

        case "FlipX":
            h0, w0 = data[0]
            h1, w1 = data[1]
            op = 27
            sel[h0:h1+1, w0:w1+1] = 1
        case "FlipY":
            h0, w0 = data[0]
            h1, w1 = data[1]
            op = 26
            sel[h0:h1+1, w0:w1+1] = 1
        case "RotateCW":
            h0, w0 = data[0]
            h1, w1 = data[1]
            op = 25
            sel[h0:h1+1, w0:w1+1] = 1
        case "RotateCCW":
            h0, w0 = data[0]
            h1, w1 = data[1]
            op = 24
            sel[h0:h1+1, w0:w1+1] = 1
        case "Move":
            h0, w0 = data[0]
            h1, w1 = data[1]
            match data[2]:
                case 'U':
                    op = 20
                case 'D':
                    op = 21
                case 'R':
                    op = 22
                case 'L':
                    op = 23

            sel[h0:h1+1, w0:w1+1] = 1
        
        case "Copy":
            h0, w0 = data[0]
            h1, w1 = data[1]
            match data[2]:
                case 'Input Grid':
                    op = 28
                case 'Output Grid':
                    op = 29
            sel[h0:h1+1, w0:w1+1] = 1
        case "Paste":
            h, w = data[0]
            op = 30
            sel[h,w] = 1

        case "FloodFill":
            h, w = data[0]
            op = 10 + data[1]
            sel[h,w] = 1

    return op, sel

traces = []
traces_info = []
with open('tests/test.pickle', 'rb') as fp:
    traces:List = pickle.load(fp)
    
with open('tests/test_info.pickle', 'rb') as fp:
    traces_info:List = pickle.load(fp)

    
render_mode =None# 'ansi'

arcenv = gym.make('ARCLE/O2ARCv2Env-v0',render_mode=render_mode,data_loader= ARCLoader(), max_grid_size=(30,30), colors = 10, max_episode_steps=None)
minienv = gym.make('ARCLE/O2ARCv2Env-v0',render_mode=render_mode, data_loader=MiniARCLoader(), max_grid_size=(30,30), colors = 10, max_episode_steps=None)

failure_trace = []
error_trace = []
omitted_trace = []
tested = 0
'''
wanna_test = [235] # Interface Errors
traces = [traces[i] for i in wanna_test]
traces_info = [traces_info[i] for i in wanna_test]
'''
for idx, trace in enumerate(traces):
    i = 0

    if len(traces_info[idx][0]) >10:
        env = minienv
    else:
        env = arcenv

    obs, info = env.reset(options= {'adaptation':False, 'prob_index':findbyname(traces_info[idx][0]), 'subprob_index': traces_info[idx][1]})
    converted = []

    omit_trace = False
    for entry in trace:
        try:
            converted.append(action_convert(entry))
            #print(entry[:-1])
        except:
            omit_trace=True
            break
    if omit_trace:
        omitted_trace.append(idx)
        continue

    tested+=1
    for i in range(len(converted)):

        op, sel = converted[i]
        h,w = obs['grid_dim']

        if 20<= op <= 27 and np.all(obs['selected'] == sel):
            sel = np.zeros((30,30), dtype=np.bool_)
        
        action = {'selection': sel, 'operation': op}

        #input()
        #print('\033[F', end='')
        try:
            obs,reward,term,trunc,info = env.step(action)
        
        except:
            error_trace.append(idx)
            break
            
        h,w = obs['grid_dim']
        if  trace[i][3].shape != (h,w) or np.any(obs['grid'][:h,:w] != trace[i][3].astype(np.uint8)):
            #print(obs['grid'][:h,:w], trace[i][3])
            #exit()
            failure_trace.append(idx)
            #time.sleep(5)
            break
            
        
        if term or trunc:
            break

print(f'Tested: {tested}, Passed: {(tested-len(error_trace)-len(failure_trace))/(tested)*100:.2f}%')
print('Error traces:', error_trace)
print('Failure traces:', failure_trace)
print('Omitted traces:', omitted_trace)

    
env.close()
