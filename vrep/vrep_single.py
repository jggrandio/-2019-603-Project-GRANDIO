import vrep
import numpy as np
import time
import sys
import math
import time
import matplotlib.pyplot as mlp
from DQNagent import Agent
from collections import deque
import random


vrep.simxFinish(-1) # just in case, close all opened connections
start_time = time.time()
clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
print(clientID) # if 1, then we are connected.
if clientID!=-1:
    print ("Connected to remote API server")
else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")

#To work by steps
#vrep.simxSynchronous(clientID,True)
returnCode=vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
#vrep.simxSynchronousTrigger(clientID);
#vrep.simxGetPingTime(clientID)4
#Connecting to all the object
err_code,camera = vrep.simxGetObjectHandle(clientID,"camera", vrep.simx_opmode_blocking)
err_code,cuboid = vrep.simxGetObjectHandle(clientID,"Cuboid", vrep.simx_opmode_blocking)
err_code,car = vrep.simxGetObjectHandle(clientID,"differential", vrep.simx_opmode_blocking)
p=np.zeros(6, dtype=int)
ppos=np.zeros((6,3))
pang=np.zeros((6,3))
for i in range(len(p)):
    err_code,p[i] = vrep.simxGetObjectHandle(clientID,"p"+str(i), vrep.simx_opmode_blocking)

#First call to cuboid and camera
errorCode,resolution,image=vrep.simxGetVisionSensorImage(clientID, camera, 0,vrep.simx_opmode_streaming )
position=vrep.simxGetObjectPosition(clientID, cuboid, -1, vrep.simx_opmode_streaming)
car_position=vrep.simxGetObjectPosition(clientID, car, -1, vrep.simx_opmode_streaming)

#se to calculate relative positions instaid of absolute
for i in range(len(p)):
    _,ppos[i]=vrep.simxGetObjectPosition(clientID, p[i], -1, vrep.simx_opmode_streaming)
    _,pang[i]=vrep.simxGetObjectOrientation(clientID, p[i], -1, vrep.simx_opmode_streaming)

returnCode=vrep.simxSetJointTargetVelocity(clientID,p[4],0,vrep.simx_opmode_oneshot)
returnCode=vrep.simxSetJointTargetVelocity(clientID,p[0],0,vrep.simx_opmode_oneshot)

#vrep.simxSynchronousTrigger(clientID);
#vrep.simxGetPingTime(clientID)
vrep.simxCallScriptFunction(clientID,"Dummy",vrep.sim_scripttype_customizationscript,"start2StepRun", [], [], [], bytearray(),vrep.simx_opmode_blocking)

_,in_car_position=vrep.simxGetObjectPosition(clientID, car, -1, vrep.simx_opmode_buffer)
_,in_car_angle=vrep.simxGetObjectOrientation(clientID, car, -1, vrep.simx_opmode_buffer)
_,in_position=vrep.simxGetObjectPosition(clientID, cuboid, -1, vrep.simx_opmode_buffer)
_,in_angle=vrep.simxGetObjectOrientation(clientID, cuboid, -1, vrep.simx_opmode_buffer)

for i in range(len(p)):
    _,ppos[i]=vrep.simxGetObjectPosition(clientID, p[i], -1, vrep.simx_opmode_buffer)
    _,pang[i]=vrep.simxGetObjectOrientation(clientID, p[i], -1, vrep.simx_opmode_buffer)







scores = deque(maxlen=50)
epi_reward_average = []

agent = Agent(state_size = 5,action_size = 25, gamma=0.99, epsilon=1, epsilon_min=0.01, epsilon_decay=0.992, learning_rate=0.005, batch_size=64, tau=0.01)
reward_need = 0

for e in range(500):
    #put everything in place
    #In this case is always ahead of the car
    cub_pos=[in_position[0],in_position[1],in_position[2]]
    _=vrep.simxSetObjectPosition(clientID, car, -1,in_car_position,vrep.simx_opmode_oneshot)
    _=vrep.simxSetObjectOrientation(clientID, car, -1,in_car_angle,vrep.simx_opmode_oneshot)
    _=vrep.simxSetObjectPosition(clientID, cuboid, -1,cub_pos,vrep.simx_opmode_oneshot)
    _=vrep.simxSetObjectOrientation(clientID, cuboid, -1,in_angle,vrep.simx_opmode_oneshot)
    for i in range(len(p)):
        _=vrep.simxSetObjectPosition(clientID, p[i], -1,ppos[i], vrep.simx_opmode_oneshot)
        _=vrep.simxSetObjectOrientation(clientID, p[i], -1,pang[i],vrep.simx_opmode_oneshot)


    #Motors to 0 Just in case
    returnCode=vrep.simxSetJointTargetVelocity(clientID,p[4],0,vrep.simx_opmode_oneshot)
    returnCode=vrep.simxSetJointTargetVelocity(clientID,p[0],0,vrep.simx_opmode_oneshot)
    #run first step
    #vrep.simxSynchronousTrigger(clientID);
    #vrep.simxGetPingTime(clientID);
    vrep.simxCallScriptFunction(clientID,"Dummy",vrep.sim_scripttype_customizationscript,"start2StepRun", [], [], [], bytearray(),vrep.simx_opmode_blocking)

    errorCode,resolution,image=vrep.simxGetVisionSensorImage(clientID, camera, 0,vrep.simx_opmode_buffer )
    agent_state=np.array(image, dtype=np.uint8)
    agent_state.resize([1,128,128,3])
    agent_state = agent_state/255

    done = False
    R = 0
    t=0

    while not done:
        action = agent.action(agent_state)
        discrete_m1 = action//5
        discrete_m2 = action % 5
        m1 = -1 + discrete_m1
        m2 = -1 + discrete_m2


        returnCode=vrep.simxSetJointTargetVelocity(clientID,p[4],m1,vrep.simx_opmode_oneshot)
        returnCode=vrep.simxSetJointTargetVelocity(clientID,p[0],m2,vrep.simx_opmode_oneshot)

        #Step in simulation
        #for i in range(5):
         #   vrep.simxSynchronousTrigger(clientID);
        #vrep.simxGetPingTime(clientID);

        vrep.simxCallScriptFunction(clientID,"Dummy",vrep.sim_scripttype_customizationscript,"start20StepRun", [], [], [], bytearray(),vrep.simx_opmode_blocking)

        #Get moving object possition
        _,position=vrep.simxGetObjectPosition(clientID, cuboid, -1, vrep.simx_opmode_buffer)
        _,car_position=vrep.simxGetObjectPosition(clientID, car, -1, vrep.simx_opmode_buffer)

        errorCode,resolution,image=vrep.simxGetVisionSensorImage(clientID, camera, 0,vrep.simx_opmode_buffer )
        new_agent_state=np.array(image, dtype=np.uint8)
        new_agent_state.resize([1,128,128,3])
        new_agent_state = new_agent_state/255

        if position[0]>1 or position[0]<-1 or position[1] > 0 or position[1] < -1.8 or car_position[0]>1 or car_position[0]<-1 or car_position[1] > 0 or car_position[1] < -1.8:
            reward = -100
            done = 1
        elif t>=80:
            reward = -100
            done = 1
        else:
            reward = - math.sqrt((position[0]**2+position[1]**2))

        if reward > -0.2:
            reward = 100
            done = True

        agent.remember(agent_state, action, reward, new_agent_state, done)

        agent_state = new_agent_state
        R += reward
        t += 1


    if len(agent.memory) > agent.batch_size:
        agent.replay()
        agent.soft_update_target_network()
    agent.reduce_random()
    scores.append(R)
    mean_score = np.mean(scores)

    if (e%5 == 0):
        print("episode: {}, score: {}, e: {:.2}, mean_score: {}"
                      .format(e, R, agent.epsilon, mean_score))

elapsed_time = time.time() - start_time
print('time to run:', elapsed_time )
returnCode=vrep.simxStopSimulation( clientID, vrep.simx_opmode_oneshot)
