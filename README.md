# MPI Distributed Deep Q-Learning

Implementation of the Reinforcement Learning Deep Q-Learning algorithm based on Message Passing Interface (MPI).

In this implementation, one process computes the training of the Deep Neural Network, while the rest of the processes run simulations to generate more data. Usually, the computation of simulations require to solve complex and computationally expensive problems. With this approach the runtimes to train controllers are reduced.

The implementation has been applied to three different problems to control, increasing their complexity until having a high complex simulation.

A study of the results is found [here](documentation/document.pdf)

### Inverted pendulum

The control problem consists of an inverted pendulum that has to be balanced by moving left or right. The environment is from [Gym library](https://www.gymlibrary.ml/environments/classic_control/cart_pole/).

![Cart Pole](documentation/cart_pole.gif "Cart Pole")


### Lunar landing

Consists on landing a spaceship, the control is done on the motors and the spaceship has to be properly landed in a determined place. The environment is from [Gym library](https://www.gymlibrary.ml/environments/box2d/lunar_lander/).

![Lunar Lander](documentation/lunar_lander.gif "Lunar Lander")

### Box slider

This environment has been created using the software [VRep](https://www.coppeliarobotics.com/). It consists on an Unmanned Ground Vehicle (UGV) that has to move a box to a given position. The information that the controller gets is the image from a camera that is placed over the robot.



