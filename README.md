# PPO_Cartpole 
Proximal Policy optimisation (PPO) algorithm is an improvement on A2C algorithm. In A2C algorithm the downside was the Advantage function and the step size. We push our agents to get better rewards and stop it from doing bad. If the step size is too small then training was very slow whereas if the step size was very large then there was very high variance. To fix this problem PPO came up with a surrogate loss function. In this new loss function we clip the advantage update not to go beyond a certain value. We always make sure that the new policy is within limits of the old policy, this taken care by KL divergence. Thus in this way we never make our model believe that one particular policy is very good or very bad.
# Implementation:
Here we have 3 fully connected layers for policy network and 2 fully connected layers for value network. This is implemented in the polnet.py file. In the ppo_.py file the main working of PPO algorithm is implemented. After assign input operations, we first calculate the new policy and then clip the new policy with old policy. Next we calculate value loss (the critic function) and entropy loss. Entropy is the factor which tells us about the exploration and exploitation strategy of our agent. Finally our agent is trained with Adam Optimizer.
In the main.py file, the PPO agent is interacted with the environment and the main loop is run in main.py.
The environment used is Cart Pole version 1. In this environment the max reward that can be achieved is 500.
# Results:
In this environment the pole has to be balanced on a cart. The amount of time you can balance that much of reward. We can see that in 1.4K steps the agent was able to achieve the maximum reward.
![alt-tex](https://github.com/aveen-d/PPO_Cartpole/blob/master/results/results.png)
Here we can see the loss and value loss decreasing over time and so is the clipped loss value. We also see the entropy loss initially more which means the agent is exploring and then decreasing that the agent started finding optimal solution and thus started exploiting.
![alt-text](https://github.com/aveen-d/PPO_Cartpole/blob/master/results/results2.png)
