# Project Title:

Deep Reinforcement Learning with OpenAI Gym (CartPole-v0)  

# Students:

Hanifi Enes Gül 115200085

Atakan Kaya 115200074

İbrahim Doğan 11512112

  
# Dependencies
gym==0.12.1
Keras==2.2.4
matplotlib==3.0.3
numpy==1.16.3
tensorflow==1.13.1
  

# Abstract

This Project is a Deep Q-Reinforcement Learning Project. the purpose of this project: without taking any human data, only actions that can be taken and the target specified model; to ensure that it reaches the target successfully. In this project, to ensure that the CartPole-v0 game selected from the OpenAI (GYM) library is successfully played in a way to achieve high scores. Simply we are building an autonomous agent that can successfully play the game.

  

# Introduction

There are three different sub-area of Machine Learning, which are Supervised Learning, Unsupervised Learning and Reinforcement Learning (RL). Reinforcement Learning technique will be implemented in this project. (Reinforcement Learning is the most academic paper written category in Machine Learning [2018]).

In this project, Reinforcement Learning methodology will be used with OpenAI Gym toolkit and Keras. Combination of RL and Neural Networks is called Deep Reinforcement Learning.

  

# Environment

We create an agent which performs “actions” in an “environment” and the agent receives various “rewards” depending on what “state” it is in when it performs the action.

As it is already mentioned above, the game that selected from OpenAI Gym Toolkit is CartPole-v0. This game will be considered as our “environment”.

![](https://lh6.googleusercontent.com/RYNQqU2HPPF8N765zDev8jgdouUzW47UYup7-pURxXiuei41A0EXOzt0zkEY65mPMzPMfJxTbIccDqybSKnZMKEyoFEWmHN5o_RCvKVBhlVJSx8VmeKSnL1_XNIwXAX4ZyrPB9OOOSvBfiePkA)

Environment (CartPole-v0) comes with its own rules and actions, observations etc. which are shown in next page in detail.

  
![cartpole ile ilgili görsel sonucu](https://lh3.googleusercontent.com/CizYrd298-X9l8lrndwHewshdFG2b7iJSf-cPfquVf4HOCaK5YgK1SPuSnGAA12is-t5LWE815iCgSYQ9HzIdPm3V79-sK96LIGiBLbzYPu3cPCBov-soNI2hWTJ9q4F1gTScdMu9vHBFsu2uA)

# Documentation of Environment

![](https://lh5.googleusercontent.com/nbEIET8XH3wqOUc_BcZjgZJRFAvrQm_nVd1WCnZd4jIfcyv7931n43SfRoxtizqEKztmw-rLNU9ue6O6lD0bDIQvhs-e0Zwwo6WA-I2C7aNu5-OJwV8F1l-T7yZ8DNkeG3hUfBbX-HzUIRpZpQ)

![](https://lh3.googleusercontent.com/hxWAjSvwvdeARJTuwg5odbAM7830HPAsRSQz1UMW6o_AfEdMnZmV_oA1vO-bJEtKAPWsDRvrwnib2i8cj8yxvP7fayvbFDuNGPhYcrhGQYhbvdKRcYpFN6Mh8J3-ei0Fph_S2vUcFHLeusnRPA)

![](https://lh3.googleusercontent.com/fnNePIrY8ljA3Bq9Wt7fAw1w6kxB6ZGhvNaIO9gQZWORHfwijvo8NG8BpRNtT-_1zvF8IQrgqcHbVqMsWCpGR5hdC1wxTCd5pSmWn6XElPiFzhUjoczamQym0VHLoxZ38aVxJH86mfXzLE8xHg)

![](https://lh4.googleusercontent.com/Cr30u-u6LGEKXzg3lmRKJ9-CpZkcBVETa19ySHX19Q-lELkyfV46mBzAxg7NPeCz9Gr_tvCGBUOsHM4_MawcthiX1_Cw5mrBmrlEDIbmS1C5hYZc0BUKqD_wq9fcPdKhnUbbgKLh7jN58nspQw)

![](https://lh4.googleusercontent.com/d9pEVjYlE426BGlCdp3TR_O8Idl4vFV6vkuuyy6L7QeJIQw6q6-ArHGdACWQ2a9lkpf0a28ME4riSVVvVu-u1n3vU7AMc1Ee99cxh3DY6E8j7GLOudycBlitWOk_00YSw5T1nHRVPbC6xm_9sQ)

Source: [https://github.com/openai/gym/wiki/CartPole-v0](https://github.com/openai/gym/wiki/CartPole-v0)

  
  

# Reinforcement Learning

As can be seen below picture, the agent plays out some activity in the environment. A mediator sees this action in the environment, and feeds back a refreshed state that the agent currently lives in, and furthermore the reward for making this move. The environment isn't known by the agent previously, but instead it is found by the agent making steady strides in time. In this way, for example, at time t the agent, in state st, may make a move a. This outcomes in another state st+1 and a reward r. This reward can be a positive real number, zero, or a negative real number. (In this environment negative real number reward is not included.) It is the objective of the agent to realize which state subordinate action to take which boosts its rewards.![](https://lh3.googleusercontent.com/d5ox3TQqnT_gxy49vs4RG07-cKQn2Df7324CW_AbXd-nTnLoo891FZUcZjrxo22sHVHhGF7k4z3hD2fKWq7Ygyo3W_BZNZF_StxWgEmZ1V_8s0pRiG6DFCkhe0PMCFd8pKJ7ct1LjVTOM0v3Zg)

  

# Keras + Reinforcement Learning

As it is mentioned, one of the aim is creating neural network that can perform Q-Learning. Current state is a must as an input. Q values will bi created for each action in that specific state with respect to Bellman Equation.

![](https://lh5.googleusercontent.com/pp_FzNg9lj8gcL8vloiIdmSKpVfNnt08IbIWuYa0h391Qaot8VOmNz6cpitCbEoM1m6M8oD_cUA9s-k0PSzsfsArNAfeqfm-qS9iZWtZLoM6PYjbXCTX7Uo-iBp9rvqAa5fZrnnqb3tE76_bXA)

Our implementation is the highlighted part below:

![](https://lh3.googleusercontent.com/x6hewoINMZV0dVBw3YPGuAo4RjweQWq3DT_v1pg_cTIIT8z4CfGzxQbu68AObXe3od_MbLFY-iNIHhq7AVD28_1fLx8n81srEcZtiQRG4gaOtinjkxfNvP3lQaCaU6JXSlQhmGnZ2CYKk8ZNrQ)

  
  

We build our model with this code block:

![](https://lh4.googleusercontent.com/MSgB5NGpZag4Qq_Aw0jHI9xD45DpnnxPNOu8P_EZo9zxmD9Mux9Lg2vh1Jp1jn49Ihzze9MYaSdkpfaCfUG7LXQW7a0kok4G-NjYLvSpILo0iyEaBTJihP7ALDwtZCvzQcXfQBa83poIWOQ1zg)

  

Keras has built-in model.png generator:

![](https://lh4.googleusercontent.com/4dxzPwQcE7hgc4BPBAI31oGEkdJoUQ8yRNSfJ8y4HK81Kvps5esr7YVWXVuEtysP_82uk54MAcSoBxPryWg3S99waWPvthuqBhyniEaKNYxXW384GMIA_J050sOMecDZ1a0US1KosXgdRyxWdw)

When we run this the output is:![](https://lh5.googleusercontent.com/NDjvPNI2hDPmqUu-jXqGKaCrEYEGwY2-HqG2gbh1P-aVUHx65A4djp81_JvcficZuoHVLzG90HCzduDlpJeg9HSQ_96RgEOLZxDnU-nH3zn-hYuIFDIY-ZdtzElK9UuqLATjwqInjn4pdAeDIQ)

  
  
  
  
  
  
  
  
  

To make it more understandable we illustrated it as:

  
![](https://lh3.googleusercontent.com/cmhZRGZ4j4fwmAUX8C1JDDQdr3Z3EXgNz0YJAMzRYP-_e083zWp6li_zy2XKQ4uNJwB2K-bhSZ-MTfAn8CV7t3YQ7kCD4D17liXosig26aI945qgMNgsIp0bNOfXOjRFfTTS6Bc8g6MfWovxqA)

# The main part:
-   Each episode runs until the environment sets the done=True.
    
-   We keep score values to be able to see the success of episode.
    
-   After applying epsilongreedy method we take step from environment and get reward.
    
-   We add reward to score.
    
-   We use memory to feed model regard to randomness with batchsize.
    
-   Q-Learning part with bellman equation applied and fitted to model.
    
-   Lastly, to check the accuracy of the agent we printed and plotted some values.
    
  
  

# Plots and Prints

As we can see in the graph, when episode number increases our agent’s success also increased.

![](https://lh4.googleusercontent.com/XRCsGf6C6oW-uuLJ9ysW_ERYplFnOFi6LvipxMBSiv--b4UZ1aeJi73qjCVhWe9zVxrjrWCoV6TT1yGyMQn9Lmh5IMA3o1hMiQKwdul1Gojmg7EgDEZlm0psDfzC9AFFvLeZhFrRLv7xjv8MHQ)

![](https://lh4.googleusercontent.com/_T7i4W-JXHLAE7KOL_l_VtEzig8VRCKhAOhMz9SP2g7rs0whgmOjzx7vzAa153txWCkIObcB0B_rqFqmHccMxAERAgWh99rMNa-ru8D4fplTtG1tMgTaWXQtNv41Vngc-NfHVVGw2kYFuz_iTA)

  

# Discussion

We tried to implement deep reinforcement learning using Keras to OpenAI’s Gym/CartPole-v0 environment. With doing some dynamic changes to code it can also work on other environments such as MountainCar-v0 etc. We commented one-hot encoding part of the code, with using correct sizes other neural networks can be also implemented such as TFLearn, DNN.

We get success in early steps nearly 40th episode. It can be improved, but it is quite successful.
