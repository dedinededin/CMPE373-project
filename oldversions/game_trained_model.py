import gym
import random
import numpy as np

from collections import deque
from collections import Counter

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
env.reset()
episodes = 100
LR = 1e-3

score_requirement = 150  # our threshold
scores = []
accepted_scores = []  # scores that met our threshold
scores_window = deque(maxlen=30)

y = 0.95
eps = 0.5
eps_min = 0.01
eps_decay = 0.995
decay_factor = 0.999
batch_size = 16
memory = deque(maxlen=1000)

print("State/Observation space:", env.observation_space.shape)
print("Number of actions:", env.action_space)


def generate_model():
    # neural network for deep q learning
    model = Sequential()
    model.add(Dense(48, input_dim=env.observation_space.shape[0], activation="tanh"))
    model.add(Dense(env.action_space.n, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=LR))
    return model


model = generate_model()

from keras.utils import plot_model

plot_model(model, to_file='model.png', show_shapes=True)

for episode in range(episodes):
    s = env.reset()
    s = np.reshape(s, [1, env.observation_space.shape[0]])
    score = 0
    # eps *=decay_factor

    while True:

        game_memory = []
        prev_obs = []
        # if len(prev_obs)==0:
        if random.uniform(0, 1) <= eps:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(s)[0])

        if eps > eps_min:
            eps *= eps_decay

        new_observation, reward, done, info = env.step(action)
        new_observation = np.reshape(new_observation, [1, env.observation_space.shape[0]])
        # new_observation=new_observation.reshape(-1,len(new_observation),1)

        prev_obs = new_observation
        score += reward

        # In case we want to use another model
        # we may want to use this one-hot encoding method,
        # this is not currently used.

        # training_data = []
        # if score >= score_requirement:
        #     accepted_scores.append(score)
        #     for data in game_memory:
        #         # convert to one-hot (this is the output layer for our neural network)
        #         if data[1] == 1:
        #             output = [0, 1]
        #         elif data[1] == 0:
        #             output = [1, 0]
        #
        #         # saving our training data
        #         training_data.append([data[0], output])

        scores_window.append(score)

        if score >= score_requirement:
            accepted_scores.append(score)

        memory.append((s, action, reward, new_observation, done))

        s = new_observation
        # env.render()
        if len(memory) >= batch_size:
            minibatch = random.sample(memory, batch_size)
            for state, action, reward, next_state, isDone in minibatch:
                if isDone:
                    target = reward
                else:
                    target = reward + y * np.amax(model.predict(next_state)[0])
                train_target = model.predict(state)
                train_target[0][action] = target  ## learning
                model.fit(state, train_target, verbose=0)

        if done:
            print("Episode number :{} is done. Score:{}".format(episode, score))
            break
    scores.append(score)

# just in case you wanted to reference later,
# our training_data that we already converted one-hot
# training_data_save = np.array(training_data)
# np.save('saved.npy',training_data_save)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

print()
print("Accepted scores that met our threshold {}".format(Counter(accepted_scores)))
print('average score:', sum(scores) / len(scores))
