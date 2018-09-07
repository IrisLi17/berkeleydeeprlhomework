import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
import os
import gym
import plot_history


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--render', type=int)
    args = parser.parse_args()
    with open(os.path.join('expert_data', args.env + ".pkl"), 'rb') as f:
        expert_data = pickle.loads(f.read())
    expert_actions = expert_data['actions']
    if len(expert_actions.shape) ==3:
        expert_actions = np.reshape(expert_actions, (expert_actions.shape[0], expert_actions.shape[2]))
    env = gym.make(args.env)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    model = keras.Sequential([
        # keras.layers.Convolution2D(32, 8, 4, input_shape=expert_data['observations'][0].shape, activation=tf.nn.relu),
        # keras.layers.Convolution2D(64, 4, 2, activation=tf.nn.relu),
        # keras.layers.Convolution2D(64, 3, 1, activation=tf.nn.relu),
        # keras.layers.Flatten()
        keras.layers.Dense(64, input_shape=(ob_dim,), activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(ac_dim)
    ])
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='mse',
                  metrics=['mae'])
    model.summary()
    history = model.fit(expert_data['observations'], expert_actions, epochs=90)
    plot_history.plot_history(history)

    returns = []
    for iter in range(30):
        obs = env.reset()
        done = False
        totalr = 0
        print("iter "+ str(iter))
        while not done:
            act = model.predict(obs[None,:])
            obs, r, done, _  = env.step(act)
            if bool(args.render):
                env.render()
            totalr += r
        print("totalr "+ str(totalr))
        returns.append(totalr)
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__== "__main__":
    main()