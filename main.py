from environment import Environment
from agent import Agent
from log import Log

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def learn(env, agent, num):
    reward = 0
    t = 0

    state = env.states[t]
    price = env.return_hist[t]

    previous_action = np.array(
        [[1]+[0 for i in range(len(env.config_csv['codes']))]])

    predicted_action = []

    # print(f'*********{len(env.states)}')

    while True:
        tmp = predicted_action
        # print(f'state\n{state}\n')
        # print(f'previous action\n{previous_action}\n')
        # print(state)

        predicted_action = agent.predict(state, previous_action, price)
        # print(f'predicted action: {predicted_action}')
        # print(f'predicted action\n{predicted_action}\n')

        env.buffer.append((state, price, predicted_action, previous_action))

        if t + 1 == len(env.states):
            break
        t = t + 1
        state = env.states[t]
        price = env.return_hist[t]
        previous_action = predicted_action

    if env.config_mode == 'train':
        agent.train(env.buffer, num)
    elif env.config_mode == 'test':
        agent.test(env.buffer)
    env.reset_buffer()


def main():

    count = 0
    mode = 'train'

    print(f'===================Start===================')

    env = Environment(mode)
    agent = Agent(env.config)
    env.prep_data()

    saveat = 50

    while True:
        if count % saveat == 1:
            mode = 'test'
            env = Environment(mode)
            env.config_model["continue"] = True
            env.config_model["save_hist"] = True
            agent.close_sess()
            agent = Agent(env.config)
            env.prep_data()
            print(f'------------------test----------------')
        elif count % saveat == 2:
            mode = 'train'
            env = Environment(mode)
            env.config_model["continue"] = True
            env.config_model["save_hist"] = True
            agent.close_sess()
            agent = Agent(env.config)
            env.prep_data()
        else:
            env.config_model["save_hist"] = False

        # print(f'===================Start===================')
        for i in range(env.config_model["epoch"]):
            print(f'round {count}')
            learn(env, agent, i)
        # logs = Log()
        # print(f'====================Done===================')

        count += 1

    print(f'====================Done===================')


if __name__ == "__main__":
    main()
