import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import argparse
from tqdm import tqdm

TRAIN_EPISODES = 15000
RUN_EPISODES = 1000

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('-r',
                   '--run',
                   action="store_true",
                   help="執行已訓練好的模型")
group.add_argument('-t',
                   '--train',
                   action="store_true",
                   help="利用超參數訓練模型")
group.add_argument('-tu',
                   '--tune',
                   action="store_true",
                   help="自動尋找最佳參數組合")

def print_success_rate(rewards_per_episode):
    """Calculate and print the success rate of the agent."""
    total_episodes = len(rewards_per_episode)
    success_count = np.sum(rewards_per_episode)
    success_rate = (success_count / total_episodes) * 100
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate

def find_best_combination():
    "自動的尋找最佳參數組合"
    best_late_success_rate = 0
    best_combination = (0,0,0)

    decay_rate = np.arange(0.00008, 0.0004, 0.00001)
    min_epsilon_values = np.arange(0, 0.00025, 0.00001)
    learning_rates = np.arange(0.5, 0.9, 0.05)

    for epsilon_decay_rate in decay_rate:
        for min_epsilon in min_epsilon_values:
            for learning_rate in learning_rates:
                print(f"Testing combination: epsilon_decay_rate={epsilon_decay_rate}, min_epsilon={min_epsilon}, learning_rate={learning_rate}")
                late_success_rate , q_table = run(15000, is_training=True, render=False, epsilon_decay_rate=epsilon_decay_rate, min_epsilon=min_epsilon, learning_rate_a=learning_rate)
                if late_success_rate > best_late_success_rate:
                    best_late_success_rate = late_success_rate
                    best_combination = (epsilon_decay_rate, min_epsilon, learning_rate)

                    print(f"New best combination found: epsilon_decay_rate={epsilon_decay_rate}, min_epsilon={min_epsilon}, learning_rate={learning_rate} with Success Rate: {best_late_success_rate:.2f}%")
                    f = open("frozen_lake8x8.pkl","wb")
                    pickle.dump(q_table, f)
                    f.close()
    print(f"Best combination: epsilon_decay_rate={best_combination[0]:.5f}, min_epsilon={best_combination[1]:.5f}, learning_rate={best_combination[2]:.2f}  with Success Rate: {best_late_success_rate:.2f}%")

def run(episodes, is_training=True, render=False , epsilon_decay_rate=0.0001 , min_epsilon=0.0001, learning_rate_a=0.55):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='ansi' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    # learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in tqdm(range(episodes), desc="Episodes", leave=False):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = q[state,:].argmax()

            new_state,reward,terminated,truncated,_ = env.step(action)
            #print(env.render())
            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * q[new_state, :].max() - q[state,action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')
    
    if is_training == False:
        print(print_success_rate(rewards_per_episode))
               
    if is_training:
        last_n_episodes = 1000
        if episodes < last_n_episodes:
            last_n_episodes = episodes
            
        late_success_rate = np.sum(rewards_per_episode[-last_n_episodes:]) / last_n_episodes * 100
        print(f"Late Stage Success Rate (Last {last_n_episodes}): {late_success_rate:.2f}%")
        return late_success_rate , q

if __name__ == '__main__':
    
    # Get and parse the arguments
    args = parser.parse_args()

    if args.run:
        run(RUN_EPISODES, is_training=False, render=False)
    elif args.train:
        run(TRAIN_EPISODES, is_training=True, render=False)
    elif args.tune:
        find_best_combination()
