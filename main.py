import tensorflow as tf
import numpy as np
import pandas as pd
from collections import deque
import os

from agent import PipeSpoolFabricationEnv

# Define the DQN model
class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(128, activation="relu")
        self.dense3 = tf.keras.layers.Dense(64, activation="relu")
        self.dense4 = tf.keras.layers.Dense(num_actions, dtype=tf.float32)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)

# Define the Prioritized Replay Buffer
class PrioritizedReplayBuffer(object):
    def __init__(self, size, alpha=0.6, beta=0.4):
        self.buffer = deque(maxlen=size)
        self.priorities = deque(maxlen=size)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)

    def update_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            self.priorities[i] = priority
        self.max_priority = max(self.priorities)

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= np.sum(probabilities)

        indices = np.random.choice(len(self.buffer), num_samples, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)

        states, actions, rewards, next_states, dones = zip(*samples)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        weights = np.array(weights, dtype=np.float32)
        indices = np.array(indices, dtype=np.int32)

        return states, actions, rewards, next_states, dones, weights, indices

# Function to select an action using epsilon-greedy policy
def select_epsilon_greedy_action(state, epsilon, num_actions, main_nn):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)  # Random action
    else:
        q_values = main_nn(state)
        return tf.argmax(q_values[0]).numpy()

# Function to perform a training iteration
def train_step(states, actions, rewards, next_states, dones, weights, indices, target_nn, main_nn, discount, optimizer,
               loss_fn):
    with tf.GradientTape() as tape:
        next_q_values = target_nn(next_states)
        max_next_q_values = tf.reduce_max(next_q_values, axis=-1)
        targets = rewards + (1.0 - dones) * discount * max_next_q_values

        q_values = main_nn(states)
        action_masks = tf.one_hot(actions, num_actions, axis=-1)
        masked_q_values = tf.reduce_sum(action_masks * q_values, axis=-1)

        td_errors = targets - masked_q_values
        priorities = np.abs(td_errors) + 1e-6
        replay_buffer.update_priorities(indices, priorities)

        loss = tf.reduce_mean(weights * loss_fn(targets, masked_q_values))

    grads = tape.gradient(loss, main_nn.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))

    return loss

# Set the hyperparameters
num_episodes = 5000
epsilon = 0.1
epsilon_decay = 0.001
min_epsilon = 0.01
batch_size = 64
discount = 0.99
target_update_frequency = 1000
replay_buffer_size = 100000

# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_dir, 'Test2.csv')
df = pd.read_csv(csv_file_path, parse_dates=["Start Date", "Due Date"])
spool_id = df["SpoolID"]
start_dates = df["Start Date"]
processing_times = df["Processing Time in Days"]
due_dates = df["Due Date"]
required_resources = df["Required Resources"]

# Create the environment
env = PipeSpoolFabricationEnv(spool_id, start_dates, processing_times, due_dates, required_resources)

# Get the dimensions of the action and state spaces
num_actions = env.action_space.n
num_features = env.observation_space.shape[0]

# Create the Q-networks
main_nn = DQN(num_actions)
target_nn = DQN(num_actions)
target_nn.set_weights(main_nn.get_weights())

# Create the replay buffer
replay_buffer = PrioritizedReplayBuffer(replay_buffer_size)

# Create the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.Huber()

# Initialize variables
episode_rewards = []
epsilon_values = []

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        state_tensor = tf.expand_dims(state, axis=0)
        action = select_epsilon_greedy_action(state_tensor, epsilon, num_actions, main_nn)
        first_selection = True

        next_state, reward, done, _ = env.step(action, first_selection)

        replay_buffer.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(batch_size)
            train_step(states, actions, rewards, next_states, dones, weights, indices, target_nn, main_nn, discount,
                       optimizer, loss_fn)

        if episode % target_update_frequency == 0:
            target_nn.set_weights(main_nn.get_weights())

    epsilon = max(epsilon - epsilon_decay, min_epsilon)

    episode_rewards.append(episode_reward)
    epsilon_values.append(epsilon)

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode: {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.1f}, Epsilon: {epsilon:.2f}")

# Save the trained model
main_nn.save_weights("trained_model.h5")