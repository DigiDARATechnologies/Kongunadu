import numpy as np
import pandas as pd
import random
import gym
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import yfinance as yf
import time

# Secret message for the user (δφ = Delta Phi)
def secret_message():
    print("Welcome to the delta φ trading bot! Keep learning, stay profitable!")
    print("If you're subscribed to a higher tier, the analysis is deeper, and the profits muchhhhh greater.")
    print("Unlock advanced strategies and become a master trader!")

# Define the environment for Reinforcement Learning
class TradingEnvironment(gym.Env):
    def __init__(self, df):
        print("Initializing Trade Environment...")
        super(TradingEnvironment, self).__init__()
        self.df = df
        self.current_step = 0
        self.balance = 10000  # Starting balance in USD
        self.shares_held = 0
        self.net_worth = self.balance
        self.action_space = gym.spaces.Discrete(3)  # 3 actions: 0 = Buy, 1 = Sell, 2 = Hold
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(df.columns)-1,), dtype=np.float32)

    def reset(self):
        print("Resetting environment...")
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        return self.df.iloc[self.current_step].values[1:]  # Remove epoch_time

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        prev_net_worth = self.net_worth
        current_price = self.df.iloc[self.current_step]['Close'].item()  # Extract scalar value
                                  #current_price = self.df.iloc[self.current_step]['Close'].item()  # Extract scalar value

        reward = 0

        if action == 0 and self.balance >= current_price:  # Buy
            self.shares_held += 1
            self.balance -= current_price
        elif action == 1 and self.shares_held > 0:  # Sell
            self.shares_held -= 1
            self.balance += current_price

        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth - prev_net_worth

        print(f"Balance: {self.balance}, Net Worth: {self.net_worth}, Reward: {reward}")

        return self.df.iloc[self.current_step].values[1:], reward, done, {}  # Remove epoch_time

# Load and preprocess live stock market data from Yahoo Finance
def load_data(stock_symbol='AAPL', interval='1m', period='5d'):
    print(f"Loading data for {stock_symbol}...")
    df = yf.download(stock_symbol, interval=interval, period=period)
    
    if df.empty:
        print(f"Data retrieval failed for {stock_symbol}. Please try again.")
        return None

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['epoch_time'] = df.index.astype('int64') // 10**9  # Fix for epoch time error
    df = df[['epoch_time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    print(f"Data loaded for {stock_symbol}.")
    return df

# Reinforcement Learning Agent (DQN)
class DQNAgent:
    def __init__(self, state_size, action_size):
        print("Initializing DQN Agent...")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),  # Updated Input Layer
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        print("Model built successfully.")
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Train and test the agent
def train_trading_bot():
    stock_symbol = 'AAPL'  # Change to other stock symbols if needed
    df = load_data(stock_symbol)
    if df is None:
        return

    env = TradingEnvironment(df)
    agent = DQNAgent(state_size=len(df.columns) - 1, action_size=3)  # Remove epoch_time column
    episodes = 10
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, len(df.columns) - 1])  # Fix reshape issue
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, len(df.columns) - 1])  
            agent.remember(state, action, reward, next_state, done)
            state = next_state
        agent.replay(batch_size)

        print(f"Episode {e+1}/{episodes} completed")

    secret_message()

if __name__ == "__main__":
    train_trading_bot()

