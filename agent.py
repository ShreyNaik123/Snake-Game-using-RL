import torch
import random 
import numpy as np
from snake_game import SnakeGameAI, Direction, Point
from collections import deque
from model import LinearQNet, QTrainer
from plotter import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
  
  def __init__(self):
     self.no_games = 0
     self.epsilon = 1
     self.gamma = 0.9
     self.memory = deque(maxlen=MAX_MEMORY)
     self.model = LinearQNet(11,256,3)
     self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
     
  
  def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)
  
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
  
  def train_long(self):
      if len(self.memory) > BATCH_SIZE:
          random_sample = random.sample(self.memory, BATCH_SIZE)
      else:
          random_sample = self.memory
      
      
      states, actions, rewards, next_states, dones = zip(*random_sample)
      self.trainer.train_step(states, actions, rewards, next_states, dones)
  
  def train_short(self, state, action, reward, next_state, done):
      self.trainer.train_step(state, action, reward, next_state, done)
  
  def get_action(self, state):
      # exploration and exploitation
      self.epsilon = 80 - self.no_games
      move = [0,0,0]
      if random.randint(0,200) < self.epsilon:
        idx = random.randint(0,2)
        move[idx] = 1
      else:
        initial_state = torch.tensor(state, dtype=torch.float)
        pred = self.model(initial_state)
        idx = torch.argmax(pred).item()
        move[idx] = 1
      
      return move

def train():
  scores = []
  mean_scores = []
  total_score = 0
  max_score = 0
  agent = Agent()
  game = SnakeGameAI()
  
  # train loop
  while True:
    old_state = agent.get_state(game)
    
    move = agent.get_action(old_state)
    
    reward,done,score = game.play_step(move)
    
    new_state = agent.get_state(game)
    
    agent.train_short(old_state, move, reward, new_state, done)
    
    agent.remember(old_state, move, reward, new_state, done)
    
    if done:
      #experience replay
      game.reset()
      agent.no_games+=1
      agent.train_long()

      if score > max_score:
        max_score = score
        agent.model.save()

      print(f"Games Played : {agent.no_games}, Score = {score}, Max Score = {max_score}")
      
      scores.append(score)
      total_score += score
      mean_scores.append(total_score/agent.no_games)
      plot(scores, mean_scores)

  
if __name__=="__main__":
  train()


