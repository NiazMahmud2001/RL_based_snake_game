import random
import argparse
import time
import pygame 


from collections import Counter, deque
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical


observation_dim = 12 # # numbers the AI sees each frame = input size of the network
# (heading x4, food position x4, body distance rays x3, snake length x1)

N_ACTIONS = 3 # (0 straight, 1 left, 2 right)
N_ENVS = 16  # 16 snakes playing at thge same time 

T = 256 # 256 frame played by model before collect and train the model 


SCREEN_W = 1200
SCREEN_H = 600
STEP = 5 # snake_dx / snake_dy -> snake moves per frame in px

SNAKE_RADIUS =9 
FOOD_SIZE = 6
GROW_PER_FOOD = 3
EAT_DIST = SNAKE_RADIUS + FOOD_SIZE / 2
DIRECTIONS = [(5, 0), (0, 5), (-5, 0), (0, -5)]
RAY_STEPS = 40 # the AI "sees" its own body up to 40 x 5 = 200 px away
EAT_DIST = SNAKE_RADIUS + (FOOD_SIZE / 2) # from this distance the food will be eaten
blue = (78, 192, 202)

# model vars
GAMMA = 0.99  # snakes are starving -- penalty
LAMBDA = 0.95 # GAE smoothing

COLORS = [
    (40, 170, 70), (220, 60, 60), (60, 90, 220), (230, 160, 30), 
    (150, 60, 200), (20, 150, 160),(200, 80, 160), (110, 110, 110), 
    (140, 200, 40), (240, 110, 60), (80, 160, 240), (170, 120, 70),
    (0, 120, 80), (180, 40, 100), (90, 60, 150), (230, 210, 60)
]  




def torus_delta(d, size):
    period = size + STEP
    half = period / 2
    return (d + half) % period - half

class SnakeEnv: 
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()


    def reset(self): 
        self.x = 300
        self.y = 200

        self.dir = 0 # movie dir : [right:0, down:1, left:2, up:3]
        self.parts = deque([(self.x, self.y)])
        self.occupied = Counter(self.parts)

        self.grow = 0
        self.score = 0

        self.frames = 0
        self.since_food = 0

        self.food_x = self.rng.randint(40, SCREEN_W - 40)
        self.food_y = self.rng.randint(40, SCREEN_H - 40)

        self.prev_dist = self._food_dist()
        return self.observe()
        

    def _food_dist(self):
            return abs(torus_delta(self.food_x - self.x, SCREEN_W)) + abs(torus_delta(self.food_y - self.y, SCREEN_H))

    
    def steps(self, action_idx_by_policy): # 0 ->straight || 1-> left || 2->right
        # One frame. Returns (observation ->12 item list, reward, done, info)
        if action_idx_by_policy == 1:
            self.dir = (self.dir - 1) % 4
        elif action_idx_by_policy == 2:
            self.dir = (self.dir + 1) % 4

        dx, dy = DIRECTIONS[self.dir]

        self.x = self.x + dx
        self.y = self.y + dy 
        if (self.x + dx) > SCREEN_W:
            self.x = 0
        if (self.x + dx) < 0:
            self.x = SCREEN_W

        if (self.y + dy) > SCREEN_H:
            self.y = 0
        if (self.y + dy) < 0:
            self.y = SCREEN_H


        self.frames += 1
        self.since_food += 1

        # self-bite: the head is on top of a body part
        if self.occupied[self.parts[0]] > 1:
            return self.observe(), -1.0, True, {"score": self.score, "reason": "bit itself"}
            # observation , reward , done , info-> {score and reason}

        
        reward = 0.0
        if abs(self.x - self.food_x) <= EAT_DIST and abs(self.y - self.food_y) <= EAT_DIST:
            self.score += 1
            self.grow += GROW_PER_FOOD
            self.since_food = 0

            # food eaten ... now generate new food
            self.food_x = self.rng.randint(40, SCREEN_W - 40)
            self.food_y = self.rng.randint(40, SCREEN_H - 40)
            reward = 1.0
        else: 
            dist = self._food_dist()
            reward = 0.01 * (self.prev_dist - dist) / STEP 
            # if snakes goes closer to food than prev distance then give a small reward
        self.prev_dist = self._food_dist()

        self.parts.appendleft((self.x, self.y))
        self.occupied[(self.x, self.y)] += 1
        if self.grow > 0:
            self.grow -= 1
        else:
            tail = self.parts.pop()
            self.occupied[tail] -= 1

        if self.since_food >= 1500: # if snake take more than 1500 frames to eat food then dies autometically
            return self.observe(), 0.0, True, {"score": self.score, "reason": "starved", "timeout": True}
        return self.observe(), reward, False, {}


    def observe(self):
        # it generate 12 dimention per snake: that why Policy and Value model takes 12 col data
        """What the AI sees each frame: 12 numbers (= OBS_DIM).
        index  value          meaning
        0-3    heading        one-hot direction: [right, down, left, up], e.g. moving right = [1, 0, 0, 0]
        4      food_dx        food left/right of the head, -1..1 (shortest way, can go through the edge)
        5      food_dy        food above/below the head, -1..1
        6      ahead          food in front of (+) or behind (-) the snake, from the snake's point of view
        7      right          food to the snake's right (+) or left (-)
        8      ray straight   free way straight ahead before hitting its own body, 0..1 (1 = free for 200 px)
        9      ray left       free way to the left
        10     ray right      free way to the right
        11     length         snake length, 0..1 (300 parts or more = 1)
        """

        heading = [0.0] * 4
        heading[self.dir] = 1.0 # current moving direction = True = 1

        food_dx = torus_delta(self.food_x - self.x, SCREEN_W) / (SCREEN_W / 2)
        food_dy = torus_delta(self.food_y - self.y, SCREEN_H) / (SCREEN_H / 2)

        # food position as seen from the snake: ahead / to the right
        fx = DIRECTIONS[self.dir][0] / STEP
        fy = DIRECTIONS[self.dir][1] / STEP
        ahead = food_dx * fx + food_dy * fy
        right = -food_dx * fy + food_dy * fx

        rays = [
            self._rays(self.dir), 
            self._rays((self.dir - 1) % 4), 
            self._rays((self.dir + 1) % 4)
        ] # 3 direction: cause snake can not go direcly opposite

        return np.array(
            heading + 
            [food_dx, food_dy, ahead, right] + 
            rays + 
            [min(len(self.parts) / 300, 1.0)],
            dtype=np.float32)


    def _rays(self, direction_idx): 
        # it calculated 40 step ahed of sime direction .. if it sees it body then stops 
        dx, dy = DIRECTIONS[direction_idx]
        x, y = self.x, self.y

        RAY_STEPS = 40
        for k in range(1, RAY_STEPS + 1):
            x = x + dx
            y = y + dy 
            
            if (x + dx) > SCREEN_W:
                x = 0
            if (x + dx) < 0:
                x = SCREEN_W

            if  (y + dy) > SCREEN_H:
                y = 0
            if  (y + dy) < 0:
                y = SCREEN_H

            if self.occupied[(x, y)] > 0: # means snake tails is there 
                return (k - 1) / RAY_STEPS
        return 1.0


# ========================================================================

class GameInTrainView: 
    def __init__(self): 
        pygame.init()
        self.window = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Snake PPO - training")

        self.font = pygame.font.SysFont(None, size=25)
        self.clock = pygame.time.Clock()
        self.fast = False
        self.frame = 0
    
    def draw(self, envs, text):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                self.fast = not self.fast
        self.frame += 1

        if self.fast and self.frame % 30:
            # as I have many snakes ... so draw only the 30th grame on fast
            return True

        self.window.fill(blue)
        for i, env in enumerate(envs): #envs(ALL SNAKES) >> env one snake 
            color = COLORS[i % len(COLORS)]
            pygame.draw.rect(self.window, color, [env.food_x, env.food_y, FOOD_SIZE + 4, FOOD_SIZE + 4]) # 

            for sp in env.parts:# env.parts -> queue contains body part of snake
                pygame.draw.circle(self.window, color, sp, SNAKE_RADIUS)


        # draw the points .. for eating food
        self.window.blit(self.font.render(text, True, (0, 0, 0)), (0, 0))
        pygame.display.flip()
        if not self.fast:
            self.clock.tick(60) # real time, like snakeGame.py
        return True




# ========================================================================

class PPOModels(nn.Module): 
    def __init__(self, hidden=128): 
        super().__init__()
        self.plicyModel = nn.Sequential(
            nn.Linear(observation_dim, hidden), 
            nn.Tanh(),
            nn.Linear(hidden, hidden), 
            nn.Tanh(), 
            nn.Linear(hidden, N_ACTIONS)
        )
        self.valueModel = nn.Sequential(
            nn.Linear(observation_dim, hidden), 
            nn.Tanh(),
            nn.Linear(hidden, hidden), 
            nn.Tanh(), 
            nn.Linear(hidden, 1)
        )
        
    def policy(self, obs):
        # input obs (N, 12) -> probs (N, 3) and log_probs (N, 3) of the 3 actions.
        logits = self.plicyModel(obs)
        return torch.softmax(logits, dim=-1) , torch.log_softmax(logits, dim=-1)
    
    def value(self, obs):
        return self.valueModel(obs).squeeze(-1)


    

# ========================================================================

def train(train_step = 3000000 , device = "cuda", watch=True):
    torch.set_num_threads(1)

    all_snakes_envs = [SnakeEnv(seed=i) for i in range(N_ENVS)]
    model = PPOModels().to(device) # includes : policy and value model
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    print(f"training PPO model on : {device}")


    # add watcher
    viewer = GameInTrainView()


    obs = torch.tensor(np.stack([e.observe() for e in all_snakes_envs]), device=device)
    scores = deque(maxlen=1000)
    best = 0 # best scores of prev best snakes
    games = 0
    steps = 0
    updates = 0

    start = time.time()
    while steps < train_step:
        #  1. rollout: play T = 256 frames in all 16 snakes, remember everything
        observation_buffer = torch.zeros(T, N_ENVS, observation_dim, device=device) # (256 batch, 16 snakes, 12-> each snake loos 12 data and process)
        action_buffer = torch.zeros(T, N_ENVS, dtype=torch.long, device=device) # action taken by policy model

        logp_buf = torch.zeros(T, N_ENVS, device=device) # shape: (256, 3)
        value_model_buf = torch.zeros(T, N_ENVS, device=device) # action taken by value model
        reward_buf = torch.zeros(T, N_ENVS, device=device)
        done_buf = torch.zeros(T, N_ENVS, device=device)

        for t in range(T):
            with torch.no_grad(): # *** send observation to Policy model and get actions for each snake
                probs, log_probs = model.policy(obs)  # input: shape(16, 12)  .. output: shape(16, 3) /
                action = torch.multinomial(input=probs, num_samples=1).squeeze(dim=-1) # (16,) pick by probability 
                """ multinomial: picks 1 random action for each of the 16 snakes (weighted by its probabilities)
                batch = [[0.70, 0.20, 0.10],    # snake 0 -> mostly 0 -> go straight
                         [0.10, 0.10, 0.80],    # snake 1 -> mostly 2 -> go right
                         [0.34, 0.33, 0.33]]    # snake 2 -> almost fully random
                torch.multinomial(input=batch, num_samples=1).squeeze(dim=1)
                # -> [0, 1, 1]   (this run's result; running again can give a different one)
                """
                observation_buffer[t], action_buffer[t] = obs, action
                logp_buf[t] = log_probs.gather(dim=1, index=action.unsqueeze(dim=1)).squeeze(dim=-1) # (16,) log-prob of the pick
                value_model_buf[t] = model.value(obs)


            actions_perFrame_next_frame = action.cpu().tolist()
            next_obs = []
            rewards = []
            dones = []
            for i, env in enumerate(all_snakes_envs):
                # send the action to snake UI to perform that action 
                # then send the outcome to Value model
                observation_12_list, stripped_reward, done, info = env.steps(actions_perFrame_next_frame[i])
                if done:
                    games += 1
                    scores.append(info["score"])
                    best = max(best, info["score"])

                    if info.get("timeout"):# starved != died: the future still had value
                        with torch.no_grad():
                            stripped_reward += GAMMA * model.value(torch.tensor(observation_12_list, device=device)).item()
                    observation_12_list = env.reset()

                rewards.append(stripped_reward) 
                dones.append(float(done))
                next_obs.append(observation_12_list)

            reward_buf[t] = torch.tensor(rewards, device=device) # CPU -> GPU, all games at once
            done_buf[t] = torch.tensor(dones, device=device)
            obs = torch.tensor(np.stack(next_obs), device=device)
            steps += N_ENVS


            if viewer:
                text = (f"frames {steps:,}   games {games}   avg score {np.mean(scores) if scores else 0:.1f}  best {best}      F = fast / real time   ESC = stop")
                if not viewer.draw(all_snakes_envs, text):
                    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, Path(__file__).with_name("snake_ppo.pt"))
                    print("stopped - model saved")
                    pygame.quit()
                    return


        # ---- 2. advantages
        with torch.no_grad():
            adv = torch.zeros_like(reward_buf)
            last_value_model_pred = model.value(obs)
            running = torch.zeros_like(last_value_model_pred) # same device as the data (CPU or GPU)

            for t in reversed(range(T)):# 256 frame played by model before collect and train the model 
                if t == T - 1: 
                    next_value = last_value_model_pred
                else: 
                    next_value = value_model_buf[t + 1]

                alive = 1.0 - done_buf[t]
                delta = reward_buf[t] + GAMMA * next_value * alive - value_model_buf[t]
                # it calculated the reward is increasing or decreasing from its prev reward
                running = delta + GAMMA * LAMBDA * alive * running
                adv[t] = running

            ret_adv =adv + value_model_buf

        
        # ---- 3. PPO update
        b_observation = observation_buffer.reshape(-1, 12) # 12 data from snake frame observation
        b_action = action_buffer.reshape(-1)
        b_logp = logp_buf.reshape(-1)

        b_adv = adv.reshape(-1)
        b_ret = ret_adv.reshape(-1)

        EPOCHS = 6
        MINIBATCH = 512
        CLIP = 0.2
        VF_COEF = 0.5
        ENT_COEF = 0.01 # bonus for trying different moves
        MAX_GRAD = 0.5
        # N_ENVS -> parallel running snakes  |||| T-> frames 

        for _ in range(EPOCHS):
            for idx in torch.randperm(T*N_ENVS, device=device).split(MINIBATCH):

                probs, log_probs = model.policy(b_observation[idx])
                new_logp = log_probs.gather(dim=1, index=b_action[idx].unsqueeze(dim=1)).squeeze(dim=1)

                ratio = (new_logp - b_logp[idx]).exp()   # new policy / old policy
                entropy = -(probs * log_probs).sum(dim=1)# how unsure the policy is

                a = (b_adv[idx] - b_adv[idx].mean()) / (b_adv[idx].std() + 1e-8)

                policy_loss = -torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a).mean()
                value_loss = (model.value(b_observation[idx]) - b_ret[idx]).pow(2).mean()

                loss = policy_loss + (VF_COEF * value_loss) - (ENT_COEF * entropy.mean())


                opt.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
                opt.step()

        updates += 1
        if updates % 10 == 0:
            elapsed = time.time() - start
            print(f"{steps:>9,} frames  {elapsed:5.0f} s  {steps / elapsed:6,.0f} fps  games {games:5d}  avg score (last 100) {np.mean(scores) if scores else 0:5.1f}  best {best}", flush=True)


    # save the model
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, Path(__file__).with_name("snake_ppo.pt"))
    print(f"done -> {Path(__file__).with_name("snake_ppo.pt")}")






if __name__ == "__main__":
    train(watch=True)


