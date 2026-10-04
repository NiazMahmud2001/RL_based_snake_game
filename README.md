# Snake + PPO from scratch (`main.py`)

16 snakes learn to play Snake at the same time. The game, the networks and the PPO training loop
are all written by hand (Pygame + plain PyTorch, no RL library).

```bash
python main.py
```

| Key | What it does |
|---|---|
| `F` | switch between real time (60 fps) and fast mode (draws every 30th frame, trains at full speed) |
| `ESC` / close window | stop training and save `snake_ppo.pt` |

---

## 1. What is in the file

```
main.py
├── constants            sizes, speeds, PPO settings
├── torus_delta()        shortest distance on a wrapping board
├── class SnakeEnv       ONE snake game (rules only, no drawing)
├── class GameInTrainView  draws all 16 games in one window
├── class PPOModels      the two networks: policy (actor) + value (critic)
└── train()              the PPO loop: rollout -> advantages -> update
```

### Constants

| Name | Value | Meaning |
|---|---|---|
| `observation_dim` | 12 | numbers the AI sees per frame = input size of both networks |
| `N_ACTIONS` | 3 | 0 = straight, 1 = turn left, 2 = turn right |
| `N_ENVS` | 16 | snakes (games) playing at the same time |
| `T` | 256 | frames each snake plays before every network update |
| `SCREEN_W`, `SCREEN_H` | 1200, 600 | board size in px |
| `STEP` | 5 | px the snake moves per frame |
| `DIRECTIONS` | right, down, left, up | clockwise, so turning = `dir ± 1` |
| `RAY_STEPS` | 40 | body sensors look 40 × 5 = 200 px ahead |
| `GAMMA` | 0.99 | how much future rewards count |
| `LAMBDA` | 0.95 | GAE smoothing (real rewards vs critic estimates) |

---

## 2. `SnakeEnv`: one game

| Method | Called by | What it does |
|---|---|---|
| `__init__(seed)` | `train()` (16 times) | own random generator, then `reset()` |
| `reset()` | `__init__`, `train()` after a game over | snake of length 1 at (300, 200) moving right, new food, score 0. Returns the first observation |
| `steps(action)` | `train()`, every frame for every snake | plays ONE frame, returns `(observation, reward, done, info)` |
| `observe()` | `reset()`, `steps()` | builds the 12 numbers the AI sees |
| `_rays(direction)` | `observe()` (3 times) | how far the snake can go in one direction before hitting its own body |
| `_food_dist()` | `reset()`, `steps()` | shortest distance to the food (used for the small "getting closer" reward) |

### What happens inside `steps(action)`, in order

```
1. turn          action 1 -> dir - 1 (left), action 2 -> dir + 1 (right), 0 -> keep going
2. move          x += dx, y += dy, then wrap around the edges
3. self-bite?    head on top of a body part  -> return reward -1, done=True   ("bit itself")
4. food?         eaten -> score +1, grow 3, new food, reward +1
                 not eaten -> reward = 0.01 * (previous distance - new distance) / 5
5. body          add the new head; drop the tail unless the snake is still growing
6. starved?      1500 frames without food -> return reward 0, done=True, "timeout": True
7. otherwise     return (observation, reward, False, {})
```

### The observation: 12 numbers (`observe()`)

| Index | Name | Meaning |
|---|---|---|
| 0–3 | heading | one-hot `[right, down, left, up]`, e.g. moving right = `[1, 0, 0, 0]` |
| 4 | `food_dx` | food left/right of the head, −1..1 (shortest way, can go through the edge) |
| 5 | `food_dy` | food above/below the head, −1..1 |
| 6 | `ahead` | food in front (+) or behind (−), from the snake's point of view |
| 7 | `right` | food to the snake's right (+) or left (−) |
| 8 | ray straight | free way straight ahead before its own body, 0..1 (1 = free for 200 px) |
| 9 | ray left | free way to the left |
| 10 | ray right | free way to the right |
| 11 | length | snake length, 0..1 (300 parts or more = 1) |

`torus_delta(d, size)` makes 4 and 5 use the shortest way: the board wraps, so food at x = 50 seen
from x = 1150 is +105 px (through the right edge), not −1100 px.

---

## 3. `PPOModels`: the two networks

```
policy (actor):   obs (N, 12) -> Linear 128 -> Tanh -> Linear 128 -> Tanh -> Linear 3  -> logits (N, 3)
value  (critic):  obs (N, 12) -> Linear 128 -> Tanh -> Linear 128 -> Tanh -> Linear 1  -> (N, 1) -> squeeze -> (N,)
```

| Method | Input | Output |
|---|---|---|
| `policy(obs)` | `(N, 12)` | `probs (N, 3)` (softmax, each row sums to 1) and `log_probs (N, 3)` |
| `value(obs)` | `(N, 12)` | `(N,)`: one number per snake, "how good is this situation" |

`N` is 16 during the rollout and 512 (a minibatch) during the update. `dim=-1` in the softmax means
"over the 3 actions of each row", so every snake gets its own probabilities.

Compared with PPO for LLMs (policy, value, reward, reference model): here the **reward model** is
the game itself, and there is **no reference model**: the clip on the old log-probs keeps updates small.

---

## 4. `GameInTrainView`: watching the training

`draw(envs, text)` is called once per frame from `train()`:

```
read keys (ESC -> return False, F -> toggle fast)
fast mode and not every 30th frame -> return True (skip drawing)
fill background, draw each snake + its food in its own colour, draw the stats text
flip the screen, wait for 60 fps unless fast
return True
```

There is no separate game loop: **the training loop is the game loop.** The window stays open
because `train()` calls `draw()` after every frame. When `draw()` returns `False`, `train()` saves
the model and returns.

---

## 5. `train()`: the whole PPO loop

```
train()
│ 16 x SnakeEnv, PPOModels -> GPU, Adam optimizer, GameInTrainView
│ obs = first observation of all snakes                               (16, 12)
│
└─ while steps < train_step:                                           ~733 iterations for 3M frames
   │
   ├─ 1. ROLLOUT   for t in range(256):
   │     ├─ policy(obs) -> probs, log_probs                            (16, 3)
   │     ├─ multinomial -> action                                      (16,)   one move per snake
   │     ├─ store obs, action, log-prob of the action, value           row t of each buffer
   │     ├─ for each of the 16 snakes: env.steps(action[i])            plain Python, CPU
   │     │     game over -> count score, (timeout -> bootstrap), env.reset()
   │     ├─ store rewards, dones                                       row t
   │     ├─ obs = new observations                                     (16, 12)
   │     └─ viewer.draw(...)                                           ESC -> save + return
   │
   ├─ 2. ADVANTAGES (GAE)   for t in reversed(range(256)):             fills adv backwards
   │
   └─ 3. UPDATE   for epoch in range(6):
                     for idx in 8 shuffled minibatches of 512:        48 gradient steps per iteration
                        clipped policy loss + value loss - entropy bonus
                        backward, clip gradients, Adam step
```

### 5.1 The buffers and how they fill up

All buffers are created at the start of every iteration, on the GPU:

| Buffer | Shape | dtype | Row `t` holds |
|---|---|---|---|
| `observation_buffer` | `(256, 16, 12)` | float | what each snake saw at frame `t` |
| `action_buffer` | `(256, 16)` | long | the move each snake made (0/1/2) |
| `logp_buf` | `(256, 16)` | float | log-probability of that move (under the policy *at that time*) |
| `value_model_buf` | `(256, 16)` | float | the critic's estimate V(s) |
| `reward_buf` | `(256, 16)` | float | reward each snake got |
| `done_buf` | `(256, 16)` | float | 1.0 if that snake's game ended at frame `t` |

One frame fills one row for all 16 snakes at once:

```
             snake 0  snake 1  ...  snake 15
frame 0    [   ✓        ✓       ...     ✓    ]   <- filled at t = 0
frame 1    [   ✓        ✓       ...     ✓    ]   <- filled at t = 1
  ...
frame 255  [   ✓        ✓       ...     ✓    ]   <- filled at t = 255
```

Shapes of the values written each frame:

| Line | Shape |
|---|---|
| `probs, log_probs = model.policy(obs)` | `(16, 3)`, `(16, 3)` |
| `action = multinomial(probs, 1).squeeze(-1)` | `(16, 1)` -> `(16,)` |
| `log_probs.gather(1, action.unsqueeze(1)).squeeze(-1)` | `(16, 1)` -> `(16,)`: picks column `action[i]` from row `i` |
| `model.value(obs)` | `(16,)` |

### 5.2 Rollout details

- **Sampling, not argmax**: `torch.multinomial` picks a move at random, weighted by its probability
  (`[0.7, 0.2, 0.1]` -> about 70% straight). This is the exploration, and PPO's ratio assumes it.
- **GPU <-> CPU**: the network runs on the GPU, the games in Python on the CPU. Each frame copies
  once in each direction for all snakes together: `action.cpu().tolist()` and
  `torch.tensor(rewards, device=device)`.
- **A game ends** (`done`): the snake is reset immediately, the other 15 keep playing, so every
  column of the buffers can contain several games one after another. `done_buf` marks the borders.
- **Starved is not the same as died**: when a game ends by timeout, the future still had value, so
  the critic's estimate of the last state is added to the reward:
  `reward += GAMMA * V(last observation)` (bootstrapping).

### 5.3 Advantages (GAE): filled backwards

Question: was each move better or worse than the critic expected?

```
delta[t] = reward[t] + GAMMA * V(next state) * alive[t] - V(state[t])     one-step surprise
adv[t]   = delta[t] + GAMMA * LAMBDA * alive[t] * adv[t + 1]              add the future, discounted
alive[t] = 1 - done[t]                                                    0 -> nothing flows across a game over
```

`adv[t]` needs `adv[t + 1]`, so the loop runs **backwards**, one row (16 snakes) at a time:

```
t = 255:  next value = V(obs after the rollout)   (last_value_model_pred, (16,))
          adv[255] = delta[255]
t = 254:  adv[254] = delta[254] + 0.99 * 0.95 * alive[254] * adv[255]
t = 253:  adv[253] = delta[253] + 0.99 * 0.95 * alive[253] * adv[254]
  ...
t = 0:    adv[0]   = delta[0]   + 0.99 * 0.95 * alive[0]   * adv[1]
```

`running` carries `adv[t + 1]` from one row to the next. Shapes: every line works on `(16,)`, and
the result `adv` is `(256, 16)` (created with `torch.zeros_like(reward_buf)`: same shape, dtype
and device).

Then the critic's training target:

```
ret_adv = adv + value_model_buf        (256, 16)   "what the value should have been"
```

### 5.4 PPO update

First flatten frames and snakes into one list of experiences:

| Tensor | Before | After `reshape` |
|---|---|---|
| `b_observation` | `(256, 16, 12)` | `(4096, 12)` |
| `b_action`, `b_logp`, `b_adv`, `b_ret` | `(256, 16)` | `(4096,)` |

Then 6 epochs, each split into 8 shuffled minibatches of 512 (`torch.randperm(4096).split(512)`):

```
probs, log_probs = model.policy(b_observation[idx])                (512, 3)
new_logp = log_probs.gather(... b_action[idx] ...)                 (512,)    same moves, NEW policy
ratio    = exp(new_logp - b_logp[idx])                             (512,)    new prob / old prob
a        = normalised advantages of this minibatch                 (512,)

policy_loss = -mean( min( ratio * a,  clip(ratio, 0.8, 1.2) * a ) )   moves that were good get
                                                                      more likely, but at most ±20%
value_loss  = mean( (V(obs) - ret_adv)^2 )                            critic learns to predict better
entropy     = -sum(probs * log_probs)                                 (512,)  how unsure the policy is

loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
zero_grad -> backward -> clip_grad_norm_(0.5) -> Adam step
```

Per iteration: 4,096 experiences, 6 × 8 = 48 gradient steps. Then the buffers are thrown away and a
new rollout starts with the updated policy (PPO is on-policy).

### 5.5 Settings used in the update

| Name | Value | Meaning |
|---|---|---|
| `EPOCHS` | 6 | passes over the 4,096 experiences |
| `MINIBATCH` | 512 | experiences per gradient step |
| `CLIP` | 0.2 | the policy may change at most ±20% per update |
| `VF_COEF` | 0.5 | weight of the critic's error in the loss |
| `ENT_COEF` | 0.01 | exploration bonus |
| `MAX_GRAD` | 0.5 | gradient clipping |
| learning rate | 3e-4 | Adam |

---

## 6. Notes on the current code

- **Edge wrapping**: `steps()` and `_rays()` move first but then check `self.x + dx` / `x + dx`, so
  the snake jumps to the other side one step early (1195 -> 0 instead of 1195 -> 1200 -> 0). It
  trains fine because the snake and the rays use the same rule; check `self.x > SCREEN_W` (without
  `+ dx`) to match `snakeGame.py` exactly.
- `scores` keeps the last 1000 games, but the progress line says "last 100".
- `train(watch=...)`: the viewer is always created; `watch` is not used yet
  (use `viewer = GameInTrainView() if watch else None` to make it optional).
- Unused / duplicated: `from torch.distributions import Categorical`, `EAT_DIST` (twice),
  `RAY_STEPS = 40` inside `_rays()`.
- Comment fixes: `logp_buf` is `(256, 16)`, not `(256, 3)`; `value_model_buf` holds the critic's
  estimates, not actions.
- For a network this small the CPU is about 2× faster than the GPU (data copies cost more than the
  math); the GPU wins for networks around 8× bigger. `device="cpu"` in `train()` to try it.
