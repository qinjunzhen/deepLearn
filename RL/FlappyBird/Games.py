import cv2
import collections
import numpy as np
import wrapped_flappy_bird as gym


class GameExtend:
    def __init__(self):
        self.observation_space = (80, 80, 4)
        self.action_space = 2
        self.env = None
        self.obs = collections.deque(maxlen=4)
        self.actions = np.eye(2)

    def play(self, action):
        next_obs, reward, done = self.env.frame_step(self.actions[action])
        next_obs = cv2.cvtColor(next_obs, cv2.COLOR_RGB2GRAY)
        next_obs = cv2.resize(next_obs, (80, 80))
        _, next_obs = cv2.threshold(next_obs, 1, 255, cv2.THRESH_BINARY)
        self.obs.append(next_obs / 255.)
        return reward, done

    def step(self, action):
        reward, done = self.play(action)
        return np.array(self.obs, dtype=np.float32).swapaxes(0, 2), reward, done

    def reset(self):
        self.obs.clear()
        self.env = gym.GameState()
        action = np.random.randint(low=0, high=2)
        _, _ = self.play(action)
        self.obs.append(self.obs[0])
        self.obs.append(self.obs[0])
        self.obs.append(self.obs[0])
        return np.array(self.obs, dtype=np.float32).swapaxes(0, 2)

    def close(self):
        self.env.close()
