import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
import cv2
import wrapped_flappy_bird as gym
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfd = tfp.distributions


class Actor(models.Model):
    def get_config(self):
        pass

    def __init__(self, input_shape, output_shape):
        super(Actor, self).__init__()
        initializer = tf.keras.initializers.random_normal(stddev=0.01)
        bias = tf.keras.initializers.constant(0.01)
        self.conv1 = layers.Conv2D(32, input_shape=input_shape, kernel_size=8, strides=4, padding='same',
                                   kernel_initializer=initializer, bias_initializer=bias, activation='relu')
        self.pooling1 = layers.MaxPooling2D(2, padding='same')
        self.conv2 = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer,
                                   bias_initializer=bias, activation='relu')
        self.pooling2 = layers.MaxPooling2D(2, padding='same')
        self.conv3 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer,
                                   bias_initializer=bias, activation='relu')
        self.pooling3 = layers.MaxPooling2D(2, padding='same')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu', kernel_initializer=initializer, bias_initializer=bias)
        self.out = layers.Dense(output_shape, kernel_initializer=initializer, bias_initializer=bias)
        self.build((1,) + input_shape)
        self.optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def call(self, inputs, training=None, mask=None):
        conv_2 = self.pooling2(self.conv2(self.pooling1(self.conv1(inputs))))
        conv = self.flatten(self.pooling3(self.conv3(conv_2)))
        return self.out(self.dense1(conv))

    @tf.function
    def update(self, obs, action, adv):
        old_pi = tfd.Categorical(logits=self.call(obs)).log_prob(action)
        for _ in tf.range(10):
            with tf.GradientTape() as tape:
                pi = tfd.Categorical(logits=self.call(obs)).log_prob(action)
                ratio = tf.exp(pi - old_pi)
                loss = -tf.reduce_mean(tf.minimum(ratio * adv, tf.clip_by_value(ratio, 0.8, 1.2) * adv))
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))


class Critic(models.Model):
    def get_config(self):
        pass

    def __init__(self, input_shape):
        super(Critic, self).__init__()
        initializer = tf.keras.initializers.random_normal(stddev=0.01)
        bias = tf.keras.initializers.constant(0.01)
        self.conv1 = layers.Conv2D(32, input_shape=input_shape, kernel_size=8, strides=4, padding='same',
                                   kernel_initializer=initializer, bias_initializer=bias, activation='relu')
        self.pooling1 = layers.MaxPooling2D(2, padding='same')
        self.conv2 = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer,
                                   bias_initializer=bias, activation='relu')
        self.pooling2 = layers.MaxPooling2D(2, padding='same')
        self.conv3 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer,
                                   bias_initializer=bias, activation='relu')
        self.pooling3 = layers.MaxPooling2D(2, padding='same')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu', kernel_initializer=initializer, bias_initializer=bias)
        self.out = layers.Dense(1, kernel_initializer=initializer, bias_initializer=bias)
        self.build((1,) + input_shape)

        self.optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def call(self, inputs, training=None, mask=None):
        conv_2 = self.pooling2(self.conv2(self.pooling1(self.conv1(inputs))))
        conv = self.flatten(self.pooling3(self.conv3(conv_2)))
        return self.out(self.dense1(conv))

    @tf.function
    def update(self, obs, score):
        advantage = score - tf.squeeze(self.call(obs))
        with tf.GradientTape() as tape:
            critic_loss = tf.reduce_mean(tf.square(score - tf.squeeze(self.call(obs))))
        grads = tape.gradient(critic_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return advantage


class ExperiencePool:
    def __init__(self, input_shape, batch_size=32):
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=batch_size)
        self.obs = np.zeros(shape=(batch_size, *input_shape), dtype=np.float32)
        self.action = np.zeros(shape=(batch_size,), dtype=np.int)
        self.reward = np.zeros(shape=(batch_size,), dtype=np.float32)
        self.done = np.zeros(shape=(batch_size,), dtype=np.float32)

    def append(self, obs, action, reward, done):
        self.buffer.append((obs, action, reward, done))
        return len(self.buffer) >= self.batch_size

    def sample(self):
        for index in range(self.batch_size):
            (obs, action, reward, done) = self.buffer[index]
            self.obs[index, :] = obs
            self.action[index] = action
            self.reward[index] = reward
            self.done[index] = done
        self.buffer.clear()
        return self.obs, self.action, self.reward, self.done


class GameExtend:
    def __init__(self):
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


class Agent:
    def __init__(self):
        self.input_shape = (80, 80, 4)
        self.output_n = 2
        self.batch_size = 32
        self.actor = Actor(input_shape=self.input_shape, output_shape=self.output_n)
        self.critic = Critic(input_shape=self.input_shape)
        self.experiences = ExperiencePool(self.input_shape, 32)
        self.score = np.zeros(self.batch_size, dtype=np.float32)

    def action(self, obs):
        obs = tf.cast([obs], dtype=tf.float32)
        return tfd.Categorical(logits=tf.squeeze(self.actor(obs))).sample().numpy()

    def update(self, obs, action, next_obs, reward, done):
        new_score = self.critic(tf.cast([next_obs], dtype=tf.float32)).numpy()[0][0]
        i = self.batch_size
        while i > 0:
            i -= 1
            if done[i]:
                new_score = reward[i]
            else:
                new_score = reward[i] + 0.9 * new_score
            self.score[i] = new_score
        advantage = self.critic.update(obs, self.score).numpy()
        self.actor.update(obs, action, advantage)

    def study(self, obs, action, next_obs, reward, done):
        if self.experiences.append(obs, action, reward, done):
            obs, action, reward, done = self.experiences.sample()
            self.update(obs, action, next_obs, reward, done)

    def trains(self):
        game = GameExtend()
        epoch = 0
        t = 0
        while t < 3000000:
            epoch += 1
            rewards = 0
            obs = game.reset()
            while True:
                t += 1
                action = self.action(obs)
                next_obs, reward, done = game.step(action)
                self.study(obs, action, next_obs, reward, done)
                obs = next_obs
                rewards += reward
                if done:
                    print("第{}轮，运行步数：{},奖励：{}".format(epoch, t, rewards))
                    break


if __name__ == "__main__":
    agent = Agent()
    agent.trains()
