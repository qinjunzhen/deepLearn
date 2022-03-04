import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, models
from ReplayBuffer import PrioritizedReplayBuffer
from Games import GameExtend

tfd = tfp.distributions


class Actor(models.Model):
    def __init__(self, input_shape, output_n):
        super(Actor, self).__init__()
        self.layer_1 = layers.Dense(256, activation='relu')
        self.layer_2 = layers.Dense(128, activation='relu')
        self.outputs = layers.Dense(output_n)
        self.build((1,) + input_shape)

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        return self.outputs(self.layer_2(self.layer_1(inputs)))


class Agent:
    def __init__(self, input_shape, output_n, batch_size=32):
        self.input_shape = input_shape
        self.output_n = output_n
        self.batch_size = batch_size
        self.model = Actor(input_shape, output_n)
        self.target_model = Actor(input_shape, output_n)
        self.experiences = PrioritizedReplayBuffer(self.input_shape, size=8196, n_step=3)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
        self.update_counts = 0
        self.target_model.set_weights(self.model.get_weights())

    @tf.function
    def update(self, obs, action, next_obs, reward, done):
        action = tf.one_hot(action, self.output_n)
        # 获取下一状态的动作
        next_action = tf.one_hot(tf.argmax(self.model(next_obs), axis=1), self.output_n)
        targets = reward + 0.99 * (1. - done) * tf.reduce_sum(self.target_model(next_obs) * next_action, axis=1)
        # targets = reward + 0.99 * (1. - done) * tf.reduce_max(self.target_model(next_obs), axis=1)
        with tf.GradientTape() as tape:
            predicts = tf.reduce_sum(self.model(obs) * action, axis=1)
            td_errors = predicts - targets
            loss = tf.reduce_mean(tf.where(tf.abs(td_errors) < 1, tf.square(td_errors) * 0.5, tf.abs(td_errors) - 0.5))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return tf.abs(td_errors + 1e-3)

    def action(self, obs):
        obs = np.array([obs], dtype=np.float32)
        epsilon = 1 - 0.99 * min(1., self.update_counts / 100.)
        if random.random() > epsilon:
            return tf.argmax(tf.squeeze(self.model(obs))).numpy()
        else:
            return np.random.randint(low=0, high=self.output_n)

    def study(self, obs, action, next_obs, reward, done):
        self.experiences.store(obs, action, reward, next_obs, done)
        if len(self.experiences) > self.batch_size:
            o, n_o, a, r, d, indices = self.experiences.sample_batch()
            td_errors = self.update(o, a, n_o, r, d).numpy()
            self.experiences.update_priorities(indices, td_errors)
            self.update_counts += 1
            if self.update_counts % 50 == 0:
                self.target_model.set_weights(self.model.get_weights())

    def trains(self, game):
        all_episode_reward = []
        for epochs in range(300):
            rewards = 0
            obs = game.reset()
            while True:
                # game.render()
                action = self.action(obs)
                next_obs, reward, done, _ = game.step(action)
                self.study(obs, action, next_obs, reward, done)
                obs = next_obs
                rewards += reward
                if done:
                    print("第{}轮奖励：{}".format(epochs + 1, rewards))
                    break
            if len(all_episode_reward) <= 0:
                all_episode_reward.append(rewards)
            else:
                all_episode_reward.append(0.9 * all_episode_reward[-1] + 0.1 * rewards)


if __name__ == "__main__":
    env = GameExtend()
    try:
        agent = Agent(env.observation_space, env.action_space)
        agent.trains(env)
    finally:
        env.reset()
        env.close()
