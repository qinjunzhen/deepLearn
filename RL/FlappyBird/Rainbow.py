import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, models
from ReplayBuffer import PrioritizedReplayBuffer
from Games import GameExtend

# 设置不使用显卡
tfd = tfp.distributions


class Actor(models.Model):
    def __init__(self, input_shape, output_n, atom_num):
        super(Actor, self).__init__()
        self.layer_1 = layers.Dense(256, activation=tf.keras.activations.tanh)
        self.layer_2 = layers.Dense(128, activation=tf.keras.activations.tanh)
        self.outputs = layers.Dense(output_n * atom_num)
        self.reshape = layers.Reshape((output_n, atom_num))
        self.build((1,) + input_shape)

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        predicts = self.reshape(self.outputs(self.layer_2(self.layer_1(inputs))))
        return self.reshape(predicts)


class Agent:
    def __init__(self, input_shape, output_n, atom_num=51, batch_size=32):
        self.min_value = float(-10)
        self.max_value = float(10)
        self.input_shape = input_shape
        self.output_n = output_n
        self.batch_size = batch_size
        self.model = Actor(input_shape, output_n, atom_num)
        self.target_model = Actor(input_shape, output_n, atom_num)
        self.experiences = PrioritizedReplayBuffer(self.input_shape, size=8196)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
        self.update_counts = 0
        self.target_model.set_weights(self.model.get_weights())
        self.delta_z = float(self.max_value - self.min_value) / (atom_num - 1)  # vrange的间隔大小
        self.reward_gamma = 0.99
        self.atom_num = atom_num
        self.times = 0.
        self.v_range = tf.linspace(self.min_value, self.max_value, atom_num)

    @tf.function
    def update(self, obs, action, next_obs, reward, done):
        supports = self.reward_gamma * (1 - tf.expand_dims(done, axis=-1))
        supports = supports * self.v_range + tf.expand_dims(reward, axis=-1)

        target_probabilities = tf.math.softmax(self.target_model(next_obs), 2)
        batch_indices = tf.argmax(tf.reduce_sum(target_probabilities * self.v_range, 2), axis=1)
        weights = tf.gather(target_probabilities, batch_indices, batch_dims=1)

        clipped_support = tf.clip_by_value(supports, self.min_value, self.max_value)
        clipped_support = clipped_support[:, None, :]
        tiled_support = tf.tile(clipped_support, [1, self.atom_num, 1])

        reshaped_target_support = tf.tile(self.v_range[:, None], [self.batch_size, 1])
        reshaped_target_support = tf.reshape(reshaped_target_support, [self.batch_size, self.atom_num, 1])

        numerator = tf.abs(tiled_support - reshaped_target_support)
        quotient = 1 - (numerator / self.delta_z)
        clipped_quotient = tf.clip_by_value(quotient, 0, 1)

        weights = weights[:, None, :]

        inner_prod = clipped_quotient * weights

        projection = tf.reduce_sum(inner_prod, 2)

        with tf.GradientTape() as tape:
            predicts = self.model(obs)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=projection,
                                                           logits=tf.gather(predicts, action, batch_dims=1))

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return loss

    @tf.function
    def _select_action(self, obs):
        predicts = tf.reduce_sum(tf.math.softmax(self.model(obs), 2) * self.v_range, 2)
        return tf.argmax(predicts, axis=1)

    def action(self, obs):
        obs = np.array([obs], dtype=np.float32)
        epsilon = 1 - 0.99 * min(1., self.update_counts / 100.)
        if random.random() > epsilon:
            q_values = self._select_action(obs).numpy()
            return q_values[0]
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
