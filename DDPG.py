''' Mountain Car Problem using Deep Deterministic Policy Gradients
Journey McDowell (c) 2018
'''

import gym
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pdb
from datetime import datetime
from Critic import Critic
from Actor import Actor
from ReplayBuffer import ReplayBuffer
from Ornstein_Uhlenbeck import Ornstein_Uhlenbeck

# prevents merging of data for tensorboard from multiple runs
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

GAMMA = 0.99
ALPHA_C = .001
ALPHA_A = .0001
EPISODES = 2000
MAX_BUFFER = 1e6
BATCH_SIZE = 64
COPY_STEPS = 10000
TRAIN_STEPS = 1
TAU = .001

class DDPG:
    def __init__(self, actor, critic, action_noise, replay_buffer):
        ''' '''
        self.actor = actor
        self.critic = critic
        self.action_noise = action_noise
        self.replay_buffer = replay_buffer
        self.steps = 0
        self.q_value_log = []
        self.ep_steps_log = []

    def learn(self, env, EPISODES, TRAIN_STEPS, COPY_STEPS, BATCH_SIZE, sess):
        # Tensorboard
        file_writer = tf.summary.FileWriter(logdir, graph=sess.graph)
        best_reward = -float('inf')
        for episodes in range(EPISODES):
            done = False
            q_log = []
            total_reward = 0.0
            s = env.reset()
            ep_steps = 0
            while not done:
                a = self.actor.predict(s + self.action_noise(), sess)
                exit()
                q_log.append(np.max(critic.predict(s, a, sess), axis=-1))
                s_, r, done, info = env.step(a)
                self.replay_buffer.add_sample((s, a, r, s_))

                if self.steps % TRAIN_STEPS == 0 and \
                                    self.replay_buffer.size() >= BATCH_SIZE:
                    batch = self.replay_buffer.sample_batch(BATCH_SIZE)
                    s_batch = np.zeros(shape=(len(batch), critic.n_states))
                    s__batch = np.zeros(shape=(len(batch), critic.n_states))
                    for i, (s, a, r, s_) in enumerate(batch):
                        s_batch[i] = s
                        s__batch[i] = s_

                    q_hat_ = self.critic.predict_target_batch(s__batch, sess)

                    x_batch = np.zeros((len(batch), self.critic.n_states))
                    y_batch = np.zeros((len(batch), self.critic.n_states))
                    a_batch = np.zeros((len(batch), self.critic.n_actions))
                    for i, (s, a, r, s_) in enumerate(batch):
                        if s_ is None:
                            y = r
                        else:
                            y = r + self.critic.gamma * q_hat_[i]
                    x_batch[i] = s
                    y_batch[i] = y
                    a_batch[i] = a

                    q_hat = self.critic.train(x_batch, a_batch, y_batch, sess)

                    a_hat = self.actor.predict(x_batch, sess)
                    grads = self.critic.get_action_grads(x_batch, a_batch, sess)
                    self.actor.train(x_batch, grads[0])


                if self.steps % COPY_STEPS == 0:
                    sess.run(self.actor.copy_online_to_target)
                    sess.run(self.critic.copy_online_to_target)

                s = s_
                total_reward += r
                self.steps += 1
                ep_steps += 1
            if total_reward > best_rewared:
                best_reward = total_reward
            print("Ep: {}, r: {:.3f}, best_r: {:.3f}, qh: {:.3f}".format(
                  episode+1, total_reward, best_reward, 
                  self.q_value_log[episode]))

            # Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=agent.q_value_log[episode], 
                                      tag="Avg_max_Q_hat")
            episode_summary.value.add(simple_value=total_reward, tag="Reward")
            episode_summary.value.add(simple_value=agent.ep_steps_log[episode], 
                                      tag="Steps")
            file_writer.add_summary(episode_summary, episode+1)
            file_writer.flush()
            '''
            if np.mean(critic.ep_steps_log[-100:]) < 130:
                print('converged')
                break'''

    def test(self, env, policy, sess):
        done = False
        s = env.reset()
        total_reward = 0.0
        steps = 0
        while not done:
            env.render()
            #a = 
            s_, r, done, info = env.step(a)
            s = s_
            total_reward += r
            steps += 1
        return steps

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    checkpoint_path = "./model/my_ddpg.ckpt"
    with tf.Session() as sess:
        actor = Actor(env.observation_space.shape[0],
                      env.action_space.shape[0], ALPHA_A, TAU, 
                      env.action_space.high)
        critic = Critic(env.observation_space.shape[0], 
                        env.action_space.shape[0], GAMMA, ALPHA_C, TAU,
                        actor.get_num_trainable_vars())
        action_noise = Ornstein_Uhlenbeck(mu=np.zeros(env.action_space.shape[0]))
        sess.run(tf.global_variables_initializer())
        replay_buffer = ReplayBuffer(MAX_BUFFER, BATCH_SIZE)
        replay_buffer.clear()
        agent = DDPG(critic, actor, action_noise, replay_buffer)
        saver = tf.train.Saver()
        agent.learn(env, EPISODES, TRAIN_STEPS, COPY_STEPS, BATCH_SIZE, sess)
        saver.save(sess, checkpoint_path)
        
        '''
        with tf.Session(graph=tf.Graph()) as sess:
            saved = tf.train.import_meta_graph(checkpoint_path + '.meta',
                                               clear_devices=True)
            saved.restore(sess, checkpoint_path)
            state = sess.graph.get_tensor_by_name('s:0')
            learned_policy = sess.graph.get_tensor_by_name()

            n_demonstrate = 3
            for ep in range(n_demonstrate):
                test_steps = test(env, learned_policy, sess)
                print("number of steps in test: {}: {}".format(ep+1, test_steps))
            env.close()'''
