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
EPISODES = 1000
MAX_BUFFER = 1e6
BATCH_SIZE = 64
COPY_STEPS = 1
TRAIN_STEPS = 1
TAU = .001

class DDPG:
    def __init__(self, sess, critic, actor, action_noise, replay_buffer):
        ''' '''
        self.sess = sess
        self.critic = critic
        self.actor = actor
        self.action_noise = action_noise
        self.replay_buffer = replay_buffer
        self.steps = 0
        self.q_value_log = []
        self.ep_steps_log = []
        self.r_log = [] 
        self.noise_log = []

        self.critic.copy_online_to_target()
        self.actor.copy_online_to_target()
 
    def learn(self, env, EPISODES, TRAIN_STEPS, COPY_STEPS, BATCH_SIZE):
        # Tensorboard
        file_writer = tf.summary.FileWriter(logdir, graph=self.sess.graph)
        best_reward = -float('inf')
        for episode in range(EPISODES):
            done = False
            q_log = []
            N_log = []
            total_reward = 0.0
            self.action_noise.reset()
            s = env.reset()
            ep_steps = 0
            while not done:
                #env.render()
                N = self.action_noise()
                a = self.actor.predict(s, train_phase=False) + N

                N_log.append(N[0])
                q_log.append(self.critic.predict(s, a, train_phase=False))
                
                s_, r, done, info = env.step(a[0])
                self.replay_buffer.add_sample((s, a[0], r, s_))

                if self.steps % TRAIN_STEPS == 0 and \
                                    self.replay_buffer.size() >= BATCH_SIZE:
                    batch = self.replay_buffer.sample_batch(BATCH_SIZE)
                    s__batch = np.zeros(shape=(len(batch), 
                                        self.critic.n_states))
                    for i, (s, a, r, s_) in enumerate(batch):
                        s__batch[i] = s_
                    
                    a_hat_ = self.actor.predict_target_batch(s__batch,
                                                             train_phase=False)
                    q_hat_ = self.critic.predict_target_batch(s__batch, a_hat_,
                                                              train_phase=False)

                    x_batch = np.zeros((len(batch), self.critic.n_states))
                    y_batch = np.zeros((len(batch), self.critic.n_actions))
                    a_batch = np.zeros((len(batch), self.critic.n_actions))
                    for i, (s, a, r, s_) in enumerate(batch):
                        if s_ is None:
                            y = r
                        else:
                            y = r + self.critic.gamma * q_hat_[i]
                        x_batch[i] = s
                        y_batch[i] = y
                        a_batch[i] = a

                    self.critic.train(x_batch, a_batch, y_batch, train_phase=True)
                    
                    a_hat = self.actor.predict_online_batch(x_batch, 
                                                            train_phase=False)
                    qa_grads = self.critic.get_qa_grads(x_batch, a_hat, 
                                                        train_phase=False)
                    self.actor.train(x_batch, qa_grads[0], BATCH_SIZE, train_phase=True)

                if self.steps % COPY_STEPS == 0:
                    self.actor.slow_update_to_target()
                    self.critic.slow_update_to_target()

                s = s_
                total_reward += r
                self.steps += 1
                ep_steps += 1
                
            self.q_value_log.append(np.mean(q_log))
            self.ep_steps_log.append(ep_steps)
            self.r_log.append(total_reward)
            self.noise_log.append(np.mean(N_log))

            if total_reward > best_reward:
                best_reward = total_reward
            print("Ep: {}, ".format(episode+1) + 
                  "r: {:.3f}, ".format(total_reward) +
                  "best_r: {:.3f}, ".format(best_reward) + 
                  "qmax: {:.3f}, ".format(self.q_value_log[episode]) +
                  "steps: {}, ".format(self.ep_steps_log[episode]) +
                  "N: {:.3f}".format(self.noise_log[episode]))

            # Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=self.q_value_log[episode], 
                                      tag="Avg_max_Q_hat")
            episode_summary.value.add(simple_value=total_reward, tag="Reward")
            episode_summary.value.add(simple_value=self.ep_steps_log[episode], 
                                      tag="Steps")
            file_writer.add_summary(episode_summary, episode+1)
            file_writer.flush()
            
            if np.mean(self.r_log[-100:]) > 90:
                print('converged')
                break

    def test(self, env, policy, state, train_phase, sess):
        done = False
        s = env.reset()
        total_reward = 0.0
        steps = 0
        while not done:
            env.render()
            a = sess.run(policy, feed_dict={state: s.reshape(1, s.shape[0]),
                                            train_phase: False})
            s_, r, done, info = env.step(a)
            s = s_
            total_reward += r
            steps += 1
        return total_reward

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    checkpoint_path = "./model/my_ddpg.ckpt"
    with tf.Session() as sess:
        critic = Critic(sess, env.observation_space.shape[0], 
                        env.action_space.shape[0], GAMMA, ALPHA_C, TAU)
        actor = Actor(sess, env.observation_space.shape[0],
                      env.action_space.shape[0], ALPHA_A, TAU, 
                      env.action_space.high)
        
        sess.run(tf.global_variables_initializer())
        action_noise = Ornstein_Uhlenbeck(mu=np.zeros(env.action_space.shape[0]))
        replay_buffer = ReplayBuffer(MAX_BUFFER, BATCH_SIZE)
        replay_buffer.clear()
        agent = DDPG(sess, critic, actor, action_noise, replay_buffer)
        saver = tf.train.Saver()
        agent.learn(env, EPISODES, TRAIN_STEPS, COPY_STEPS, BATCH_SIZE)
        saver.save(sess, checkpoint_path)
        
    with tf.Session(graph=tf.Graph()) as sess:
        saved = tf.train.import_meta_graph(checkpoint_path + '.meta',
                                           clear_devices=True)
        saved.restore(sess, checkpoint_path)
        state = sess.graph.get_tensor_by_name('Actor/s:0')
        train_phase = sess.graph.get_tensor_by_name('Actor/train_phase_actor:0')
        learned_policy = sess.graph.get_tensor_by_name(
                'Actor/pi_online_network/Mul:0')

        n_demonstrate = 3
        pdb.set_trace()
        for ep in range(n_demonstrate):
            r = agent.test(env, learned_policy, state, train_phase, sess)
            #print("number of steps in test: {}: {}".format(ep+1, test_steps))
            print("Reward in test {}: {:.3f}".format(ep+1, r))
        env.close()
