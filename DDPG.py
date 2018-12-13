''' TruckbackerUpper using Deep Deterministic Policy Gradients
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
import random
import time
from datetime import timedelta
import os
import select
import sys
import gym_truck_backerupper

GAMMA = 0.99
ALPHA_C = .001
ALPHA_A = .0001
EPISODES = 1000
MAX_BUFFER = 1e6
BATCH_SIZE = 64
COPY_STEPS = 1
TRAIN_STEPS = 1
N_NEURONS1 = 400
N_NEURONS2 = 300
TAU = .001
SEEDS = [0, 1, 12]
LABEL = 'reward_scheme_A'
BN = False
L2 = False

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
        self.a_log = []
        self.goal_log = []
        
        # Unique to truck backerupper
        self.rms_psi_1_log = []
        self.rms_psi_2_log = []
        self.rms_d2_log = []

        self.w = 0.5

        self.convergence_flag = False
        self.completed_episodes = 0

        self.critic.copy_online_to_target()
        self.actor.copy_online_to_target()
    
    # Unique to truck backerupper
    def rms(self, x, axis=None):
        return np.sqrt(np.mean(np.square(x), axis=axis))
 
    def learn(self, env, EPISODES, TRAIN_STEPS, COPY_STEPS, BATCH_SIZE):
        # Tensorboard
        file_writer = tf.summary.FileWriter(logdir, graph=self.sess.graph)
        best_reward = -float('inf')
        start_rendering = False
        settle_model = False
        for episode in range(EPISODES):
            done = False
            q_log = []
            N_log = []
            action_log = []
            
            # Unique to truck backerupper
            psi_1_e_log = []
            psi_2_e_log = []
            d2_e_log = []

            total_reward = 0.0
            self.action_noise.reset()
            #env.manual_course([25.0, 25.0, 225.0], [-25.0, -25.0, 180.0])
            env.manual_course([25.0, 0.0, 180.0], [-5.0, 0.0, 180.0])
            env.manual_offset(2.0, 0.0, 0.0)
            s = env.reset()
            ep_steps = 0
            while not done:
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    command = input()
                    if command == 'render' or command == 'settle':
                        if command == 'render':
                            start_rendering = True
                            print('set render...')
                        elif command == 'settle':
                            print('settling..')
                            settle_model = True
                    else:
                        start_rendering = False
                        env.close()
                        print('hide render...')
                if start_rendering:
                    env.render()

                N = self.action_noise()
                N_log.append(N[0])
                a = self.actor.predict(s, train_phase=False)[0]  

                K = np.array([-27.606229206749300, 99.829605935742920, 
                              -7.853981633974539]) 
                if np.random.uniform(0, 1) < 0.5:
                    #a = np.clip((1 - self.w) * a + self.w * K.dot(s),
                    #            env.action_space.low, env.action_space.high)
                    a = np.clip(K.dot(s), env.action_space.low, 
                                env.action_space.high)
                else:
                    a = np.clip(a + N, 
                                env.action_space.low, env.action_space.high)
                action_log.append(a)
                
                q_log.append(self.critic.predict(s, a, train_phase=False))
                
                s_, r, done, info = env.step(a)
                self.replay_buffer.add_sample((s, a, r, s_, done))
                
                # Unique to truck backerupper
                psi_1_e_log.append(s[0])
                psi_2_e_log.append(s[1])
                d2_e_log.append(s[2])
                
                if done:
                    print()
                    for key, value in info.items():
                        if value:
                            print(key, value)
                    print()

                if self.steps % TRAIN_STEPS == 0 and \
                                    self.replay_buffer.size() >= BATCH_SIZE:
                    s_batch, a_batch, r_batch, s__batch, d_batch = \
                                    self.replay_buffer.sample_batch(BATCH_SIZE)
                    
                    a_hat_ = self.actor.predict_target_batch(s__batch,
                                                             train_phase=True)
                    q_hat_ = self.critic.predict_target_batch(s__batch, a_hat_,
                                                              train_phase=True)
                    y_batch = []
                    for i in range(BATCH_SIZE):
                        if d_batch[i]:
                            y_batch.append(r_batch[i])
                        else:
                            y_batch.append(r_batch[i] + self.critic.gamma * q_hat_[i])

                    self.critic.train(s_batch, a_batch,
                                      np.reshape(y_batch, (BATCH_SIZE, 1)), 
                                      train_phase=True)
                    
                    a_hat = self.actor.predict_online_batch(s_batch, 
                                                            train_phase=False)
                    qa_grads = self.critic.get_qa_grads(s_batch, a_hat, 
                                                        train_phase=False)
                    self.actor.train(s_batch, qa_grads[0], BATCH_SIZE, 
                                     train_phase=True)

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
            self.a_log.append(np.mean(action_log))
            self.goal_log.append(info['goal'])
            
            # Unique convergence criteria to truck backerupper
            self.rms_psi_1_log.append(self.rms(psi_1_e_log))
            self.rms_psi_2_log.append(self.rms(psi_2_e_log))
            self.rms_d2_log.append(self.rms(d2_e_log))

            if total_reward > best_reward:
                best_reward = total_reward
            print("Ep: {}, ".format(episode+1) + 
                  "r: {:.3f}, ".format(total_reward) +
                  "best_r: {:.3f}, ".format(best_reward) + 
                  "qmax: {:.3f}, ".format(self.q_value_log[episode]) +
                  "steps: {}, ".format(self.ep_steps_log[episode]) +
                  "N: {:.3f}, ".format(self.noise_log[episode]) + 
                  "a: {:.3f}, ".format(self.a_log[episode]))

            # Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=total_reward, tag="Reward")
            episode_summary.value.add(simple_value=self.q_value_log[episode], 
                                      tag="Avg_max_Q_hat")
            episode_summary.value.add(simple_value=self.a_log[episode],
                                      tag="Avg_action")
            episode_summary.value.add(simple_value=self.ep_steps_log[episode], 
                                      tag="Steps")
            file_writer.add_summary(episode_summary, global_step=episode+1)
            file_writer.flush()

            self.completed_episodes += 1

            print("Last 100: goal = {}, rms psi_2 = {:.4f}, rms d2 = {:.4f}".format(
                  sum(self.goal_log[-100:]),
                  np.mean(self.rms_psi_2_log[-100:]),
                  np.mean(self.rms_d2_log[-100:])))
            
            if (sum(self.goal_log[-100:]) >= 90 and 
                np.mean(self.rms_psi_2_log[-100:]) <= 0.0897 * 1.10 and
                np.mean(self.rms_d2_log[-100:]) <= 1.0542 * 1.10):
                print('converged')
                self.convergence_flag = True
                break
            
            if settle_model:
                print('saving early...')
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
            if done:
                print()
                for key, value in info.items():
                    if value:
                        print(key, value)
                print()
            s = s_
            total_reward += r
            steps += 1
        return total_reward

if __name__ == '__main__':
    avg_total_test_rewards = []
    avg_convergence_ep = []
    convergence = []
    avg_train_time = []
    if not os.path.exists('./models'):
        os.mkdir('./models')
    env = gym.make('TruckBackerUpper-v0').unwrapped

    for seed_idx in range(len(SEEDS)):
        # prevents merging of data for tensorboard from multiple runs
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = "tf_logs"
        logdir = "{}/{}-{}-run-{}/".format(root_logdir, now, LABEL, 
                  str("seed-") + str(SEEDS[seed_idx])) 

        checkpoint_path = "./models/" + LABEL + "_seed_" + str(SEEDS[seed_idx]) + "/my_ddpg.ckpt"

        np.random.seed(SEEDS[seed_idx])
        tf.set_random_seed(SEEDS[seed_idx])
        env.seed(SEEDS[seed_idx])
        random.seed(SEEDS[seed_idx])
        with tf.Session() as sess:
            critic = Critic(sess, env.observation_space.shape[0], 
                            env.action_space.shape[0], GAMMA, ALPHA_C, TAU,
                            N_NEURONS1, N_NEURONS2, BN, L2, SEEDS[seed_idx])
            actor = Actor(sess, env.observation_space.shape[0],
                          env.action_space.shape[0], ALPHA_A, TAU, 
                          env.action_space.high, N_NEURONS1, N_NEURONS2, BN,
                          SEEDS[seed_idx])
            
            sess.run(tf.global_variables_initializer())
            action_noise = Ornstein_Uhlenbeck(mu=np.zeros(env.action_space.shape[0]))
            replay_buffer = ReplayBuffer(MAX_BUFFER, BATCH_SIZE)
            replay_buffer.clear()
            agent = DDPG(sess, critic, actor, action_noise, replay_buffer)
            saver = tf.train.Saver()

            if len(sys.argv) >= 2:
                saver.restore(sess, 
                        tf.train.latest_checkpoint("./models/" + sys.argv[1] + "/"))
                print('~~~~~~~~~~~~~~~~~~')
                print('Model Restored')
                print('~~~~~~~~~~~~~~~~~~')
            
            startTime = time.time()
            agent.learn(env, EPISODES, TRAIN_STEPS, COPY_STEPS, BATCH_SIZE)
            avg_train_time.append(time.time() - startTime)
            convergence.append(agent.convergence_flag)
            avg_convergence_ep.append(agent.completed_episodes) 
            saver.save(sess, checkpoint_path)
        tf.reset_default_graph()
        
        with tf.Session() as sess:
            avg_test_reward = []
            saved = tf.train.import_meta_graph(checkpoint_path + '.meta',
                                               clear_devices=True)
            saved.restore(sess, checkpoint_path)
            state = sess.graph.get_tensor_by_name('Actor/s:0')
            train_phase = sess.graph.get_tensor_by_name('Actor/train_phase_actor:0')
            learned_policy = sess.graph.get_tensor_by_name(
                    'Actor/pi_online_network/pi_hat/Mul_4:0')

            n_demonstrate = 1
            #pdb.set_trace()
            for ep in range(n_demonstrate):
                #env.manual_track = False
                env.manual_course([25.0, 0.0, 180.0], [-5.0, 0.0, 180.0])
                env.manual_offset(2.0, 0.0, 0.0)
                r = agent.test(env, learned_policy, state, train_phase, sess)
                #print("number of steps in test: {}: {}".format(ep+1, test_steps))
                #print("Reward in test {}: {:.3f}".format(ep+1, r))
                avg_test_reward.append(r)
            avg_total_test_rewards.append(np.mean(avg_test_reward))
            env.close()
        tf.reset_default_graph()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("avg_r: {:.3f}, ".format(np.mean(avg_total_test_rewards)) + 
          "std_r: {:.3f}, ".format(np.std(avg_total_test_rewards)) + 
          "avg_cvg: {} eps, ".format(int(np.mean(avg_convergence_ep))) + 
          "std_cvg: {} eps".format(int(np.std(avg_convergence_ep))))
    print("Converged: {} times".format(sum(convergence)))
    print("avg_t {}, ".format(timedelta(seconds=np.mean(avg_train_time))) +
          "std_t {}".format(timedelta(seconds=np.std(avg_train_time))))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
