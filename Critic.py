import numpy as np
import tensorflow as tf
import gym
import pdb
from ReplayBuffer import ReplayBuffer
from Ornstein_Uhlenbeck import Ornstein_Uhlenbeck

class Critic:
    def __init__(self, sess, n_states, n_actions, gamma, alpha, tau, 
                 num_actor_vars):
        self.sess = sess
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        with tf.variable_scope("Critic"):
            self.s = tf.placeholder(tf.float32, shape=(None, self.n_states),
                                    name='s')
            self.s_ = tf.placeholder(tf.float32, shape=(None, self.n_states),
                                     name='s_')
            self.a = tf.placeholder(tf.float32, shape=(None, self.n_actions),
                                    name='a')
            self.y = tf.placeholder(tf.float32, shape=(None, self.n_actions), 
                                    name='y')

            with tf.variable_scope('Q_online_network') as scope:
                self.Q_online = self.build_network(self.s, self.a, 
                                                   trainable=True, 
                                                   reuse=False)

            with tf.variable_scope('Q_target_network', reuse=False):
                self.Q_target = tf.stop_gradient(self.build_network(self.s_,
                                                 self.a, trainable=False, 
                                                 reuse=False))
            
            self.vars_Q_online = tf.trainable_variables()[num_actor_vars:]
            self.vars_Q_target = tf.trainable_variables()[(len(
                                                          self.vars_Q_online)+
                                                          num_actor_vars):]
            
            with tf.name_scope("copy"):
                copy_ops = []
                for i, var in enumerate(self.vars_Q_target):
                    copy_ops.append(var.assign(
                            tf.multiply(self.vars_Q_online[i], self.tau) + \
                            tf.multiply(self.vars_Q_target[i], 1-self.tau)))

                self.copy_online_to_target = tf.group(*copy_ops, 
                                                name="copy_online_to_target")

            with tf.name_scope("loss"):
                self.loss = tf.losses.mean_squared_error(labels=self.y,
                                                    predictions=self.Q_online)
            
            with tf.name_scope("train"):
                optimizer = tf.train.AdamOptimizer(self.alpha)
                self.training_op = optimizer.minimize(self.loss)

            self.q_grads = tf.gradients(self.Q_online, self.a)

    def fan_init(self, n):
        return tf.random_uniform_initializer(-1.0/np.sqrt(n), 1.0/np.sqrt(n))

    def init_last(self, n):
        return tf.random_uniform_initializer(minval=-n, maxval=n)
    
    def build_network(self, s, a, trainable, reuse):
        regularizer = tf.contrib.layers.l2_regularizer(.01)
        hidden1 = tf.layers.dense(s, 400, name="hidden1",
                                  activation=tf.nn.relu,
                                  kernel_initializer=self.fan_init(self.n_states),
                                  bias_initializer=self.fan_init(self.n_states),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  trainable=trainable,
                                  reuse=reuse)
        
        ## Apply batch normalization to layers prior to action input
        #hidden1 = tf.layers.batch_normalization(hidden1)
        
        ## Add action tensor to 2nd hidden layer
        aug_a = tf.layers.dense(a, 400, activation=tf.nn.relu,
                                kernel_initializer=self.fan_init(self.n_states),
                                bias_initializer=self.fan_init(self.n_states),
                                kernel_regularizer=regularizer,
                                bias_regularizer=regularizer,
                                trainable=trainable,
                                reuse=reuse)
        aug = tf.concat([hidden1, aug_a], axis=1)
        #aug = tf.concat([hidden1, a], axis=1)
        hidden2 = tf.layers.dense(aug, 300, name="hidden2",
                                  activation=tf.nn.relu,
                                  kernel_initializer=self.fan_init(400),
                                  bias_initializer=self.fan_init(400),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  trainable=trainable,
                                  reuse=reuse)
        
        ## Set final layer init weights to ensure initial value estimates near zero
        Q_hat = tf.layers.dense(hidden2, self.n_actions, activation=None, 
                                name="Q_hat",
                                kernel_initializer=self.init_last(0.003),
                                bias_initializer=self.init_last(0.003),
                                kernel_regularizer=regularizer,
                                bias_regularizer=regularizer,
                                trainable=trainable,
                                reuse=reuse)
        return Q_hat
    
    def predict(self, s, a):
        return self.sess.run(self.Q_online, 
                        feed_dict={self.s: s.reshape(1, s.shape[0]), 
                                   self.a: a.reshape(1, a.shape[0])})

    def predict_online_batch(self, s, a):
        return self.sess.run(self.Q_online, feed_dict={self.s: s, self.a: a})

    def predict_target_batch(self, s_, a):
        return self.sess.run(self.Q_target, feed_dict={self.s_: s_, self.a: a})
    
    def train(self, x_batch, a_batch, y_batch):
        self.sess.run(self.training_op, feed_dict={
                      self.s: x_batch, self.a: a_batch, self.y: y_batch})

    def get_q_grads(self, s, a):
        return self.sess.run(self.q_grads, feed_dict={self.s: s, self.a: a})

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    with tf.Session() as sess:
        critic = Critic()
