import numpy as np
import tensorflow as tf
import pdb
import gym
from ReplayBuffer import ReplayBuffer
from Ornstein_Uhlenbeck import Ornstein_Uhlenbeck

class Actor:
    def __init__(self, n_states, n_actions, alpha, tau, action_bound):
        ''' '''
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.tau = tau
        self.action_bound = action_bound

        with tf.variable_scope("policy_network"):
            self.s = tf.placeholder(tf.float32, shape=(None, self.n_states),
                                    name='s')
            ## from critic network
            self.a = tf.placeholder(tf.float32, shape=(None, self.n_actions),
                                     name='a')
            
            with tf.variable_scope('pi_online_network') as scope:
                self.PI_online = self.build_network(self.s, 
                                                    trainable=True,
                                                    reuse=False)

            with tf.variable_scope('pi_target_network') as scope:
                self.PI_target = tf.stop_gradient(self.build_network(
                                                  self.s,
                                                  trainable=False, 
                                                  reuse=False))

            self.vars_PI_online = tf.trainable_variables()
            self.vars_PI_target = tf.trainable_variables()[len(
                                                        self.vars_PI_online):]
            
            with tf.name_scope("copy"):
                copy_ops = []
                for i, var in enumerate(self.vars_PI_target):
                    copy_ops.append(var.assign(
                                tf.multiply(self.vars_PI_online[i], self.tau) + \
                                tf.multiply(self.vars_PI_online[i], 1-self.tau)))
                self.copy_online_to_target = tf.group(*copy_ops,
                                                name="copy_online_to_target")

            with tf.name_scope("loss"):
               self.action_grad = tf.gradients(self.PI_online, 
                                               self.vars_PI_online,
                                               -self.a)

            with tf.name_scope("train"):
                optimizer = tf.train.AdamOptimizer(self.alpha)
                self.training_op = optimizer.apply_gradients(zip(self.action_grad,
                                                             self.vars_PI_online))
            self.num_trainable_vars = len(self.vars_PI_online) + len(
                                          self.vars_PI_target)

    def fan_init(self, n):
        return tf.random_uniform_initializer(-1.0/np.sqrt(n), 1.0/np.sqrt(n))

    def init_last(self, n):
        return tf.random_uniform_initializer(minval=-n, maxval=n)

    def build_network(self, s, trainable, reuse):
        regularizer = tf.contrib.layers.l2_regularizer(.01)
        hidden1 = tf.layers.dense(s, 400, name="hidden1",
                                  activation=tf.nn.relu,
                                  kernel_initializer=self.fan_init(self.n_states),
                                  bias_initializer=self.fan_init(self.n_states),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  trainable=trainable,
                                  reuse=reuse)
        ## Apply batch normalization
        hidden1 = tf.layers.batch_normalization(hidden1)

        hidden2 = tf.layers.dense(hidden1, 300, name="hidden2",
                                  activation=tf.nn.relu,
                                  kernel_initializer=self.fan_init(400),
                                  bias_initializer=self.fan_init(400),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer,
                                  trainable=trainable,
                                  reuse=reuse)
        ## Apply batch normalization
        hidden2 = tf.layers.batch_normalization(hidden2)

        ## Set final layer init weights to ensure initial value esttimates near 0
        PI_hat = tf.layers.dense(hidden2, self.n_actions, activation=tf.nn.tanh,
                                 name="PI_hat", 
                                 kernel_initializer=self.init_last(0.003),
                                 bias_initializer=self.init_last(0.003),
                                 kernel_regularizer=regularizer,
                                 bias_regularizer=regularizer,
                                 trainable=trainable,
                                 reuse=reuse)
        PI_hat_scaled = tf.multiply(PI_hat, self.action_bound)
        return PI_hat_scaled
    
    def predict(self, s, sess):
        return sess.run(self.PI_online, 
                        feed_dict={self.s: s.reshape(1, s.shape[0])})

    def predict_online_batch(self, s, sess):
        return sess.run(self.PI_online, feed_dict={self.s: s})

    def predict_target_batch(self, s, sess):
        return sess.run(self.PI_target, feed_dict={self.s: s})

    def train(self, replay_buffer, action_grad, sess):
        sess.run(self.training_op, feed_dict={
                            self.s: x_batch, self.y: y_batch})

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

if __name__ == '__main__':
    with tf.Session() as sess:
        agent = Actor()
