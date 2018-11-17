import numpy as np
import tensorflow as tf
import pdb
import gym
from ReplayBuffer import ReplayBuffer
from Ornstein_Uhlenbeck import Ornstein_Uhlenbeck

class Actor:
    def __init__(self, sess, n_states, n_actions, alpha, tau,
                 batch_size, action_bound):
        ''' '''
        self.sess = sess
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.action_bound = action_bound

        with tf.variable_scope("Actor"):
            self.s = tf.placeholder(tf.float32, shape=(None, self.n_states),
                                    name='s')
            self.s_ = tf.placeholder(tf.float32, shape=(None, self.n_states),
                                     name='s_')
            ## from critic network
            self.q_grads = tf.placeholder(tf.float32, shape=(None, self.n_actions),
                                         name='q_grads')
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")
            
            with tf.variable_scope('pi_online_network') as scope:
                self.PI_online = self.build_network(self.s, 
                                                    trainable=True,
                                                    reuse=False)

            with tf.variable_scope('pi_target_network') as scope:
                self.PI_target = tf.stop_gradient(self.build_network(
                                                  self.s_,
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
                                tf.multiply(self.vars_PI_target[i], 1-self.tau)))
                self.copy_online_to_target = tf.group(*copy_ops,
                                                name="copy_online_to_target")

            with tf.name_scope("loss"):
                self.actor_grads = tf.gradients(self.PI_online, 
                                                self.vars_PI_online,
                                                -self.q_grads)
                '''
                self.actor_grads = list(map(lambda x: tf.div(
                                    x, self.batch_size), self.a_actor_grads))'''

            with tf.name_scope("train"):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    optimizer = tf.train.AdamOptimizer(self.alpha)
                    self.training_op = optimizer.apply_gradients(
                                            zip(self.actor_grads,
                                                self.vars_PI_online))

            self.num_trainable_vars = len(self.vars_PI_online) + len(
                                          self.vars_PI_target)

    def fan_init(self, n):
        return tf.random_uniform_initializer(-1.0/np.sqrt(n), 1.0/np.sqrt(n))

    def init_last(self, n):
        return tf.random_uniform_initializer(minval=-n, maxval=n)

    def batch_norm(self, x, train_phase=None):
        return tf.layers.batch_normalization(x, training=train_phase)

    def build_network(self, s, trainable, reuse):
        #s = self.batch_norm(s, train_phase=self.train_phase)
        hidden1 = tf.layers.dense(s, 400, name="hidden1",
                                  activation=tf.nn.relu,
                                  kernel_initializer=self.fan_init(self.n_states),
                                  bias_initializer=self.fan_init(self.n_states),
                                  trainable=trainable,
                                  reuse=reuse)
        ## Apply batch normalization
        #hidden1 = tf.layers.batch_normalization(hidden1)

        hidden2 = tf.layers.dense(hidden1, 300, name="hidden2",
                                  activation=tf.nn.relu,
                                  kernel_initializer=self.fan_init(400),
                                  bias_initializer=self.fan_init(400),
                                  trainable=trainable,
                                  reuse=reuse)
        ## Apply batch normalization
        #hidden2 = tf.layers.batch_normalization(hidden2)

        ## Set final layer init weights to ensure initial value estimates near 0
        PI_hat = tf.layers.dense(hidden2, self.n_actions, activation=tf.nn.tanh,
                                 name="PI_hat", 
                                 kernel_initializer=self.init_last(0.003),
                                 bias_initializer=self.init_last(0.003),
                                 trainable=trainable,
                                 reuse=reuse)
        PI_hat_scaled = tf.multiply(PI_hat, self.action_bound)
        #print(PI_hat_scaled.name)
        return PI_hat_scaled
    
    def predict(self, s, train_phase=None):
        return self.sess.run(self.PI_online, 
                             feed_dict={self.s: s.reshape(1, s.shape[0]), 
                                        self.train_phase: train_phase})

    def predict_online_batch(self, s, train_phase=None):
        return self.sess.run(self.PI_online, feed_dict={self.s: s,
                                                self.train_phase: train_phase})

    def predict_target_batch(self, s_, train_phase=None):
        return self.sess.run(self.PI_target, feed_dict={self.s_: s_, 
                                                self.train_phase: train_phase})

    def train(self, x_batch, q_grads, train_phase=None):
        self.sess.run(self.training_op, feed_dict={
                            self.s: x_batch, self.q_grads: q_grads,
                            self.train_phase: train_phase})

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

if __name__ == '__main__':
    with tf.Session() as sess:
        agent = Actor()
