import numpy as np
import tensorflow as tf
import gym
import pdb

class Critic:
    def __init__(self, sess, n_states, n_actions, gamma, alpha, tau, n_neurons1,
                 n_neurons2, bn):
        ''' '''
        self.sess = sess
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.n_neurons1 = n_neurons1
        self.n_neurons2 = n_neurons2
        self.bn = bn
        
        with tf.variable_scope("Critic"):
            self.s = tf.placeholder(tf.float32, shape=(None, self.n_states),
                                    name='s')
            self.s_ = tf.placeholder(tf.float32, shape=(None, self.n_states),
                                     name='s_')
            self.a = tf.placeholder(tf.float32, shape=(None, self.n_actions),
                                    name='a')
            self.a_ = tf.placeholder(tf.float32, shape=(None, self.n_actions),
                                     name='a_')
            self.y = tf.placeholder(tf.float32, shape=(None, self.n_actions), 
                                    name='y')
            self.train_phase_critic = tf.placeholder(tf.bool, shape=(None),
                                                    name='train_phase_critic') 
            with tf.variable_scope('Q_online_network'):
                self.Q_online = self.build_network(self.s, self.a, 
                                                   trainable=True, 
                                                   bn=bn,
                                                   n_scope='batch_norm')

            with tf.variable_scope('Q_target_network'):
                self.Q_target = tf.stop_gradient(self.build_network(self.s_,
                                                 self.a_, trainable=True,
                                                 bn=bn,
                                                 n_scope='batch_norm'))
            
            self.vars_Q_online = tf.get_collection(
                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='Critic/Q_online_network')
            self.vars_Q_target = tf.get_collection(
                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='Critic/Q_target_network')
            with tf.name_scope("copy"):
                copy_ops = []
                for i, var in enumerate(self.vars_Q_target):
                    copy_ops.append(var.assign(self.vars_Q_online[i]))
                self.copy_online_2_target = tf.group(*copy_ops,
                                                name="copy_online_2_target")

            with tf.name_scope("slow_update"):
                slow_update_ops = []
                for i, var in enumerate(self.vars_Q_target):
                    if var.name.startswith('Critic/Q_target_network/batch_norm'):
                        pass
                    else:
                        slow_update_ops.append(var.assign(
                            tf.multiply(self.vars_Q_online[i], self.tau) + \
                            tf.multiply(self.vars_Q_target[i], 1.0-self.tau)))
                self.slow_update_2_target = tf.group(*slow_update_ops, 
                                                name="slow_update_2_target")

        with tf.name_scope("Critic_Loss"):
            td_error = tf.square(self.y - self.Q_online)
            self.loss = tf.reduce_mean(td_error)
            optimizer = tf.train.AdamOptimizer(self.alpha)
            self.training_op = optimizer.minimize(self.loss)

        with tf.name_scope("qa_grads"):
            self.qa_gradients = tf.gradients(self.Q_online, self.a)

    def fan_init(self, n):
        return tf.random_uniform_initializer(-1.0/np.sqrt(n), 1.0/np.sqrt(n))

    def init_last(self, n):
        return tf.random_uniform_initializer(minval=-n, maxval=n)

    def batch_norm_layer(self, x, train_phase, scope_bn):
        return tf.contrib.layers.batch_norm(x, scale=True, is_training=train_phase, 
                updates_collections=None, decay=0.999, scope=scope_bn)
    
    def build_network(self, s, a, trainable, bn, n_scope):
        regularizer = tf.contrib.layers.l2_regularizer(.01)
        fan = 1.0/np.sqrt(self.n_neurons1)
        ''' 
        if bn:
            s = self.batch_norm_layer(s, train_phase=self.train_phase_critic,
                                      scope_bn=n_scope+'0')'''
        hidden1 = tf.layers.dense(s, self.n_neurons1, name="hidden1",
                                  activation=None,
                                  kernel_initializer=self.fan_init(self.n_states),
                                  bias_initializer=self.fan_init(self.n_states),
                                  kernel_regularizer=regularizer,
                                  bias_regularizer=None,
                                  trainable=trainable)
        if bn:
            hidden1 = self.batch_norm_layer(hidden1, train_phase=self.train_phase_critic,
                                            scope_bn=n_scope+'1')
        hidden1 = tf.nn.relu(hidden1)
        
        ## Add action tensor to 2nd hidden layer
        with tf.variable_scope("hidden2"):
            w1 = tf.Variable(tf.random_uniform([self.n_neurons1, self.n_neurons2],
                                               minval=-fan, maxval=fan),
                                               trainable=trainable)
            w2 = tf.Variable(tf.random_uniform([self.n_actions, self.n_neurons2], 
                                                minval=-fan, maxval=fan),
                                                trainable=trainable)
            b = tf.Variable(tf.random_uniform([self.n_neurons2], 
                                              minval=-fan, maxval=fan),
                                              trainable=trainable)
        tf.contrib.layers.apply_regularization(regularizer, weights_list=[w1, w2])
        augment = tf.matmul(hidden1, w1) + tf.matmul(a, w2) + b

        hidden2 = tf.nn.relu(augment)
        
        ## Set final layer init weights to ensure initial value estimates near zero
        Q_hat = tf.layers.dense(hidden2, self.n_actions, activation=None, 
                                name="Q_hat",
                                kernel_initializer=self.init_last(0.003),
                                bias_initializer=self.init_last(0.003),
                                kernel_regularizer=None,
                                bias_regularizer=None,
                                trainable=trainable)
        return Q_hat
    
    def predict(self, s, a, train_phase=None):
        return self.sess.run(self.Q_online, 
                        feed_dict={self.s: s.reshape(1, s.shape[0]), 
                                   self.a: a.reshape(1, a.shape[0]),
                                   self.train_phase_critic: train_phase})

    def predict_online_batch(self, s, a, train_phase=None):
        return self.sess.run(self.Q_online, feed_dict={self.s: s, self.a: a,
                             self.train_phase_critic: train_phase})

    def predict_target_batch(self, s_, a_, train_phase=None):
        return self.sess.run(self.Q_target, feed_dict={self.s_: s_, self.a_: a_,
                             self.train_phase_critic: train_phase})
    
    def train(self, s_batch, a_batch, y_batch, train_phase=None):
        self.sess.run(self.training_op, feed_dict={
                      self.s: s_batch, self.a: a_batch, self.y: y_batch,
                      self.train_phase_critic: train_phase})
    
    def slow_update_to_target(self):
        self.sess.run(self.slow_update_2_target)

    def copy_online_to_target(self):
        self.sess.run(self.copy_online_2_target)

    def get_qa_grads(self, s, a, train_phase=None):
        return self.sess.run(self.qa_gradients, feed_dict={self.s: s, self.a: a,
                             self.train_phase_critic: train_phase})

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    with tf.Session() as sess:
        critic = Critic()
