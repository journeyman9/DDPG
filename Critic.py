import numpy as np
import tensorflow as tf
import gym
import pdb

class Critic:
    def __init__(self, sess, n_states, n_actions, gamma, alpha, tau):
        ''' '''
        self.sess = sess
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        '''
        with tf.variable_scope("s"):
            self.s = tf.placeholder(tf.float32, shape=(None, self.n_states),
                                    name='s')
        with tf.variable_scope("s_"):
            self.s_ = tf.placeholder(tf.float32, shape=(None, self.n_states),
                                     name='s_')
        with tf.variable_scope("a"):
            self.a = tf.placeholder(tf.float32, shape=(None, self.n_actions),
                                    name='a')
        with tf.variable_scope("y"):
            self.y = tf.placeholder(tf.float32, shape=(None, self.n_actions), 
                                    name='y')'''
        with tf.variable_scope("Critic"):
            self.s = tf.placeholder(tf.float32, shape=(None, self.n_states),
                                    name='s')
            self.s_ = tf.placeholder(tf.float32, shape=(None, self.n_states),
                                     name='s_')
            self.a = tf.placeholder(tf.float32, shape=(None, self.n_actions),
                                    name='a')
            self.y = tf.placeholder(tf.float32, shape=(None, self.n_actions), 
                                    name='y')
            self.train_phase_critic = tf.placeholder(tf.bool, shape=(None),
                                                    name='train_phase_critic') 
            with tf.variable_scope('Q_online_network') as scope:
                self.Q_online = self.build_network(self.s, self.a, 
                                                   trainable=True, 
                                                   reuse=False,
                                                   n_scope='online_norm_')

            with tf.variable_scope('Q_target_network', reuse=False):
                self.Q_target = tf.stop_gradient(self.build_network(self.s_,
                                                 self.a, trainable=True, 
                                                 reuse=False,
                                                 n_scope='target_norm_'))
            
            self.vars_Q_online = tf.get_collection(
                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='Critic/Q_online_network')
            self.vars_Q_target = tf.get_collection(
                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='Critic/Q_target_network')
            '''
            for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='Critic/Q_online_network'):
                print(i.name)

            print()

            for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='Critic/Q_target_network'):
                print(i.name)

            print()

            print(len(self.vars_Q_online))
            print(len(self.vars_Q_target))
            '''

            with tf.name_scope("copy"):
                copy_ops = []
                for i, var in enumerate(self.vars_Q_target):
                    copy_ops.append(var.assign(
                            tf.multiply(self.vars_Q_online[i], self.tau) + \
                            tf.multiply(self.vars_Q_target[i], 1-self.tau)))
                self.copy_online_2_target = tf.group(*copy_ops, 
                                                name="copy_online_to_target")

        with tf.name_scope("Critic_Loss"):
            self.loss = tf.losses.mean_squared_error(labels=self.y,
                                                     predictions=self.Q_online) 
            ''' 
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                scope='Critic')
            with tf.control_dependencies(self.update_ops):
                optimizer = tf.train.AdamOptimizer(self.alpha)
                self.training_op = optimizer.minimize(self.loss)
            '''
            optimizer = tf.train.AdamOptimizer(self.alpha)
            self.training_op = optimizer.minimize(self.loss)

        with tf.name_scope("qa_grads"):
            '''self.qa_grads = tf.placeholder(tf.float32, 
                                           shape=(None, self.n_actions), 
                                           name='qa_grads')'''
            self.qa_gradients = tf.gradients(self.Q_online, self.a)

    def fan_init(self, n):
        return tf.random_uniform_initializer(-1.0/np.sqrt(n), 1.0/np.sqrt(n))

    def init_last(self, n):
        return tf.random_uniform_initializer(minval=-n, maxval=n)

    def batch_norm_layer(self, x, train_phase, scope_bn):
        return tf.contrib.layers.batch_norm(x, scale=True, is_training=train_phase, 
                updates_collections=None, scope=scope_bn)
    
    def build_network(self, s, a, trainable, reuse, n_scope):
        regularizer = tf.contrib.layers.l2_regularizer(.01)
        hidden1 = tf.layers.dense(s, 400, name="hidden1",
                                  activation=tf.nn.relu,
                                  kernel_initializer=self.fan_init(self.n_states),
                                  bias_initializer=self.fan_init(self.n_states),
                                  trainable=trainable,
                                  reuse=reuse)
        
        ## Apply batch normalization to layers prior to action input
        #hidden1 = tf.layers.batch_normalization(hidden1, training=self.train_phase_critic)
 
        hidden1 = self.batch_norm_layer(s, train_phase=self.train_phase_critic,
                                        scope_bn=n_scope+'0')
        
        ## Add action tensor to 2nd hidden layer
        aug = tf.concat([hidden1, a], axis=1)
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
    
    def predict(self, s, a, train_phase=None):
        return self.sess.run(self.Q_online, 
                        feed_dict={self.s: s.reshape(1, s.shape[0]), 
                                   self.a: a.reshape(1, a.shape[0]),
                                   self.train_phase_critic: train_phase})

    def predict_online_batch(self, s, a, train_phase=None):
        return self.sess.run(self.Q_online, feed_dict={self.s: s, self.a: a,
                             self.train_phase_critic: train_phase})

    def predict_target_batch(self, s_, a, train_phase=None):
        return self.sess.run(self.Q_target, feed_dict={self.s_: s_, self.a: a,
                             self.train_phase_critic: train_phase})
    
    def train(self, x_batch, a_batch, y_batch, train_phase=None):
        self.sess.run(self.training_op, feed_dict={
                      self.s: x_batch, self.a: a_batch, self.y: y_batch,
                      self.train_phase_critic: train_phase})
        '''
        self.sess.run([self.training_op, self.update_ops], feed_dict={
                      self.s: x_batch, self.a: a_batch, self.y: y_batch,
                      self.train_phase_critic: train_phase})
        '''

    def copy_online_to_target(self):
        self.sess.run(self.copy_online_2_target)

    def get_qa_grads(self, s, a, train_phase=None):
        return self.sess.run(self.qa_gradients, feed_dict={self.s: s, self.a: a,
                             self.train_phase_critic: train_phase})

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    with tf.Session() as sess:
        critic = Critic()
