import numpy as np
import tensorflow as tf
import gym
import pdb

class Critic:
    def __init__(self, sess, n_states, n_actions, gamma, alpha, tau, n_neurons1,
                 n_neurons2, bn, l2, seed):
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
        self.l2 = l2
        self.seed = seed
        
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
                                                   reg=self.l2,
                                                   network='online')

            with tf.variable_scope('Q_target_network'):
                self.Q_target = tf.stop_gradient(self.build_network(self.s_,
                                                 self.a_, trainable=True,
                                                 bn=bn, reg=False,
                                                 network='target'))
            
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
                    slow_update_ops.append(var.assign(
                            tf.multiply(self.vars_Q_online[i], self.tau) + \
                            tf.multiply(self.vars_Q_target[i], 1.0-self.tau)))
                self.slow_update_2_target = tf.group(*slow_update_ops, 
                                                name="slow_update_2_target")

        with tf.name_scope("Critic_Loss"):
            td_error = tf.square(self.y - self.Q_online)
            self.loss = tf.reduce_mean(td_error)
            if self.l2: 
                #reg_term = tf.reduce_sum(tf.get_collection(
                #                         tf.GraphKeys.REGULARIZATION_LOSSES,
                #                         scope='Critic/Q_online_network'))
                reg_term = tf.losses.get_regularization_loss()
                self.loss += reg_term

            optimizer = tf.train.AdamOptimizer(self.alpha)
            self.training_op = optimizer.minimize(self.loss)

        with tf.name_scope("qa_grads"):
            self.qa_gradients = tf.gradients(self.Q_online, self.a)

    def fan_init(self, n):
        return 1.0/np.sqrt(n)

    def batch_norm_layer(self, x, train_phase, bn_scope):
        return tf.contrib.layers.batch_norm(x, scale=True, is_training=train_phase, 
                updates_collections=None, decay=0.999, epsilon=0.001, scope=bn_scope)
    
    def build_network(self, s, a, trainable, bn, reg, network):
        '''
        if bn:
            s = self.batch_norm_layer(s, train_phase=self.train_phase_critic,
                                      scope_bn='batch_norm0')'''
        w1_init = tf.Variable(self.fan_init(self.n_states) * (
                2.0 * tf.contrib.stateless.stateless_random_uniform(
                                            [self.n_states, self.n_neurons1],
                                            seed=[self.seed, 0]) - 1.0))
        w2_init = tf.Variable(self.fan_init(self.n_neurons1 + self.n_actions) *
                             (2.0 * tf.contrib.stateless.stateless_random_uniform(
                                            [self.n_neurons1, self.n_neurons2], 
                                            seed=[self.seed+2, 0]) - 1.0))
        w3_init = tf.Variable(self.fan_init(self.n_neurons1 + self.n_actions) * 
                             (2.0 * tf.contrib.stateless.stateless_random_uniform(
                                            [self.n_actions, self.n_neurons2], 
                                            seed=[self.seed+3, 0]) - 1.0))
        w4_init = tf.Variable(.003 * (2 *
                             tf.contrib.stateless.stateless_random_uniform(
                                            [self.n_neurons2, self.n_actions],
                                            seed=[self.seed+5, 0]) - 1.0))
        if network == 'online':
            if reg:
                regularizer = tf.contrib.layers.l2_regularizer(.01)
            else:
                regularizer = None
            with tf.variable_scope('reg', regularizer=regularizer):
                w1 = tf.get_variable(name='Critic/Q_online_network/hidden1/w1',
                                     initializer=w1_init)
                w2 = tf.get_variable(name='Critic/Q_online_network/hidden2/w2',
                                     initializer=w2_init)
                w3 = tf.get_variable(name='Critic/Q_online_network/hidden2/w3',
                                     initializer=w3_init)
                w4 = tf.get_variable(name='Critic/Q_online_network/Q_hat/w4',
                                     initializer=w4_init)
        elif network == 'target':
                w1 = tf.get_variable(name='Critic/Q_target_network/hidden1/w1',
                                     initializer=w1_init)
                w2 = tf.get_variable(name='Critic/Q_target_network/hidden2/w2',
                                     initializer=w2_init)
                w3 = tf.get_variable(name='Critic/Q_target_network/hidden2/w3',
                                     initializer=w3_init)
                w4 = tf.get_variable(name='Critic/Q_target_network/Q_hat/w4',
                                     initializer=w4_init)

        with tf.variable_scope("hidden1"):
            '''w1 = tf.Variable(self.fan_init(self.n_states) * 
                             (2.0 * tf.contrib.stateless.stateless_random_uniform(
                                            [self.n_states, self.n_neurons1],
                                            seed=[self.seed, 0]) - 1.0),
                                            trainable=trainable, name='w1')'''
            b1 = tf.Variable(self.fan_init(self.n_states) * 2 * 
                            (2.0 * tf.contrib.stateless.stateless_random_uniform(
                                            [self.n_neurons1],
                                            seed=[self.seed+1, 0]) - 1.0),
                                            trainable=trainable)
            hidden1 = tf.matmul(s, w1) + b1
        
            if bn:
                hidden1 = self.batch_norm_layer(hidden1, train_phase=self.train_phase_critic,
                                            scope_bn='batch_norm1')
            hidden1 = tf.nn.relu(hidden1)
        
        ## Add action tensor to 2nd hidden layer
        with tf.variable_scope("hidden2"):
            '''w2 = tf.Variable(self.fan_init(self.n_neurons1 + self.n_actions) *
                             (2.0 * tf.contrib.stateless.stateless_random_uniform(
                                            [self.n_neurons1, self.n_neurons2], 
                                            seed=[self.seed+2, 0]) - 1.0),
                                            trainable=trainable, name='w2')
            w3 = tf.Variable(self.fan_init(self.n_neurons1 + self.n_actions) * 
                             (2.0 * tf.contrib.stateless.stateless_random_uniform(
                                            [self.n_actions, self.n_neurons2], 
                                            seed=[self.seed+3, 0]) - 1.0),
                                            trainable=trainable, name='w3')'''
            b2 = tf.Variable(self.fan_init(self.n_neurons1 + self.n_actions) *
                             (2.0 * tf.contrib.stateless.stateless_random_uniform(
                                              [self.n_neurons2], 
                                              seed=[self.seed+4, 0]) - 1.0),
                                              trainable=trainable)
            augment = tf.matmul(hidden1, w2) + tf.matmul(a, w3) + b2

            hidden2 = tf.nn.relu(augment)

        with tf.variable_scope("Q_hat"):
            ## Set final layer init weights to ensure initial value estimates near zero
            '''w4 = tf.Variable(.003 * (2 *
                             tf.contrib.stateless.stateless_random_uniform(
                                            [self.n_neurons2, self.n_actions],
                                            seed=[self.seed+5, 0]) - 1.0),
                                            trainable=trainable, name='w4')'''
            b3 = tf.Variable(.003 * (2 *
                             tf.contrib.stateless.stateless_random_uniform(
                                            [self.n_actions],
                                            seed=[self.seed+6, 0]) - 1.0),
                                            trainable=trainable)
            Q_hat = tf.matmul(hidden2, w4) + b3 
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
