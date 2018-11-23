import numpy as np
import tensorflow as tf
import pdb
import gym

class Actor:
    def __init__(self, sess, n_states, n_actions, alpha, tau, action_bound,
                 n_neurons1, n_neurons2, bn, seed):
        ''' '''
        self.sess = sess
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.tau = tau
        self.action_bound = tf.cast(action_bound, tf.float32)
        self.n_neurons1 = n_neurons1
        self.n_neurons2 = n_neurons2
        self.bn = bn
        self.seed = seed

        with tf.variable_scope("Actor"):
            self.s = tf.placeholder(tf.float32, shape=(None, self.n_states),
                                    name='s')
            self.s_ = tf.placeholder(tf.float32, shape=(None, self.n_states),
                                     name='s_')
            self.qa_grads = tf.placeholder(tf.float32, 
                                           shape=(None, self.n_actions), 
                                           name='qa_grads')
            self.train_phase_actor = tf.placeholder(tf.bool, shape=(None), 
                                                    name='train_phase_actor')
            self.batch_size = tf.placeholder(tf.float32,
                                             name='batch_size')

            with tf.variable_scope('pi_online_network'):
                self.pi_online = self.build_network(self.s, 
                                                    trainable=True,
                                                    bn=self.bn,
                                                    n_scope='batch_norm')

            with tf.variable_scope('pi_target_network'):
                self.pi_target = tf.stop_gradient(self.build_network(
                                                  self.s_,
                                                  trainable=True, 
                                                  bn=self.bn,
                                                  n_scope='batch_norm'))

            self.vars_pi_online = tf.get_collection(
                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='Actor/pi_online_network')
            self.vars_pi_target = tf.get_collection(
                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='Actor/pi_target_network')
            with tf.name_scope("copy"):
                copy_ops = []
                for i, var in enumerate(self.vars_pi_target):
                    copy_ops.append(var.assign(self.vars_pi_online[i]))
                self.copy_online_2_target = tf.group(*copy_ops,
                                                name="copy_online_2_target")

            with tf.name_scope("slow_update"):
                slow_update_ops = []
                for i, var in enumerate(self.vars_pi_target):
                    slow_update_ops.append(var.assign(
                            tf.multiply(self.vars_pi_online[i], self.tau) + \
                            tf.multiply(self.vars_pi_target[i], 1.0-self.tau)))
                self.slow_update_2_target = tf.group(*slow_update_ops,
                                                name="slow_update_2_target")
        with tf.name_scope("Actor_Loss"):
            raw_actor_grads = tf.gradients(self.pi_online, 
                                           self.vars_pi_online,
                                           -self.qa_grads)
            self.actor_grads = list(map(lambda x: tf.div(
                                    x, self.batch_size), 
                                    raw_actor_grads))
             
            optimizer = tf.train.AdamOptimizer(self.alpha)
            self.training_op = optimizer.apply_gradients(
                                                zip(self.actor_grads,
                                                self.vars_pi_online))

    def fan_init(self, n):
        return 1.0/np.sqrt(n)

    def batch_norm_layer(self, x, train_phase, scope_bn):
        return tf.contrib.layers.batch_norm(x, scale=True, is_training=train_phase,
               updates_collections=None, decay=0.999, epsilon=0.001, scope=scope_bn)

    def build_network(self, s, trainable, bn, n_scope):
        if bn:
            s = self.batch_norm_layer(s, train_phase=self.train_phase_actor, 
                                      scope_bn=n_scope+'0')
        with tf.variable_scope("hidden1"):
            w1 = tf.Variable(self.fan_init(self.n_states) * 
                             tf.contrib.stateless.stateless_truncated_normal(
                                        [self.n_states, self.n_neurons1],
                                        seed=[self.seed, 0]),
                                        trainable=trainable)
            b1 = tf.Variable(self.fan_init(self.n_states) * 
                             tf.contrib.stateless.stateless_truncated_normal(
                                        [self.n_neurons1],
                                        seed=[self.seed+1, 0]),
                                        trainable=trainable)
            hidden1 = tf.matmul(s, w1) + b1
            if bn:
                hidden1 = self.batch_norm_layer(hidden1, 
                                            train_phase=self.train_phase_actor,
                                            scope_bn=n_scope+'1')
            hidden1 = tf.nn.relu(hidden1)

        with tf.variable_scope("hidden2"):
            w2 = tf.Variable(self.fan_init(self.n_neurons1) * 
                             tf.contrib.stateless.stateless_truncated_normal(
                                        [self.n_neurons1, self.n_neurons2],
                                        seed=[self.seed+2, 0]),
                                        trainable=trainable)
            b2 = tf.Variable(self.fan_init(self.n_neurons1) * 
                             tf.contrib.stateless.stateless_truncated_normal(
                                        [self.n_neurons2],
                                        seed=[self.seed+3, 0]),
                                        trainable=trainable)
            hidden2 = tf.matmul(hidden1, w2) + b2
    
            if bn:
                hidden2 = self.batch_norm_layer(hidden2, 
                                            train_phase=self.train_phase_actor,
                                            scope_bn=n_scope+'2')
            hidden2 = tf.nn.relu(hidden2)
        
        ## Set final layer init weights to ensure initial value estimates near 0
        with tf.variable_scope("pi_hat"):
            w3 = tf.Variable(0.003 * 
                             tf.contrib.stateless.stateless_truncated_normal(
                                        [self.n_neurons2, self.n_actions],
                                        seed=[self.seed+4, 0]),
                                        trainable=trainable)
            b3 = tf.Variable(0.003 * 
                             tf.contrib.stateless.stateless_truncated_normal(
                                        [self.n_actions],
                                        seed=[self.seed+5, 0]),
                                        trainable=trainable)
            pi_hat = tf.matmul(hidden2, w3) + b3
            pi_hat = tf.nn.tanh(pi_hat)
            pi_hat_scaled = tf.multiply(pi_hat, self.action_bound) 
        return pi_hat_scaled
    
    def predict(self, s, train_phase=None):
        return self.sess.run(self.pi_online, 
                             feed_dict={self.s: s.reshape(1, s.shape[0]), 
                                    self.train_phase_actor: train_phase})

    def predict_online_batch(self, s, train_phase=None):
        return self.sess.run(self.pi_online, feed_dict={self.s: s,
                                        self.train_phase_actor: train_phase})

    def predict_target_batch(self, s_, train_phase=None):
        return self.sess.run(self.pi_target, feed_dict={self.s_: s_, 
                                        self.train_phase_actor: train_phase})

    def train(self, s_batch, qa_grads, batch_size, train_phase=None):
        self.sess.run(self.training_op, feed_dict={
                            self.s: s_batch, self.qa_grads: qa_grads,
                            self.batch_size: batch_size,
                            self.train_phase_actor: train_phase})

    def copy_online_to_target(self):
        self.sess.run(self.copy_online_2_target)

    def slow_update_to_target(self):
        self.sess.run(self.slow_update_2_target)

if __name__ == '__main__':
    with tf.Session() as sess:
        agent = Actor()
