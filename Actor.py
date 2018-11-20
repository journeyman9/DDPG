import numpy as np
import tensorflow as tf
import pdb
import gym

class Actor:
    def __init__(self, sess, n_states, n_actions, alpha, tau, action_bound):
        ''' '''
        self.sess = sess
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.tau = tau
        self.action_bound = tf.cast(action_bound, tf.float32)
        ''' 
        self.s = tf.get_default_graph().get_tensor_by_name("s/s:0")
        self.s_ = tf.get_default_graph().get_tensor_by_name("s_/s_:0")
        self.qa_grads = tf.get_default_graph(
                                ).get_tensor_by_name("qa_grads/qa_grads:0")'''
        

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

            with tf.variable_scope('pi_online_network') as scope:
                self.PI_online = self.build_network(self.s, 
                                                    trainable=True,
                                                    reuse=False,
                                                    n_scope='online_norm_')

            with tf.variable_scope('pi_target_network', reuse=False):
                self.PI_target = tf.stop_gradient(self.build_network(
                                                  self.s_,
                                                  trainable=True, 
                                                  reuse=False, 
                                                  n_scope='target_norm_'))
            self.vars_PI_online = tf.get_collection(
                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='Actor/pi_online_network')
            self.vars_PI_target = tf.get_collection(
                                        tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='Actor/pi_target_network')
            '''            
            for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='Actor/pi_online_network'):
                print(i.name)

            print()

            for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='Actor/pi_target_network'):
                print(i.name)
            print()
            print(len(self.vars_PI_online))
            print(len(self.vars_PI_target))
            exit()'''

            with tf.name_scope("copy"):
                copy_ops = []
                for i, var in enumerate(self.vars_PI_target):
                    copy_ops.append(var.assign(self.vars_PI_online[i]))
                self.copy_online_2_target = tf.group(*copy_ops,
                                                name="copy_online_2_target")

            with tf.name_scope("slow_update"):
                slow_update_ops = []
                for i, var in enumerate(self.vars_PI_target):
                    slow_update_ops.append(var.assign(
                            tf.multiply(self.vars_PI_online[i], self.tau) + \
                            tf.multiply(self.vars_PI_target[i], 1.0-self.tau)))
                self.slow_update_2_target = tf.group(*slow_update_ops,
                                                name="slow_update_target")
        with tf.name_scope("Actor_Loss"):
            raw_actor_grads = tf.gradients(self.PI_online, 
                                           self.vars_PI_online,
                                           -self.qa_grads)
            self.actor_grads = list(map(lambda x: tf.div(
                                    x, tf.cast(self.batch_size, tf.float32)), 
                                    raw_actor_grads))
             
            optimizer = tf.train.AdamOptimizer(self.alpha)
            self.training_op = optimizer.apply_gradients(
                                                zip(self.actor_grads,
                                                self.vars_PI_online))

    def fan_init(self, n):
        return tf.random_uniform_initializer(-1.0/np.sqrt(n), 1.0/np.sqrt(n))

    def init_last(self, n):
        return tf.random_uniform_initializer(minval=-n, maxval=n)
    
    def batch_norm_layer(self, x, train_phase, scope_bn):
        return tf.contrib.layers.batch_norm(x, scale=True, is_training=train_phase,
               updates_collections=None, decay=0.999, scope=scope_bn)

    def build_network(self, s, trainable, reuse, n_scope):
        # Apply batch normalization to states
        s = self.batch_norm_layer(s, train_phase=self.train_phase_actor, 
                                  scope_bn=n_scope+'0') 

        hidden1 = tf.layers.dense(s, 400, name="hidden1",
                                  activation=None,
                                  kernel_initializer=self.fan_init(self.n_states),
                                  bias_initializer=self.fan_init(self.n_states),
                                  trainable=trainable,
                                  reuse=reuse)
        ## Apply batch normalization
        hidden1 = self.batch_norm_layer(hidden1, train_phase=self.train_phase_actor,
                                            scope_bn=n_scope+'1')
        hidden1 = tf.nn.relu(hidden1)

        hidden2 = tf.layers.dense(hidden1, 300, name="hidden2",
                                  activation=None,
                                  kernel_initializer=self.fan_init(400),
                                  bias_initializer=self.fan_init(400),
                                  trainable=trainable,
                                  reuse=reuse)
        ## Apply batch normalization
        hidden2 = self.batch_norm_layer(hidden2, train_phase=self.train_phase_actor,
                                        scope_bn=n_scope+'2')
        hidden2 = tf.nn.relu(hidden2)
        
        ## Set final layer init weights to ensure initial value estimates near 0
        PI_hat = tf.layers.dense(hidden2, self.n_actions, activation=tf.nn.tanh,
                                 name="PI_hat", 
                                 kernel_initializer=self.init_last(0.003),
                                 bias_initializer=self.init_last(0.003),
                                 trainable=trainable,
                                 reuse=reuse)
        PI_hat_scaled = tf.multiply(PI_hat, self.action_bound)
        return PI_hat_scaled
    
    def predict(self, s, train_phase=None):
        return self.sess.run(self.PI_online, 
                             feed_dict={self.s: s.reshape(1, s.shape[0]), 
                                    self.train_phase_actor: train_phase})

    def predict_online_batch(self, s, train_phase=None):
        return self.sess.run(self.PI_online, feed_dict={self.s: s,
                                        self.train_phase_actor: train_phase})

    def predict_target_batch(self, s_, train_phase=None):
        return self.sess.run(self.PI_target, feed_dict={self.s_: s_, 
                                        self.train_phase_actor: train_phase})

    def train(self, x_batch, qa_grads, batch_size, train_phase=None):
        self.sess.run(self.training_op, feed_dict={
                            self.s: x_batch, self.qa_grads: qa_grads,
                            self.batch_size: batch_size,
                            self.train_phase_actor: train_phase})

    def copy_online_to_target(self):
        self.sess.run(self.copy_online_2_target)

    def slow_update_to_target(self):
        self.sess.run(self.slow_update_2_target)

if __name__ == '__main__':
    with tf.Session() as sess:
        agent = Actor()
