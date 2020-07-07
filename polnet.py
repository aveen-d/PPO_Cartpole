import gym
import numpy as np
import tensorflow as tf


class pol_net:
    def __init__(self, name: str, en1, temp=0.1):

        ob_space = en1.observation_space
        act_space = en1.action_space

        with tf.variable_scope(name):
            self.obv = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape), name='obs')

            with tf.variable_scope('policy_net'):
                lay1 = tf.layers.dense(inputs=self.obv, units=20, activation=tf.tanh)
                lay2 = tf.layers.dense(inputs=lay1, units=20, activation=tf.tanh)
                lay3 = tf.layers.dense(inputs=lay2, units=act_space.n, activation=tf.tanh)
                self.axn_prob = tf.layers.dense(inputs=tf.divide(lay3, temp), units=act_space.n, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                lay1 = tf.layers.dense(inputs=self.obv, units=20, activation=tf.tanh)
                lay2 = tf.layers.dense(inputs=lay1, units=20, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=lay2, units=1, activation=None)

            self.axn_stocas = tf.multinomial(tf.log(self.axn_prob), num_samples=1)
            self.axn_stocas = tf.reshape(self.axn_stocas, shape=[-1])

            self.axn_deter = tf.argmax(self.axn_prob, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obv, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.axn_stocas, self.v_preds], feed_dict={self.obv: obv})
        else:
            return tf.get_default_session().run([self.axn_deter, self.v_preds], feed_dict={self.obv: obv})

    def get_action_prob(self, obv):
        return tf.get_default_session().run(self.axn_prob, feed_dict={self.obv: obv})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
