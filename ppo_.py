import tensorflow as tf
import copy


class trainppo:
    def __init__(self, pol, pol_old, gam=0.95, clip_value=0.2, c_1=1, c_2=0.01):
 

        self.pol = pol
        self.pol_old = pol_old
        self.gam = gam

        pi_trainable = self.pol.get_trainable_variables()
        old_pi_trainable = self.pol_old.get_trainable_variables()

       
        with tf.variable_scope('assign_op'):
            self.assgn_op = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assgn_op.append(tf.assign(v_old, v))

        # train_op inputs
        with tf.variable_scope('train_inp'):
            self.axns = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.rwds = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.nxt_val_preds = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        axn_prob = self.pol.axn_prob
        axn_old_prob = self.pol_old.axn_prob

        # action probabilities taken by agent wiht policy
        axn_prob = axn_prob * tf.one_hot(indices=self.axns, depth=axn_prob.shape[1])
        axn_prob = tf.reduce_sum(axn_prob, axis=1)

        # action probabilities taken by agent wiht old policy
        axn_old_prob = axn_old_prob * tf.one_hot(indices=self.axns, depth=axn_old_prob.shape[1])
        axn_old_prob = tf.reduce_sum(axn_old_prob, axis=1)

        with tf.variable_scope('loss/clip'):

            rtos = tf.exp(tf.log(axn_prob) - tf.log(axn_old_prob))
            clip_rtos = tf.clip_by_value(rtos, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            clip_los = tf.minimum(tf.multiply(self.gaes, rtos), tf.multiply(self.gaes, clip_rtos))
            clip_los = tf.reduce_mean(clip_los)
            tf.summary.scalar('loss_clip', clip_los)

       
        with tf.variable_scope('loss/vf'):
            val_pred = self.pol.val_pred
            valf_los = tf.squared_difference(self.rwds + self.gam * self.nxt_val_preds, val_pred)
            valf_los = tf.reduce_mean(valf_los)
            tf.summary.scalar('loss_vf', valf_los)

     
        with tf.variable_scope('loss/entropy'):
            entrp = -tf.reduce_sum(self.pol.axn_prob *
                                     tf.log(tf.clip_by_value(self.pol.axn_prob, 1e-10, 1.0)), axis=1)
            entrp = tf.reduce_mean(entrp, axis=0)  # mean of entropy of pi(obs)
            tf.summary.scalar('entropy', entrp)

        with tf.variable_scope('loss'):
            _loss = loss_clip - c_1 * valf_los + c_2 * entrp
            _loss = -_loss 
            tf.summary.scalar('loss', _loss)

        self.merged = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
        self.train_op = optimizer.minimize(_loss, var_list=pi_trainable)

    def train(self, obs, axns, rwds, nxt_val_preds, gaes):
        tf.get_default_session().run([self.train_op], feed_dict={self.pol.obs: obs,
                                                                 self.pol_old.obs: obs,
                                                                 self.axns: axns,
                                                                 self.rwds: rwds,
                                                                 self.nxt_val_preds: nxt_val_preds,
                                                                 self.gaes: gaes})

    def get_summary(self, obs, axns, rwds, nxt_val_preds, gaes):
        return tf.get_default_session().run([self.merged], feed_dict={self.pol.obs: obs,
                                                                      self.pol_old.obs: obs,
                                                                      self.axns: axns,
                                                                      self.rwds: rwds,
                                                                      self.nxt_val_preds: nxt_val_preds,
                                                                      self.gaes: gaes})

    def assign_policy_parameters(self):
 
        return tf.get_default_session().run(self.assgn_op)

    def get_gaes(self, rwds, val_pred, nxt_val_preds):
        deltas = [r_t + self.gam * v_next - v for r_t, v_next, v in zip(rwds, nxt_val_preds, val_pred)]
      
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  
            gaes[t] = gaes[t] + self.gam * gaes[t + 1]
        return gaes