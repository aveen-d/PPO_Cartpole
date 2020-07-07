import gym
import numpy as np
import tensorflow as tf
from polnet import pol_net
from ppo_ import trainppo

iter = int(1e5)
gam = 0.95


def main():
    en1 = gym.make('CartPole-v1')
    en1.seed(0)
    obvseration_sp = en1.observation_space
    pol = pol_net('policy', en1)
    pol_old = pol_net('old_policy', en1)
    _ppo = trainppo(pol, pol_old, gamma=gam)
    sav_ = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./log/train', sess.graph)
        sess.run(tf.global_variables_initializer())
        obv_ = en1.reset()
        rew_ = 0
        sucs_no = 0

        for itr in range(iter):  
            obsv_ = []
            axns = []
            val_pred = []
            rwds = []
            pol_run_stps = 0
            while True:  
                pol_run_stps += 1
                obv_ = np.stack([obv_]).astype(dtype=np.float32)  # placeholder for policy.obs
                act, v_pred = pol.act(obs=obv_, stochastic=True)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                obsv_.append(obv_)
                axns.append(act)
                val_pred.append(v_pred)
                rwds.append(rew_)

                next_obs, rew_, done, info = en1.step(act)

                if done:
                    v_preds_next = val_pred[1:] + [0]  
                    obv_ = en1.reset()
                    rew_ = -1
                    break
                else:
                    obv_ = next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=pol_run_stps)])
                               , itr)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rwds))])
                               , itr)

            if sum(rwds) >= 195:
                sucs_no += 1
                if sucs_no >= 100:
                    sav_.save(sess, './model/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                sucs_no = 0

            gaes = _ppo.get_gaes(rewards=rwds, v_preds=val_pred, v_preds_next=v_preds_next)

            # list to numpy array to feed tf.placeholder
            obsv_ = np.reshape(obsv_, newshape=[-1] + list(obvseration_sp.shape))
            axns = np.array(axns).astype(dtype=np.int32)
            rwds = np.array(rwds).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()

            _ppo.assign_policy_parameters()

            inp = [obsv_, axns, rwds, v_preds_next, gaes]

            # train
            for epoch in range(4):
                sample_indices = np.random.randint(low=0, high=obsv_.shape[0], size=64)  # format of indices[low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # train data saple
                _ppo.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          rewards=sampled_inp[2],
                          v_preds_next=sampled_inp[3],
                          gaes=sampled_inp[4])

            summary = _ppo.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      rewards=inp[2],
                                      v_preds_next=inp[3],
                                      gaes=inp[4])[0]

            writer.add_summary(summary, itr)
        writer.close()


if __name__ == '__main__':
    main()