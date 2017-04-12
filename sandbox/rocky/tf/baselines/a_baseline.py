from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
from rllab.core.serializable import Serializable
import rllab.misc.logger as logger
import numpy as np


class AfunctionBaseline(Baseline, Serializable):

    def __init__(self, env_spec, qf, policy):
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec
        self.qf = qf
        self.policy = policy

    def get_action(self, mean, log_std_dev):
        # act = tf.random_normal((num,), mean=mean, stddev=tf.exp(log_std_dev))
        act = tf.random_normal([])
        act = act * tf.exp(log_std_dev) + mean
        return act

    def get_qbaseline_sim(self, obs_var, action_var=None, scale_reward=1.0):
        info_vars = self.policy.dist_info_sym(obs_var)
        if action_var is None:
            action_var = info_vars["mean"]
        q_samples = []

        # print(actions.get_shape())
        for ai in range(1000):
            action = self.get_action(info_vars["mean"], info_vars["log_std"])
            q_temp = self.qf.get_qval_sym(
                obs_var, action,
                deterministic=True,
            )
            q_samples.append(q_temp)
        value = tf.stop_gradient(tf.reduce_mean(q_samples, reduction_indices=0))

        aprime_samples = []
        # actions = self.get_action(1000, info_vars["mean"], info_vars["log_std"])
        for ai in range(1000):
            action = self.get_action(info_vars["mean"], info_vars["log_std"])
            q_temp = self.qf.get_qval_sym(
                obs_var, action,
                deterministic=True,
            )
            a_temp = q_temp - value
            aprime_temp = tf.gradients(a_temp, action)[0]
            aprime_samples.append(aprime_temp*a_temp)
        aprime = tf.reduce_mean(aprime_samples, reduction_indices=0)
        # action_mu = info_vars["mean"]
        qvalue = self.qf.get_qval_sym(
            obs_var, action_var,
            deterministic=True,
        )
        avalue = qvalue - value
        avalue /= scale_reward
        info_vars["avalue"] = avalue
        info_vars["aprime"] = aprime
        info_vars["value"] = value
        # info_vars["qprime"] = tf.gradients(qvalue, action_mu)[0]
        # info_vars["action_mu"] = action_mu

        f_baseline = tensor_utils.compile_function(
            inputs=[obs_var, action_var],
            outputs=qvalue,
        )

        self.opt_info = {
            "f_baseline": f_baseline,
        }

        return info_vars

    @overrides
    def fit(self, paths):
        logger.log("Using qf_baseline.")

    @overrides
    def predict(self, path):
        f_baseline = self.opt_info["f_baseline"]
        return f_baseline(path["observations"])

