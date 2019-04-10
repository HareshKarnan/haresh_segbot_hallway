#!/usr/bin/env python3

import gym
import numpy
import time
from gym import wrappers
import numpy as np
# ROS packages required
import rospy
import rospkg
# import our training environment
import tensorflow as tf
from openai_ros.task_envs.hallway import hallway_collision_avoidance
from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy,CnnPolicy,nature_cnn,ActorCriticPolicy,CnnLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines import A2C,TRPO
# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor
tf.reset_default_graph()

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            # extracted_features = nature_cnn(self.processed_obs, **kwargs)
            self.processed_obs = tf.reshape(self.processed_obs,[-1,23,1])
            extracted_features = tf.layers.conv1d(self.processed_obs,filters=10,kernel_size=2,activation=activ)
            extracted_features = tf.layers.conv1d(extracted_features,filters=5,kernel_size=2,activation=activ,strides=2)

            extracted_features = tf.layers.flatten(extracted_features)

            pi_h = extracted_features
            for i, layer_size in enumerate([128, 128, 128,128]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = extracted_features
            for i, layer_size in enumerate([128, 128,128]):
                vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.value_fn = value_fn
        self.initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})

class CustomCnnPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            # extracted_features = nature_cnn(self.processed_obs, **kwargs)
            extracted_features = tf.layers.conv2d(self.processed_obs,filters=10,kernel_size=2,activation=activ)
            extracted_features = tf.layers.batch_normalization(inputs=extracted_features,axis=-1)
            extracted_features = tf.layers.conv2d(extracted_features,filters=5,kernel_size=2,activation=activ,strides=2)
            extracted_features = tf.layers.batch_normalization(inputs=extracted_features,axis=-1)
            extracted_features = tf.layers.flatten(extracted_features)

            pi_h = extracted_features
            for i, layer_size in enumerate([128, 128, 128,128]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = extracted_features
            for i, layer_size in enumerate([128, 128,128]):
                vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.value_fn = value_fn
        self.initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})

if __name__ == '__main__':

    rospy.init_node('segbot_collision_avoid', anonymous=True, log_level=rospy.WARN)

    # Create the Gym environment
    env = gym.make('HallwayCollision-v0')
    env = DummyVecEnv([lambda: env])

    rospy.loginfo("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('haresh_segbot_hallway')
    # outdir = pkg_path + '/training_results'
    # env = wrappers.Monitor(env, outdir, force=True)
    # rospy.loginfo("Monitor Wrapper started")


    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file

    env.reset()

    # rospy.logerr("Successfully simulated the robot without any errors")
    # n_cpu = 4
    # env = SubprocVecEnv([lambda: env for i in range(n_cpu)])

    # model = A2C(CnnLnLstmPolicy, env, verbose=1,tensorboard_log="./trpo_depth/",entcoeff=0.01,timesteps_per_batch=64)
    model = A2C(CnnLnLstmPolicy, env, verbose=1,tensorboard_log="./trpo_depth/",ent_coef=0.01,n_steps=64)

    model.learn(total_timesteps=100000,tb_log_name="a2c_cnn_ln_depth")
    model.save("trpo_hallway_depth")

    # from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
    # from stable_baselines import DQN
    # model = DQN(CnnPolicy, env, verbose=1,train_freq=64,prioritized_replay=True,tensorboard_log='./trpo_depth')
    # model.learn(total_timesteps=25000,tb_log_name='DQN_Depth')

    obs = env.reset()

    time.sleep(5)
    env.close()