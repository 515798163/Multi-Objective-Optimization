#-*-coding:gb2312-*-
from typing import Optional, Union, Tuple

import SociaNetwork
import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.registration import EnvSpec
import enum
import numpy as np
import pandas as pd
import math
import logging
import scipy.sparse as sp

# NORMALIZE_FEATURE_PATH = '../data/facebook_nprmalize_features.csv'
NORMALIZE_all_FEATURE_PATH = '../data/facebook/facebook_nprmalize_all_features.csv'


NODE_NUMBER = 4039
FEATURE_CATEGORY = 9
# r_sources = [10, 200, 40, 600, 8]

IDX = 0
NOISE_RANGE = 10

COUNT_RUMOR_NODE_NUMBER = 5

logging.basicConfig(level=logging.INFO,
                    filename='../log/all_reward.txt',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

class Actions:
    def __init__(self):
        self.n = NODE_NUMBER
        self.action_space = {}
        self.update_action_space()

    '''�����罻����ͼ�����¶����ռ�'''
    def update_action_space(self):
        for i in range(NODE_NUMBER):
            self.action_space[i] = i

    '''���ݶ�������ʵ�ʵĽڵ�'''
    def select_node(self, action):
        node_list = []
        for item in action:
            item = int(item)
            node_list.append(self.action_space[item])
        return node_list

class State:
    def __init__(self):
        features_pd = pd.read_csv(NORMALIZE_all_FEATURE_PATH)
        self.features = np.asarray(features_pd, dtype=np.float32)

    def reset(self):
        '''��ʼ��������Ϊ״̬'''
        features_pd = pd.read_csv(NORMALIZE_all_FEATURE_PATH)
        self.features = np.asarray(features_pd, dtype=np.float32)


        return self.features


    def encode(self, action_list, r_sources):
        # global IDX
        # IDX += 1
        # '''ÿ500�ּ�������'''
        # if IDX % NOISE_RANGE == 0:
        #     '''�������'''
        #     noise = np.random.normal(loc=0.0, scale=1, size=self.features.shape)
        #     self.features += noise
        #     print("add noise")
        #     # if IDX == NOISE_RANGE:
        #     #     print("end add noise")
        #     #     logging.info("end add noise{}".format(IDX))


        '''��ѡ�нڵ��������Ϊ0'''
        elect_list = action_list + r_sources
        for i in elect_list:
            for j in range(len(self.features[i])):
                # self.features[i][j] = 0
                self.features[i][j] = 0

        '''��ҥ�ԡ���ҥ�Խڵ��״̬��Ϊ��ѡ��״̬'''
        for i in action_list:
            self.features[i][-1] = 1


        for i in r_sources:
            self.features[i][-2] = 1



        return normalize(self.features)

    @property
    def shape(self):
        return (NODE_NUMBER, FEATURE_CATEGORY)

def normalize(mx):
    # mx = np.asarray(mx)
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  #  ���������
    r_inv = np.power(rowsum, -1).flatten()  # ��͵�-1�η�
    r_inv[np.isinf(r_inv)] = 0.   # �����inf��ת����0
    r_mat_inv = sp.diags(r_inv)  # ����Խ�Ϸ����
    mx = r_mat_inv.dot(mx)  # ����D-1*A���ǶԳƷ�ʽ���򻯷�ʽ
    return mx


class SocialEnv(gym.Env):
    spec = EnvSpec("facebook-t0")

    def __init__(self):
        self.graph = SociaNetwork.SocialNetwork()
        self.state = State()    #״̬
        self.action = Actions() #����
        # self.action_space = gym.spaces.Discrete(n=self.action.n)    #��ɢ�����ռ�
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(COUNT_RUMOR_NODE_NUMBER,), dtype=np.float32)    #���������ռ�

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.state.shape, dtype=np.float32)   #������ʽ�Ĺ۲�ռ�
        self.seed()

    def step(self, actions):

        if True in np.isnan(actions):
            print("*************1")

        # actionΪ��ά��������[0.4011, 0.3826, -0.2315, 0.0657, 0.0002]
        # ��Ҫ��[-1, 1]�Ķ�����ӳ�䵽[0, 4038]��
        # ���ص�action��һ��numpy���飬Ԫ����list����[[1.5 2.  3. ]]
        action_idx = ((actions + 1) / 2) * (NODE_NUMBER-1)

        for i in range(len(action_idx)):
            if math.isnan(action_idx[i]):
                example = 0
            else:
                example = round(action_idx[i])
            action_idx[i] = example
        # �ڶ����ռ���ѡ��ڵ�
        action_list = self.action.select_node(action_idx)
        self.graph.update_crumor_sources(action_list)

        # '''��ȡѡ��Ľڵ㣬������ӵ���ҥ�����Ӽ���'''
        # action_list = []
        # action = self.action.action_space[action_idx]
        # action_list.append(action)
        # self.graph.update_crumor_sources(action_list)

        '''�ж��Ƿ����'''
        if len(self.graph.c_sources) == SociaNetwork.CRUMOR_NUMBER:
            is_done = True
        else:
            is_done = False

        '''���������ѡȡ���ӽڵ㣬�ͽ����Ӽ��ŵ��罻�����д���'''
        if is_done:
            self.graph.social_network_communication()


        '''����״̬�ռ�, ��������״̬'''
        new_state = self.state.encode(action_list, self.graph.r_sources)

        '''info'''
        info = ""

        '''���㽱��'''
        reward = 0

        if is_done:
            rumor_nodes_list, crumor_nodes_list, urumor_nodes_list = self.graph.get_nodes_status_lists()
            # reward = math.log(len(crumor_nodes_list))
            # logging.info("crumor_nodes_list:{},rumor_nodes_list:{}".format(len(crumor_nodes_list), len(rumor_nodes_list)))
            # ��ֹҥ�Խڵ���࣬5���ڵ㶼ѡ��ҥ�Խڵ㣬�������������󣬷�ҥ�ԵĽڵ����0��Ȼ��ͱ���
            len_crumor_nodes_list = len(crumor_nodes_list)
            if len_crumor_nodes_list == 0:
                len_crumor_nodes_list = SociaNetwork.CRUMOR_NUMBER

            reward = math.log(len_crumor_nodes_list) - math.log(len(rumor_nodes_list))
            # reward = math.log(len(crumor_nodes_list)) - math.log(len(rumor_nodes_list))
            # action_set = set(self.graph.c_sources)
            # if len(action_set) < SociaNetwork.CRUMOR_NUMBER:
            #     reward -= 10

            info = {
                "c_rumor": action_list,
                "reward": reward
            }
            logging.info("reward:{}, r_sources:{}, c_sources:{}, actions:{}".format(reward,self.graph.r_sources, self.graph.c_sources, actions))

            if reward >= 3.8:
                infect_time = self.graph.get_nodes_infect_time()
                max_time = 0
                for i in infect_time:
                    if i > max_time and i != SociaNetwork.MAX_TIME:
                        max_time = i
                logging.info("reward:{},r_sources:{} ,c_sources:{},infect_time:{}".format(reward, self.graph.r_sources, self.graph.c_sources, max_time))



        return new_state, reward, is_done, info



    def reset(self):
        del self.graph
        self.graph = SociaNetwork.SocialNetwork()
        # self.graph.initialize_node_state()
        # self.graph.update_node_status(r_sources=r_sources, c_sources=[])

        '''��ʼ��״̬�����ؽڵ��������Ϊ��ʼ�۲����'''
        observation = self.state.reset()

        '''��ѡ�нڵ��������Ϊ0'''
        elect_list = self.graph.c_sources + self.graph.r_sources
        for i in elect_list:
            for j in range(len(observation[i])):
                # self.features[i][j] = 0
                observation[i][j] = 0

        # '''��ҥ�Խڵ��������Ϊ0'''
        # for i in self.graph.r_sources:
        #     for j in range(len(observation[i])):
        #         # self.features[i][j] = 0
        #         observation[i][j] = 0
        for i in self.graph.r_sources:
            observation[i][-2] = 1

        # '''����ҥ�Խڵ��������Ϊ0'''
        # for i in self.graph.c_sources:
        #     for j in range(len(observation[i])):
        #         # self.features[i][j] = 0
        #         observation[i][j] = 0
        for i in self.graph.c_sources:
            observation[i][-1] = 1

        return normalize(observation)


    def render(self, mode="human"):
        pass

    def close(self):
        super().close()

    def seed(self, seed=None):
        return super().seed(seed)



















