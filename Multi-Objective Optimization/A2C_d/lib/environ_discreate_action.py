# -*- coding: utf-8 -*-
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
NOISE_RANGE = 100

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

    '''根据社交网络图，更新动作空间'''
    def update_action_space(self):
        for i in range(NODE_NUMBER):
            self.action_space[i] = i

    # '''根据动作返回实际的节点'''
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
        '''初始化特征作为状态'''
        features_pd = pd.read_csv(NORMALIZE_all_FEATURE_PATH)
        self.features = np.asarray(features_pd, dtype=np.float32)


        return self.features


    def encode(self, action_list, r_sources):
        global IDX
        IDX += 1
        '''每500轮加入噪声'''
        if IDX % NOISE_RANGE == 0:
            '''添加噪声'''
            noise = np.random.normal(loc=0.0, scale=1, size=self.features.shape)
            self.features += noise
            print("add noise")
            # if IDX == NOISE_RANGE:
            #     print("end add noise")
            #     logging.info("end add noise{}".format(IDX))


        '''将选中节点的特征置为0'''
        elect_list = action_list + r_sources
        for i in elect_list:
            for j in range(len(self.features[i])):
                # self.features[i][j] = 0
                self.features[i][j] = 0

        '''将谣言、反谣言节点的状态置为被选中状态'''
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
    rowsum = np.array(mx.sum(1))  #  矩阵行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.   # 如果是inf，转换成0
    r_mat_inv = sp.diags(r_inv)  # 构造对角戏矩阵
    mx = r_mat_inv.dot(mx)  # 构造D-1*A，非对称方式，简化方式
    return mx


class SocialEnv(gym.Env):
    spec = EnvSpec("facebook-v0")

    def __init__(self):
        self.graph = SociaNetwork.SocialNetwork()
        self.state = State()    #状态
        self.action = Actions() #动作
        self.action_space = gym.spaces.Discrete(n=self.action.n)    #离散动作空间
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(COUNT_RUMOR_NODE_NUMBER,), dtype=np.float32)    #连续动作空间


        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.state.shape, dtype=np.float32)   #张量形式的观察空间
        self.seed()

    def step(self, actions):

        # 在动作空间中选择节点
        action_list = self.action.select_node(actions)
        self.graph.update_crumor_sources(action_list)

        # '''获取选择的节点，将其添加到反谣言种子集中'''
        # action_list = []
        # action = self.action.action_space[action_idx]
        # action_list.append(action)
        # self.graph.update_crumor_sources(action_list)

        '''判断是否结束'''
        if len(self.graph.c_sources) == SociaNetwork.CRUMOR_NUMBER:
            is_done = True
        else:
            is_done = False

        '''如果结束了选取种子节点，就将种子集放到社交网络中传播'''
        if is_done:
            self.graph.social_network_communication()


        '''更新状态空间, 并返回新状态'''
        new_state = self.state.encode(action_list, self.graph.r_sources)

        '''info'''
        info = ""

        '''计算奖励'''
        reward = 0

        if is_done:
            rumor_nodes_list, crumor_nodes_list, urumor_nodes_list = self.graph.get_nodes_status_lists()
            # reward = math.log(len(crumor_nodes_list))
            # logging.info("crumor_nodes_list:{},rumor_nodes_list:{}".format(len(crumor_nodes_list), len(rumor_nodes_list)))
            # 防止谣言节点过多，5个节点都选到谣言节点，那样传播结束后，反谣言的节点就是0，然后就报错
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

        '''初始化状态，返回节点的特征作为初始观察矩阵'''
        observation = self.state.reset()

        '''将选中节点的特征置为0'''
        elect_list = self.graph.c_sources + self.graph.r_sources
        for i in elect_list:
            for j in range(len(observation[i])):
                # self.features[i][j] = 0
                observation[i][j] = 0

        # '''将谣言节点的特征置为0'''
        # for i in self.graph.r_sources:
        #     for j in range(len(observation[i])):
        #         # self.features[i][j] = 0
        #         observation[i][j] = 0
        for i in self.graph.r_sources:
            observation[i][-2] = 1

        # '''将反谣言节点的特征置为0'''
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



















