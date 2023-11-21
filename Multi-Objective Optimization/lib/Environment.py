from lib import SocialNetwork
import random
import numpy as np
import gym

class Action:
    def __init__(self, graph):
        self.n = graph.node_number
        self.action_space = {}
        self.update_action_space(graph)

    '''返回随机动作的索引'''
    def sample(self):
        index = random.sample(self.action_space.keys(), 1)
        return index[0]

    '''根据社交网络图，更新动作空间'''
    def update_action_space(self, graph):
        action_space = {}
        for i in range(graph.node_number):
            action_space[i] = i
        self.action_space = action_space

class Observation:
    def __init__(self, graph):
        # self.state_list = np.empty(shape=graph.node_number*2)
        self.state_list = np.empty(shape=graph.node_number)
        self.update_state(graph)
        self.shape = self.state_list.shape

    '''更新状态列表'''
    def update_state(self, graph):
        node_status_list = [0 for x in range(graph.node_number)]
        node_time_list = graph.get_nodes_infect_time()
        for index in graph.r_sources:
            node_status_list[index] = 1
        for index in graph.c_sources:
            node_status_list[index] = 2

        # state_list = node_status_list + node_time_list
        state_list = node_status_list
        self.state_list = np.asarray(state_list, dtype='float64')
        # print(self.state_list)

        # '''数据标准化'''
        # mean = self.state_list.mean(axis=0)
        # self.state_list -= mean
        # std = self.state_list.std(axis=0)
        # self.state_list /= std

class Environment:
    def __init__(self):
        self.graph = SocialNetwork.SocialNetwork()
        self.action_space = Action(graph=self.graph)
        self.observation_space = Observation(graph=self.graph)


    '''初始化状态，返回一个初始观察向量'''
    def reset(self):
        self.graph = SocialNetwork.SocialNetwork()
        self.action_space.update_action_space(graph=self.graph)
        self.observation_space.update_state(graph=self.graph)

        return self.observation_space.state_list

    '''执行一步动作'''
    def step(self, action_index):
        # '''0.获得传播前的节点感染情况，以便计算奖励'''
        # old_rumor_nodes_list, old_crumor_nodes_list, old_urumor_nodes_list \
        #     = self.graph.get_nodes_status_lists()
        # # if len(old_crumor_nodes_list) == 0:
        # #     old_reward = 0.0
        # # else:
        # #     old_reward = self.graph.node_number / len(old_crumor_nodes_list)
        # old_reward = len(old_crumor_nodes_list)

        '''1.执行一步操作，在社交网络中传播'''
        # action = self.action_space.action_space.get(action_index[0])   #根据索引获取反谣言种子节点
        # print(action_index)



        action_list = []
        action_list.append(action_index)
        self.graph.update_crumor_sources(action_list)

        '''5.判断是否结束'''
        if len(self.graph.c_sources) == SocialNetwork.CRUMOR_NUMBER:
            is_done = True
        else:
            is_done = False

        if is_done:
            self.graph.social_network_communication()
        # print('c_sources={}'.format(self.graph.c_sources))

        '''2.更新状态空间和动作空间'''
        self.action_space.update_action_space(graph=self.graph)
        self.observation_space.update_state(graph=self.graph)

        '''3.获得新状态'''
        new_state = self.observation_space.state_list

        # '''4.计算这一步的奖励'''
        # rumor_nodes_list, crumor_nodes_list, urumor_nodes_list = self.graph.get_nodes_status_lists()
        # reward = len(crumor_nodes_list) - old_reward



        # if is_done:
        '''4.计算这一步的奖励'''
        if is_done:

            rumor_nodes_list, crumor_nodes_list, urumor_nodes_list = self.graph.get_nodes_status_lists()
            print("*******************")
            print(len(crumor_nodes_list))
            print("*******************")
            # reward = len(crumor_nodes_list) - old_reward
            reward = len(crumor_nodes_list)
            # else:
            #     reward = 0.0
            infect_time = self.graph.get_nodes_infect_time()
            max_time = 0
            for i in infect_time:
                if i > max_time and i != SocialNetwork.MAX_TIME:
                    max_time = i
            if max_time > SocialNetwork.TIME_THRESHOLD:
                reward = 0
        else:
            reward = 0



        '''6.info信息'''
        info = ""

        return new_state, reward, is_done, info