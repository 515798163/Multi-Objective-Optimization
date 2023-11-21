import random
import networkx as nx
import pandas as pd
import numpy as np

NODE_NUMBER = 4039
S_SOURCES = [0, 2]
MAX_PATH = 10000
MAX_TIME = 10000
TIME_THRESHOLD = 18

RUMOR_STATUS = 1
CRUMOR_STATUS = 2
SAFE_STATUS = 0
RUMOR_NUMBER = 15
CRUMOR_NUMBER = 5
data = pd.read_csv('facebook_matrix.csv')
matrix = np.asarray(data)
print(matrix)
print(len(matrix))
graph = nx.from_numpy_matrix(matrix)

shortest_path = pd.read_csv('facebook_path.csv')
shortest_path_np = np.asarray(shortest_path)
shortest_path_list = shortest_path_np.tolist()

# matrix = np.asarray(nx.adjacency_matrix(G).todense())
# print(matrix)
# print(nx.number_of_nodes(G))
# G.add_node(1, status='red', time=1)
# print(nx.nodes(G))
# print(G.nodes[0]['status'])
# print(nx.number_of_edges(graph))
# print(nx.edges(graph))

class SocialNetwork:
    def __init__(self):
        self.G = nx.from_numpy_matrix(matrix)
        self.nodes = nx.nodes(self.G)       #图的所有节点
        self.edges = nx.edges(self.G)       #图的所有边
        self.node_number = nx.number_of_nodes(self.G)   #图的节点数
        self.edges_number = nx.number_of_edges(self.G)  #图的边数
        self.initialize_node_state()                    #初始化节点状态
        self.r_sources = self.random_rumor_sources()    #随机生成谣言
        self.c_sources = []                             #初始谣言集为空
        self.update_node_status(r_sources=self.r_sources, c_sources=self.c_sources)    #更新谣言节点
        self.shortest_path = shortest_path_list

    '''初始化节点状态'''
    def initialize_node_state(self):
        for i in range(self.node_number):
            self.G.add_node(i, status=SAFE_STATUS, time=MAX_TIME)

    '''修改节点状态'''
    def update_node_status(self, r_sources=[], c_sources=[]):
        for i in range(len(c_sources)):
            self.G.nodes[c_sources[i]]['status'] = CRUMOR_STATUS
            # self.G.nodes[c_sources[i]]['time'] = 0
        for i in range(len(r_sources)):
            self.G.nodes[r_sources[i]]['status'] = RUMOR_STATUS
            # self.G.nodes[r_sources[i]]['time'] = 0

    '''随机产生谣言'''
    def random_rumor_sources(self):
        # num = range(self.node_number)
        # r_sources = random.sample(num, RUMOR_NUMBER)
        # r_sources = [10, 200, 40, 600, 8] #t = 0
        r_sources = [10, 200, 40, 600, 8, 9, 75, 79, 172, 188, 285, 291, 577, 640, 658] #t = 1  15个
        # r_sources = [10, 200, 40, 600, 8, 9, 75, 79, 172, 188, 285, 291, 577, 640, 658, 25, 67, 98, 103, 113, 122, 169,
        #              185, 186, 199, 201, 212, 224, 231, 232, 258, 271, 277, 304, 322, 323, 325, 332, 341, 342, 599, 628,
        #              632, 643]    #t = 2  44个
        return r_sources

    def update_crumor_sources(self, c_sources=[]):
        old_sources = self.c_sources + c_sources
        self.c_sources = old_sources
        self.update_node_status(r_sources=self.r_sources, c_sources=self.c_sources)



    '''获得节点状态列表'''
    def get_nodes_states(self):
        nodes_status_list = []
        for i in range(self.node_number):
            nodes_status_list.append(self.G.nodes[i]['status'])
        return nodes_status_list

    '''获得节点感染时间列表'''
    def get_nodes_infect_time(self):
        nodes_infect_time_list = []
        for i in range(self.node_number):
            nodes_infect_time_list.append(self.G.nodes[i]['time'])
        return nodes_infect_time_list

    '''传入谣言、反谣言种子集，然后在社交网络传播'''
    def social_network_communication(self):
        self.initialize_node_state()
        self.update_node_status(self.r_sources,self.c_sources)
        r_time_list = self.get_nodes_infect_time()
        c_time_list = self.get_nodes_infect_time()

        # print('r_time_list={}'.format(r_time_list))
        # print('c_time_list={}'.format(c_time_list))


        r_shortest_path_lists = self.get_shortest_path(self.r_sources)
        c_shortest_path_lists = self.get_shortest_path(self.c_sources)

        # print('r_time_list={}'.format(r_shortest_path_lists))
        # print('c_time_list={}'.format(c_shortest_path_lists))


        # 比较每个源节点到任意节点的感染时间，取最小感染时间
        for shortest_path_list in r_shortest_path_lists:
            # print("0->任何节点{0}".format(shortest_path_list))
            for i in range(self.node_number):
                if (shortest_path_list[i] < r_time_list[i]):
                    r_time_list[i] = shortest_path_list[i]

        for shortest_path_list in c_shortest_path_lists:
            # print("0->任何节点{0}".format(shortest_path_list))
            for i in range(self.node_number):
                if (shortest_path_list[i] < c_time_list[i]):
                    c_time_list[i] = shortest_path_list[i]

        self.update_infected_nodes(r_time_list, c_time_list)



    '''获得感染节点，修改感染状态'''
    def update_infected_nodes(self, r_time_list, c_time_list):
        # print(G.nodes[1])
        # print(type(G.nodes[1]))
        # time_list = []
        for i in range(self.node_number):
            if r_time_list[i] <= c_time_list[i] and r_time_list[i] != MAX_TIME:
                self.G.nodes[i]['status'] = RUMOR_STATUS
                self.G.nodes[i]['time'] = r_time_list[i]
                # time_list.append(r_time_list[i])
            if r_time_list[i] > c_time_list[i] and c_time_list[i] != MAX_TIME:
                self.G.nodes[i]['status'] = CRUMOR_STATUS
                self.G.nodes[i]['time'] = c_time_list[i]
                # time_list.append(c_time_list[i])
            if r_time_list[i] == c_time_list[i] and r_time_list[i] == MAX_TIME:
                self.G.nodes[i]['status'] = SAFE_STATUS
                self.G.nodes[i]['time'] = MAX_TIME
                # time_list.append(MAX_TIME)
        # return time_list

    '''获得单源最短路径，并返回列表'''
    # def get_shortest_path(self, sources):
    #     shortest_path_lists = []
    #     # 获得每个源节点到任意节点的单源最短路径
    #     for i in sources:
    #         one_of_shortest_path = []
    #         for j in range(self.node_number):
    #             try:
    #                 one_of_shortest_path.append(nx.dijkstra_path_length(self.G, i, j))
    #                 # print("{0}->{1}={2}".format(i,j,nx.dijkstra_path_length(G, i, j)))
    #                 # print(nx.dijkstra_path(G,i,j))
    #             except:
    #                 one_of_shortest_path.append(MAX_PATH)
    #                 # print("no path")
    #         shortest_path_lists.append(one_of_shortest_path)
    #     return shortest_path_lists

    def get_shortest_path(self, sources):
        shortest_path_lists = []
        for i in sources:
            shortest_path_lists.append(self.shortest_path[i])
        return shortest_path_lists

    def get_nodes_status_lists(self):
        rumor_nodes_list = []
        crumor_nodes_list = []
        urumor_nodes_list = []
        for i in range(self.node_number):
            if self.G.nodes[i]['status'] == RUMOR_STATUS:
                rumor_nodes_list.append(self.G.nodes[i])
            if self.G.nodes[i]['status'] == CRUMOR_STATUS:
                crumor_nodes_list.append(self.G.nodes[i])
            if self.G.nodes[i]['status'] == SAFE_STATUS:
                urumor_nodes_list.append(self.G.nodes[i])
        return rumor_nodes_list, crumor_nodes_list, urumor_nodes_list