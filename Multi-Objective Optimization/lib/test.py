from lib import SocialNetwork
graph = SocialNetwork.SocialNetwork()

print(graph.c_sources)
print(graph.r_sources)
graph.update_crumor_sources([81,81])
graph.social_network_communication()
rumor_nodes_list, crumor_nodes_list, urumor_nodes_list =\
    graph.get_nodes_status_lists()
print(len(rumor_nodes_list))
print(len(crumor_nodes_list))
print(len(urumor_nodes_list))
graph.nodes[10]['status'] = 5
print(graph.nodes[10]['status'])
