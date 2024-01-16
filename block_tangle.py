# Tangle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import threading
import random

class node_graph():

    def __init__(self):
        self.time = 1
        self.nodes = []
        self.count = 0
        self.edgelist = []
        self.nodeIDlist = []
        self.g = nx.Graph()

    def plot_graph(self):
            self.g.add_nodes_from(self.nodeIDlist)
            self.g.add_edges_from(self.edgelist)
            nx.draw_networkx(self.g)
    
    def new_node(self):
        #include adding the neighbourhood
        nodeID = self.count
        self.count += 1
        if not self.nodes:
            n = node([], nodeID)
            self.nodeIDlist.append(nodeID)

        elif len(self.nodes) == 1:
            n = node(self.nodes, nodeID)
            self.edgelist.append((nodeID, self.nodes[0].id))
            self.nodeIDlist.append(nodeID)

        else:
            edges = []
            j = 0
            while j < 2:
                item = random.choice(self.nodes)
                if item not in edges:
                    edges.append(item)
                    j += 1
            n = node(edges, nodeID)
            self.edgelist.append((nodeID, edges[0].id))
            self.edgelist.append((nodeID, edges[1].id))
            self.nodeIDlist.append(nodeID)

        self.update_neighbours(n)
        self.nodes.append(n)
        print("Finished updating")
    
    def delete_node(self):
        #update the graph so that when the node is deleted, its neighbours are connected
        #if they are already connected to eachother then connect to a random other node
        #if there are only 2 nodes then end function
        return 0
    
    def update_neighbours(self, newNode):
        #Update the neighouhoods of all nodes in graph
        if len(self.nodes) == 1:
            single = self.nodes[0]
            single.update_neighbourhood(newNode)

        elif len(self.nodes) > 1:
            for i in self.nodes:
                if i.id == newNode.neighbourhood[0]:
                    i.update_neighbourhood(newNode)
                elif i.id == newNode.neighbourhood[1]:
                    i.update_neighbourhood(newNode)
        else:
            print("First node")
    

class node():

    def __init__(self, edges, nodeID):
        self.id = nodeID
        self.neighbourhood = edges
        self.signature = np.random.randint(2048)
    
    def ww_assign():
        #Create and assign the witness weight of the node
        return 0
    
    def orphaned_block():
        #Ask neighbours for the block
        return 0
    
    def issue_block():
        #Take a bunch of parameters for the block and transactions within
        #Issue the block to the tangle with these parameters and the node's witness weight
        return 0
    
    def get_block_list():
        #Return all the blocks issued by this node
        return 0
    
    def tip_select():
        #Select the blocks and or transactions to approve (up to k)
        #Ask the block tangle for all the unapproved tips
        return 0
    
    def update_neighbourhood(self, newNeighbour):
        self.neighbourhood.append(newNeighbour.id)


class Block():

    def __init__():
        #Create the block based on the info given
        #Inputs
        #Outputs
        #References and vote types
        #Signature of node 
        return 0
    
    def is_visible(self):
        return self.tangle.time >= self.time + 1.0

    def is_tip(self):
        return self.tangle.time < self.approved_time

    def is_tip_delayed(self):
        return self.tangle.time - 1.0 < self.approved_time
    
    def approved_directly_by(self):
        return [p for p in self._approved_directly_by if p.is_visible()]
    
    def approved_by(self):
        for t in self._approved_directly_by:
            if t not in self.tangle.t_cache:
                self.tangle.t_cache.add(t)
                self.tangle.t_cache.update(t.approved_by())

        return self.tangle.t_cache

    # add a function to make a transaction within the block


class block_tangle():
    
    def __init__(self, rate=50, alpha=0.001, tip_selection='mcmc', plot=False, nodes=[]):
        self.time = 1.0
        self.rate = rate
        self.alpha = alpha
        self.genesis = Genesis(self)
        self.blocks = [self.genesis]
        self.count = 1
        self.tip_selection = tip_selection
        self.issuers = nodes
    
    def next_block():
        #get the block the node has been issued by the node
        #put it into the block list
        return 0
    
    def tips():
        #get all unapproved tips 
        return 0
    
    
class Genesis(Block):

    def __init__(self, tangle):
        self.tangle = tangle
        self.time = 0
        self.approved_transactions = []
        self.approved_time = float('inf')
        self._approved_directly_by = set()
        self.num = 0
        if hasattr(self.tangle, 'G'):
            self.tangle.G.add_node(self.num, pos=(self.time, 0))

    def __repr__(self):
        return '<Genesis>'