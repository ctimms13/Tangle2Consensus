# Tangle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import threading
import random


class node_graph():

    def __init__(self, blocktangle):
        self.time = 1
        self.nodes = []
        self.count = 0
        self.edgelist = []
        self.nodeIDlist = []
        self.nodeWeights = []
        self.blocktangle = blocktangle
        self.g = nx.Graph()

    def plot_graph(self):
            self.g.add_nodes_from(self.nodeIDlist)
            self.g.add_edges_from(self.edgelist)
            nx.draw_networkx(self.g)
    
    def printStats(self):
        print("Nodes RAW", self.nodes)
        print("Edges", self.edgelist)
        print("Node IDS", self.nodeIDlist)
        print("Weights", self.nodeWeights)
    
    def assignWW(self):
        ww = 1 / len(self.nodes)
        self.nodeWeights.clear()
        return ww

    def new_node(self):
        nodeID = self.count
        self.count += 1
        if not self.nodes:
            ww = 1
            n = node([], nodeID,  self.blocktangle, ww)
            self.nodeIDlist.append(nodeID)
            self.nodeWeights.append([nodeID, n.signature, ww])
            for m in self.nodes:
                m.ww = ww
                self.nodeWeights.append([m.id, m.signature, m.ww])

        elif len(self.nodes) == 1:
            ww = self.assignWW()
            n = node(self.nodes, nodeID, self.blocktangle, ww)
            for m in self.nodes:
                m.ww = ww
                self.nodeWeights.append([m.id, m.signature, m.ww])
            self.edgelist.append((nodeID, self.nodes[0].id))
            self.nodeIDlist.append(nodeID)
            self.nodeWeights.append([nodeID, n.signature, ww])

        else:
            edges = []
            ww = self.assignWW()
            j = 0
            while j < 2:
                item = random.choice(self.nodes)
                if item not in edges:
                    edges.append(item)
                    j += 1
            n = node(edges, nodeID, self.blocktangle, ww)
            for m in self.nodes:
                m.ww = ww
                self.nodeWeights.append([m.id, m.signature, m.ww])
            self.nodeWeights.append([nodeID, n.signature, ww])
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

    def __init__(self, edges, nodeID, blockTangle, ww):
        self.id = nodeID
        self.neighbourhood = edges
        self.signature = np.random.randint(2048)
        self.ww = ww
        self.block_tangle = blockTangle
       
    def orphaned_block():
        #Ask neighbours for the block
        return 0
    
    def issue_block(self):
        #Take a bunch of parameters for the block and transactions within
        inputs = random.randint(0, 2000000000)
        outputs = random.randint(0, 2000000000)
        nodeSig = self.signature
        self.block_tangle.next_block(inputs, outputs, nodeSig)
    
    def get_block_list():
        #Return all the blocks issued by this node
        return 0
    
    def update_neighbourhood(self, newNeighbour):
        self.neighbourhood.append(newNeighbour.id)

class Block(object):

    def __init__(self, time, inputs, outputs, approved_tips, ID, nodeSig, bt):
        #Create the block based on the info given
        self.time = time
        self.blockTangle = bt
        self.blockID = ID
        self.Transaction = transaction(inputs, outputs, self.blockID)
        self.references = []
        self.approved_blocks = approved_tips
        self.approved_time = float('inf')
        self.signature = nodeSig 
        self._approved_by = []
        self._approved_directly_by = []
    
    def is_visible(self):
        #print(self.time, self.blockTangle.time)
        return self.blockTangle.time >= self.time + 1.0

    def is_tip(self):
        return self.blockTangle.time < self.approved_time
    
    def approved_directly_by(self):
        return [p for p in self._approved_directly_by if p.is_visible()]
    
    def approved_by(self):
        for t in self._approved_directly_by:
            if t not in self.tangle.t_cache:
                self.tangle.t_cache.add(t)
                self.tangle.t_cache.update(t.approved_by())
        return self.tangle.t_cache

class transaction():
    def __init__(self, inputs, outputs, blockid):
        self.inputs = inputs
        self.outputs = outputs
        self.blockID = blockid

class block_tangle(object):
    def __init__(self, rate=50, alpha=0.001, tip_selection='urts', plot=False):
        self.time = 1.0
        self.rate = rate
        self.alpha = alpha
        self.genesis = Genesis(self)
        self.blocks = [self.genesis]
        self.count = 1
        self.tip_selection = tip_selection
        #self.issuers = nodes
        self.t_cache = set()
        self.tip_walk_cache = []
        self.g = nx.DiGraph()
    
    def plotBlock(self):
        #self.g.add_nodes_from(self.b)
        #self.g.add_edges_from(self.edgelist)
        nx.draw_networkx(self.g)


    def next_block(self, inputs, outputs, sig):
        dt_time = np.random.exponential(1.0/self.rate)
        self.time += dt_time
        self.count += 1

        if self.tip_selection == 'mcmc':
            #approved_tips = set(self.mcmc())
            approved_tips = [self.mcmc()]
        elif self.tip_selection == 'urts':
            #approved_tips = set(self.urts())
            approved_tips = [self.urts()]
        else:
            raise Exception()

        block = Block(self.time, inputs, outputs, approved_tips, self.count -1, sig, self)
        self.blocks.append(block)
        for t in approved_tips:
            t.approved_time = np.minimum(self.time, t.approved_time)
            t._approved_directly_by.add(block)
            self.g.add_edges_from([(block.blockID, t.blockID)])
    
    def tips(self):
        #get all unapproved tips 
        print(self.blocks)
        return [t for t in self.blocks if t.is_visible()]
    
    def urts(self):
        tips = self.tips()
        print("Into URTS", tips)
        if len(tips) == 0:
            choice = [t for t in self.blocks if t.is_visible()]
            if len(choice) == 0:
                print("Fuck")
                return [self.genesis]
            #return np.random.choice([t for t in self.blocks if t.is_visible()])
        elif len(tips) == 1:
            return tips[0]
        else:
            k = random.randint(2, len(tips)-1)  # added k because the new protocol allows for up to k approvals per block
            return np.random.choice(tips, k)
"""
    def mcmc(self):
        num_particles = 10
        lower_bound = int(np.maximum(0, self.count - 20.0*self.rate))
        upper_bound = int(np.maximum(1, self.count - 10.0*self.rate))

        candidates = self.transactions[lower_bound:upper_bound]

        particles = np.random.choice(candidates, num_particles)
        distances = {}
        for p in particles:
            t = threading.Thread(target=self._walk2(p))
            t.start()

        tips = self.tip_walk_cache[:2]
        self.tip_walk_cache = list()

        return tips

    def _walk2(self, starting_transaction):
        p = starting_transaction
        while not p.is_tip_delayed() and p.is_visible():
            if len(self.tip_walk_cache) >= 2:
                return

            next_transactions = p.approved_directly_by()
            if self.alpha > 0:
                p_cw = p.cumulative_weight_delayed()
                c_weights = np.array([])
                for transaction in next_transactions:
                    c_weights = np.append(c_weights, transaction.cumulative_weight_delayed())

                deno = np.sum(np.exp(-self.alpha * (p_cw - c_weights)))
                probs = np.divide(np.exp(-self.alpha * (p_cw - c_weights)), deno)
            else:
                probs = None

            p = np.random.choice(next_transactions, p=probs)

        self.tip_walk_cache.append(p)
    
    def _walk(self, starting_transaction):
        p = starting_transaction
        count = 0
        while not p.is_tip_delayed() and p.is_visible():
            next_transactions = p.approved_directly_by()
            if self.alpha > 0:
                p_cw = p.cumulative_weight_delayed()
                c_weights = np.array([])
                for transaction in next_transactions:
                    c_weights = np.append(c_weights, transaction.cumulative_weight_delayed())

                deno = np.sum(np.exp(-self.alpha * (p_cw - c_weights)))
                probs = np.divide(np.exp(-self.alpha * (p_cw - c_weights)), deno)
            else:
                probs = None

            p = np.random.choice(next_transactions, p=probs)
            count += 1

        return p, count

  
"""
  
class Genesis(Block):

    def __init__(self, blockT):
        self.blockTangle = blockT
        self.time = 0
        self.approved_transactions = []
        self.approved_time = float('inf')
        self._approved_directly_by = set()
        self.blockID = 0
    def __repr__(self):
        return '<Genesis>'
    
class Universe():

    def __init__(self):
        self.true_block_tangle = block_tangle()
        self.true_node_graph = node_graph(self.true_block_tangle)
    
    def start_universe(self, nodeNum):
        i = 0
        while i < nodeNum:
            self.true_node_graph.new_node()
            i += 1
        