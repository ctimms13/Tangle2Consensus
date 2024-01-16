import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import threading

class Ledger_Dag(object):

    def __init__(self, rate=50, alpha=0.001, tip_selection='mcmc', plot=False):
        self.time = 1.0
        self.rate = rate
        self.alpha = alpha

        self.genesis = Genesis(self)
        self.transactions = [self.genesis]
        self.count = 1
        self.tip_selection = tip_selection

    def next_block(self):
        return 0

    def tips(self):
        return [t for t in self.transactions if t.is_visible() and t.is_tip_delayed()]

    def urts(self):
        tips = self.tips()
        if len(tips) == 0:
            return np.random.choice([t for t in self.transactions if t.is_visible()]),
        if len(tips) == 1:
            return tips[0],
        return np.random.choice(tips, 2)

    def mcmc(self):
        num_particles = 10
        lower_bound = int(np.maximum(0, self.count - 20.0*self.rate))
        upper_bound = int(np.maximum(1, self.count - 10.0*self.rate))

        candidates = self.transactions[lower_bound:upper_bound]
        #at_least_5_cw = [t for t in self.transactions[lower_bound:upper_bound] if t.cumulative_weight_delayed() >= 5]

        particles = np.random.choice(candidates, num_particles)
        distances = {}
        for p in particles:
            t = threading.Thread(target=self._walk2(p))
            t.start()
#            tip, distance = self._walk(p)
#            distances[tip] = distance
            
        #return [key for key in sorted(distances, key=distances.get, reverse=False)[:2]]
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

    def plot(self):
        if hasattr(self, 'G'):
            pos = nx.get_node_attributes(self.G, 'pos')
            nx.draw_networkx_nodes(self.G, pos)
            nx.draw_networkx_labels(self.G, pos)
            nx.draw_networkx_edges(self.G, pos, edgelist=self.G.edges(), arrows=True)
            plt.xlabel('Time')
            plt.yticks([])
            plt.show()


class Transaction(object):

    def __init__(self, tangle, time, approved_transactions, num):
        #In the UTXO model the transaction is a list of inputs and outputs 
        return 0


class Genesis(Transaction):

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
