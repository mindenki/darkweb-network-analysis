import networkx as nx
import numpy as np
import random
import copy
from typing import Literal

class NetworkAttackSimulation():
    def __init__(self, 
                 graph: nx.DiGraph, 
                 type_of_attack: str="random", 
                 type_of_recovery:str = None, 
                 num_of_iter: int=100, 
                 #metrics: list=None, 
                 recovery_prob: float=.01, 
                 random_seed: int=42
                 ):
        self.graph = graph
        self.type_of_attack = type_of_attack
        self.type_of_recovery = type_of_recovery
        self.num_of_iter = num_of_iter
        #self.metrics = metrics
        self.recovery_prob = recovery_prob
        self.random_seed = random_seed
        self.history:list[dict] = []
        self.change_history:list[dict] = []
        self.attack_log: list[list] = []
        self.recovery_log: list[list] = []
        self.removed_nodes: set = set()
        self.original_graph: nx.DiGraph = copy.deepcopy(self.graph)
        self.recovery_edge_probability: int = 0.001
        self.iterations_completed: int = 0
        np.random.seed(self.random_seed)
    
    def reset(self):
        """ Reset the simulation to its initial state."""
        # reset the graph
        self.graph = copy.deepcopy(self.original_graph)
        
        # clear the logs
        self.history = []
        self.change_history = []
        self.attack_log = []
        self.recovery_log = []
        self.removed_nodes = set()
        
        # reseed just in case
        np.random.seed(self.random_seed)
        
    def random_attack(self):
        nodes = list(self.graph.nodes) 
        if not nodes:
            return
        node = random.choice(nodes)
        self.graph.remove_node(node)
        self.attack_log.append([node])
        self.removed_nodes.append(node)

    def targeted_attack(self, metric: Literal["betweenness", "closeness", "in_degree", "out_degree", "pagerank", "harmonic"] = 'harmonic', num_nodes_to_remove: int = 1):
        
        G = self.graph
        
        if G.number_of_nodes() == 0:
            self.attack_log.append([])
            return
        
        dict_metric = {
            "betweenness": nx.betweenness_centrality,
            "closeness": nx.closeness_centrality,
            "pagerank": nx.pagerank,
            "in_degree": dict(G.in_degree()),
            "out_degree": dict(G.out_degree()),
            "harmonic": nx.harmonic_centrality
        }
        
        if metric not in dict_metric:
            raise ValueError(f"Invalid metric: {metric}. Choose from: {list(dict_metric.keys())}")
        
        metric_obj = dict_metric[metric]
        
        if callable(metric_obj):
            centrality = metric_obj(G)
        else:
            centrality = metric_obj
        
        if not isinstance(centrality, dict):
            raise TypeError(f"Centrality computation failed. Expected dict, got {type(centrality)}")
        
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        nodes_to_remove = [node for node, _ in sorted_nodes[:num_nodes_to_remove]]
        
        for node in nodes_to_remove:
            if G.has_node(node):
                G.remove_node(node)
                self.removed_nodes.add(node)
        
        self.attack_log.append(nodes_to_remove)
        
        return

    def measure_network_state(self):
        """
        Compute a set of structural network metrics for the current graph state.
        Returns a dictionary that can be appended to `history`.
        """
        G = self.graph
        metrics = {}
        change = {}

        # Average in and out degree
        metrics["num_nodes"] = G.number_of_nodes()
        metrics["num_edges"] = G.number_of_edges()

        if metrics["num_nodes"] > 0:
            in_degrees = [d for d in G.in_degree()]
            out_degrees = [d for d in G.out_degree()]
            metrics["avg_in_degree"] = np.mean(in_degrees)
            metrics["avg_out_degree"] = np.mean(out_degrees)
        else:
            metrics["avg_in_degree"] = 0
            metrics["avg_out_degree"] = 0
        
        # Density
        metrics["density"] = nx.density(G)

        #strongly and weakly connected components
        sccs = list(nx.strongly_connected_components(G))
        num_scc = len(sccs)
        largest_scc = max(sccs, key=len)
        size_largest_scc = len(largest_scc)
            
        wccs = list(nx.weakly_connected_components(G))
        num_wcc = len(wccs)
        largest_wcc = max(wccs, key=len)
        size_largest_wcc = len(largest_wcc)

        metrics["num_scc"] = num_scc
        metrics["size_largest_scc"] = size_largest_scc
        metrics["num_wcc"] = num_wcc
        metrics["size_largest_wcc"] = size_largest_wcc
       
        #Centrality (betweenness, closeness, pagerank)
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        pagerank = nx.pagerank(G)
        
        #avg centrality values
        metrics["avg_betweenness"] = np.mean(list(betweenness.values()))
        metrics["avg_closeness"] = np.mean(list(closeness.values()))
        metrics["avg_pagerank"] = np.mean(list(pagerank.values()))
        
        #top 3 nodes by centrality
        # metrics["top_3_betweenness"] = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
        # metrics["top_3_closeness"] = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:3]
        # metrics["top_3_pagerank"] = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]
        
        #path lengths
        largest_scc = G.subgraph(largest_scc).copy()
        asp = nx.average_shortest_path_length(largest_scc)
        diam = nx.diameter(largest_scc)
        metrics["avg_shortest_path_length"] = asp
        metrics["diameter"] = diam

        # Clustering
        global_clustering = nx.transitivity(G) # global cluserting
        avg_clustering = nx.average_clustering(G)   # average local clustering
        metrics["global_clustering"] = global_clustering
        metrics["avg_clustering"] = avg_clustering
        
        # Assortativity
        assortativity = nx.degree_assortativity_coefficient(G)
        metrics["assortativity"] = assortativity
        
        #Change compared to last state
        if len(self.history) > 0:
            last_metrics = self.history[-1]
            for key in metrics.keys():
                change_key = f"change_in_{key}"
                if last_metrics[key] != 0:
                    change[change_key] = (metrics[key] - last_metrics[key]) / abs(last_metrics[key])
                else:
                    change[change_key] = 0
        else:
            for key in metrics.keys():
                change_key = f"change_in_{key}"
                change[change_key] = 0
                    
        self.history.append(metrics)
        self.change_history.append(change)
        
        return

    def uniform_recovery(self):
        recovered_this_iter = []
        still_removed = []
        for node in self.removed_nodes:
            if random.random() < self.recovery_prob: # removed node has a high probabibility to be brought back.
                self.graph.add_node(node)
                if self.graph.number_of_nodes() > 1: # creating new edges
                    target = random.choice(list(self.graph.nodes)) # randomly choooses nodes in the edge to connect to (maybe connect to higher degree nodes, centrality.)
                    if target != node:
                        self.graph.add_edge(node, target) # forms new edges
                recovered_this_iter.append(node)
            else:
                still_removed.append(node)
        self.removed_nodes = still_removed
        self.recovery_log.append(recovered_this_iter)

    
  
    def weighted_recovery(self, comeback_probability, metric_of_choice):
        """
        Perform weighted recovery of removed nodes.
        Each removed node has a comeback probability weighted by a centrality metric.

        """
        G = self.graph
        Go = self.original_graph
        
        # No removed nodes → nothing to do
        if len(self.removed_nodes) == 0:
            self.recovery_log.append([])
            return
        
        # Compute metric weights
        if metric_of_choice == "pagerank":
            weights = nx.pagerank(Go)
        elif metric_of_choice == "betweenness":
            weights = nx.betweenness_centrality(Go)
        elif metric_of_choice == "closeness":
            weights = nx.closeness_centrality(Go)
        elif metric_of_choice == "degree":
            weights = dict(Go.degree())
        else:
            raise ValueError("Invalid metric_of_choice")
        
        # Normalize weights so they sum to 1
        all_removed = list(self.removed_nodes)
        metric_values = np.array([weights[n] for n in all_removed])
        total = metric_values.sum()
        metric_probs = metric_values / total

        # final probability 
        final_probs = comeback_probability * metric_probs
        
        # Sample which nodes return
        recovered_nodes = [
            node for node, p in zip(all_removed, final_probs)
            if np.random.rand() < p
        ]
        
        # Re-add recovered nodes
        for node in recovered_nodes:
            if not self.G.has_node(node):
                self.G.add_node(node)
            
        #------------------------This part is not yet decided---------------------------------
        
            # re-add outgoing edges from original graph
            # if node in self.original_G:
            #     for _, nbr, data in self.original_G.out_edges(node, data=True):
            #         if nbr in self.G:
            #             self.G.add_edge(node, nbr, **data)
                
            #     # re-add incoming edges
            #     for nbr, _, data in self.original_G.in_edges(node, data=True):
            #         if nbr in self.G:
            #             self.G.add_edge(nbr, node, **data)
        #-------------------------------------------------------------------------------------              
            
            # Add random edges uniformly with a small probability
                # edge_prob = self.recovery_edge_probability 
                # possible_neighbors = [n for n in self.G.nodes if n != node]

                # # 1) Random OUTGOING edges:
                # for nbr in possible_neighbors:
                #     if np.random.rand() < edge_prob:
                #         self.G.add_edge(node, nbr)

                # # 2) Random INCOMING edges:
                # for nbr in possible_neighbors:
                #     if np.random.rand() < edge_prob:
                #         self.G.add_edge(nbr, node)
        #-------------------------------------------------------------------------------------
                # remove from removed set
                self.removed_nodes.remove(node)
        
        # Log this iteration
        self.recovery_log.append(recovered_nodes)

    def plots(self):
        pass
    def run(self, num_of_iter: int=None):
        """ Run the simulation over the specified number of iterations. 
        At each iteration, perform an attack, measure the network state, and optionally perform recovery.
        """
        
        
        self.reset() # maybe not needed
        
        
        if num_of_iter is not None:
            self.num_of_iter = num_of_iter
            
        
        # main loop
        for i in range(self.num_of_iter):
            
            
            # ATTACK
            if self.type_of_attack == "random":
                self.random_attack()
            elif self.type_of_attack == "targeted":
                self.targeted_attack()
            else:
                raise ValueError(f"Unknown attack type: {self.type_of_attack}")
            
        
            
            # RECOVERY
            if self.type_of_recovery == "uniform":
                self.uniform_recovery()
            elif self.type_of_recovery == "weighted":
                if self.recovery_prob is None:
                    raise ValueError("recovery_prob must be set for weighted recovery")
                self.weighted_recovery(comeback_probability=self.recovery_prob, metric_of_choice="pagerank")
            
            
            
            # MEASURE
            self.measure_network_state()
            
            self.iterations_completed += 1
            # LOGGING
            print(f"Iteration {i+1}/{self.num_of_iter}: Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
            print(f"  Removed nodes so far: {len(self.removed_nodes)}")
            print(f"  Attack log this iteration: {self.attack_log[-1]}")
            print(f"  Recovery log this iteration: {self.recovery_log[-1]}")