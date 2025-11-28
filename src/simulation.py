import networkx as nx
import numpy as np
import random
import copy
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np

class NetworkAttackSimulation():
    def __init__(self, 
                 graph: nx.DiGraph, 
                 type_of_attack: str="random", 
                 type_of_recovery:str = 'uniform', 
                 num_of_iter: int=100, 
                 metric: str="pagerank", 
                 random_seed: int=42,
                 recovery_scale: float=10,
                 recovery_prob: float=0.01
                 ):
        
        self.graph = graph
        self.type_of_attack = type_of_attack
        self.type_of_recovery = type_of_recovery
        self.num_of_iter = num_of_iter
        self.metric = metric
        self.recovery_scale = recovery_scale
        self.recovery_prob = recovery_prob
        self.random_seed = random_seed
        self.history:list[dict] = []
        self.change_history:list[dict] = []
        self.attack_log: list[list] = []
        self.recovery_log: list[list] = []
        self.removed_nodes: set = set()
        self.original_graph: nx.DiGraph = copy.deepcopy(self.graph)
        self.recovery_edge_probability: float = 0.001
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
        self.removed_nodes.add(node)

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
            in_degrees = dict(G.in_degree())  
            out_degrees = dict(G.out_degree())
            metrics["avg_in_degree"] = np.mean(list(in_degrees.values()))
            metrics["avg_out_degree"] = np.mean(list(out_degrees.values()))
        else:
            metrics["avg_in_degree"] = 0
            metrics["avg_out_degree"] = 0
        
        # Density
        metrics["density"] = nx.density(G)

        #strongly and weakly connected components
        sccs = list(nx.strongly_connected_components(G))
        num_scc = len(sccs)
        largest_scc_set = max(sccs, key=len)
        size_largest_scc = len(largest_scc_set)
            
        wccs = list(nx.weakly_connected_components(G))
        num_wcc = len(wccs)
        largest_wcc_set = max(wccs, key=len)
        size_largest_wcc = len(largest_wcc_set)

        metrics["num_scc"] = num_scc
        metrics["size_largest_scc"] = size_largest_scc
        metrics["num_wcc"] = num_wcc
        metrics["size_largest_wcc"] = size_largest_wcc
       
        #Centrality (betweenness, closeness, pagerank)
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        pagerank = nx.pagerank(G)
        harmonic = nx.harmonic_centrality(G)
        
        #avg centrality values
        metrics["avg_betweenness"] = np.mean(list(betweenness.values()))
        metrics["avg_closeness"] = np.mean(list(closeness.values()))
        metrics["avg_pagerank"] = np.mean(list(pagerank.values()))
        metrics["avg_harmonic"] = np.mean(list(harmonic.values()))
        
        #top 3 nodes by centrality
        # metrics["top_3_betweenness"] = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
        # metrics["top_3_closeness"] = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:3]
        # metrics["top_3_pagerank"] = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]
        # metrics["top_3_harmonic"] = sorted(harmonic.items(), key=lambda x: x[1], reverse=True)[:3]
        
        #path lengths (only if there is a strongly connected component)
        if size_largest_scc > 1:    
            largest_scc_subgraph = G.subgraph(largest_scc_set).copy()
            asp = nx.average_shortest_path_length(largest_scc_subgraph)
            diam = nx.diameter(largest_scc_subgraph)
            metrics["avg_shortest_path_length"] = asp
            metrics["diameter"] = diam
        else:
            metrics["avg_shortest_path_length"] = 0
            metrics["diameter"] = 0
            

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
        still_removed = set()
        for node in self.removed_nodes:
            if random.random() < self.recovery_prob: # removed node has a high probabibility to be brought back.
                self.graph.add_node(node)
                # if self.graph.number_of_nodes() > 1: # creating new edges
                #     target = random.choice(list(self.graph.nodes)) # randomly choooses nodes in the edge to connect to (maybe connect to higher degree nodes, centrality.)
                #     if target != node:
                #         self.graph.add_edge(node, target) # forms new edges
                recovered_this_iter.append(node)
            else:
                still_removed.add(node)
        self.removed_nodes = still_removed | self.removed_nodes
        self.recovery_log.append(recovered_this_iter)

    
  
    def weighted_recovery(self, metric):
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
        if metric == "pagerank":
            weights = nx.pagerank(Go)
        elif metric == "betweenness":
            weights = nx.betweenness_centrality(Go)
        elif metric == "closeness":
            weights = nx.closeness_centrality(Go)
        elif metric == "harmonic":
            weights = nx.harmonic_centrality(Go)
        elif metric == "in_degree":
            weights = dict(Go.in_degree())
            normalized_weights = np.array(list(weights.values()))
            normalized_weights = normalized_weights / normalized_weights.sum()
            weights = dict(zip(weights.keys(), normalized_weights))
        elif metric == "out_degree":
            weights = dict(Go.out_degree())
            normalized_weights = np.array(list(weights.values()))
            normalized_weights = normalized_weights / normalized_weights.sum()
            weights = dict(zip(weights.keys(), normalized_weights))
        else:
            raise ValueError("Invalid metric")
        
        all_removed = list(self.removed_nodes)
        #since the weights are already between 0 and 1 and they sum up to 1, we can just use them directly
        metric_probs = np.array([weights[n] for n in all_removed])

        # final probability 
        final_probs = self.recovery_scale * metric_probs
        
        # Sample which nodes return
        recovered_nodes = [
            node for node, p in zip(all_removed, final_probs)
            if np.random.rand() < p
        ]
        
        # Re-add recovered nodes
        for node in recovered_nodes:
            if not G.has_node(node):
                G.add_node(node)
            
        #------------------------This part is not yet decided---------------------------------
        
            # re-add outgoing edges from original graph
            if node in self.original_graph:
                for _, nbr, data in self.original_graph.out_edges(node, data=True):
                    if nbr in self.graph:
                        self.graph.add_edge(node, nbr, **data)
                
                # re-add incoming edges
                for nbr, _, data in self.original_graph.in_edges(node, data=True):
                    if nbr in self.graph:
                        self.graph.add_edge(nbr, node, **data)
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
                self.removed_nodes.discard(node)
        
        # Log this iteration
        self.recovery_log.append(recovered_nodes)

    def plots(self, figsize=(16, 12), save_fig_path=None):
        
        if len(self.history) == 0:
            print("No simulation data to plot")
            return
        
        iterations = range(len(self.history))
        
        fig, axes = plt.subplots(4, 3, figsize=figsize)
        fig.suptitle(f'Network Attack Simulation Results\n'
                    f'Attack: {self.type_of_attack} | Recovery: {self.type_of_recovery} | '
                    f'Metric: {self.metric}', 
                    fontsize=14, fontweight='bold')
        
        axes = axes.flatten()
        
                    #--------------- 1. Number of Nodes and Edges ---------------
        ax = axes[0]
        num_nodes = [h['num_nodes'] for h in self.history]
        num_edges = [h['num_edges'] for h in self.history]
        ax.plot(iterations, num_nodes, label='Nodes', marker='o', markersize=3, linewidth=2)
        ax.plot(iterations, num_edges, label='Edges', marker='s', markersize=3, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Count')
        ax.set_title('Network Size Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
                    #--------------- 2. Average Degree ---------------
        ax = axes[1]
        avg_in_degree = [h['avg_in_degree'] for h in self.history]
        avg_out_degree = [h['avg_out_degree'] for h in self.history]
        ax.plot(iterations, avg_in_degree, label='In-Degree', marker='o', markersize=3, linewidth=2)
        ax.plot(iterations, avg_out_degree, label='Out-Degree', marker='s', markersize=3, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Degree')
        ax.set_title('Average Node Degree')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
                    #--------------- 3. Density ---------------
        ax = axes[2]
        density = [h['density'] for h in self.history]
        ax.plot(iterations, density, color='green', marker='o', markersize=3, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Density')
        ax.set_title('Network Density')
        ax.grid(True, alpha=0.3)
        
                    #--------------- 4. Connected Components ---------------
        ax = axes[3]
        num_scc = [h['num_scc'] for h in self.history]
        num_wcc = [h['num_wcc'] for h in self.history]
        ax.plot(iterations, num_scc, label='Strongly Connected', marker='o', markersize=3, linewidth=2)
        ax.plot(iterations, num_wcc, label='Weakly Connected', marker='s', markersize=3, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of Components')
        ax.set_title('Connected Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
                    # --------------- 5. Largest Component Size ---------------
        ax = axes[4]
        size_largest_scc = [h['size_largest_scc'] for h in self.history]
        size_largest_wcc = [h['size_largest_wcc'] for h in self.history]
        ax.plot(iterations, size_largest_scc, label='Largest SCC', marker='o', markersize=3, linewidth=2)
        ax.plot(iterations, size_largest_wcc, label='Largest WCC', marker='s', markersize=3, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Size')
        ax.set_title('Largest Component Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
                    #--------------- 6. Average Centrality Measures ---------------
        ax = axes[5]
        avg_betweenness = [h['avg_betweenness'] for h in self.history]
        ax.plot(iterations, avg_betweenness, color='purple', marker='o', markersize=3, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Betweenness')
        ax.set_title('Average Betweenness Centrality')
        ax.grid(True, alpha=0.3)
        
                    #--------------- 7. Closeness Centrality ---------------
        ax = axes[6]
        avg_closeness = [h['avg_closeness'] for h in self.history]
        ax.plot(iterations, avg_closeness, color='orange', marker='o', markersize=3, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Closeness')
        ax.set_title('Average Closeness Centrality')
        ax.grid(True, alpha=0.3)
        
                    #--------------- 8. PageRank ---------------
        ax = axes[7]
        avg_pagerank = [h['avg_pagerank'] for h in self.history]
        ax.plot(iterations, avg_pagerank, color='red', marker='o', markersize=3, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average PageRank')
        ax.set_title('Average PageRank')
        ax.grid(True, alpha=0.3)
        
                    #--------------- 9. Path Length Metrics ---------------
        ax = axes[8]
        avg_shortest_path = [h['avg_shortest_path_length'] for h in self.history]
        diameter = [h['diameter'] for h in self.history]
        ax.plot(iterations, avg_shortest_path, label='Avg Path Length', marker='o', markersize=3, linewidth=2)
        ax.plot(iterations, diameter, label='Diameter', marker='s', markersize=3, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Length')
        ax.set_title('Path Length Metrics (on Largest SCC)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
                    #--------------- 10. Clustering Coefficients ---------------
        ax = axes[9]
        global_clustering = [h['global_clustering'] for h in self.history]
        avg_clustering = [h['avg_clustering'] for h in self.history]
        ax.plot(iterations, global_clustering, label='Global', marker='o', markersize=3, linewidth=2)
        ax.plot(iterations, avg_clustering, label='Average Local', marker='s', markersize=3, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Clustering Coefficient')
        ax.set_title('Clustering Coefficients')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
                    #--------------- 11. Assortativity ---------------
        ax = axes[10]
        assortativity = [h['assortativity'] for h in self.history]
        ax.plot(iterations, assortativity, color='brown', marker='o', markersize=3, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Assortativity Coefficient')
        ax.set_title('Degree Assortativity')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
                    #--------------- 12. Attack and Recovery Activity ---------------
        ax = axes[11]
        nodes_attacked = [len(log) for log in self.attack_log]
        nodes_recovered = [len(log) for log in self.recovery_log]
        ax.bar(iterations, nodes_attacked, alpha=0.7, label='Attacked', color='red')
        ax.bar(iterations, nodes_recovered, alpha=0.7, label='Recovered', color='green', bottom=0)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of Nodes')
        ax.set_title('Attack vs Recovery Activity')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_fig_path:
            plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_fig_path}")
        
        plt.show()
        
        # Summary statistics
        print("\n---Simulation Summary---")
        print(f"Initial nodes: {self.history[0]['num_nodes']}")
        print(f"Final nodes: {self.history[-1]['num_nodes']}")
        print(f"Initial edges: {self.history[0]['num_edges']}")
        print(f"Final edges: {self.history[-1]['num_edges']}")
        print(f"Total nodes attacked: {sum(len(log) for log in self.attack_log)}")
        print(f"Total nodes recovered: {sum(len(log) for log in self.recovery_log)}")
        print(f"Net nodes removed: {len(self.removed_nodes)}")
        
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
                self.targeted_attack(metric=self.metric)
            else:
                raise ValueError(f"Unknown attack type: {self.type_of_attack}")
            
        
            
            # RECOVERY
            if self.type_of_recovery == "uniform":
                self.uniform_recovery()
            elif self.type_of_recovery == "weighted":
                if self.recovery_prob is None:
                    raise ValueError("recovery_prob must be set for weighted recovery")
                self.weighted_recovery(metric=self.metric)
            
            
            
            # MEASURE
            self.measure_network_state()
            
            self.iterations_completed += 1
            # LOGGING
            print(f"Iteration {i+1}/{self.num_of_iter}: Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
            print(f"  Removed nodes so far: {len(self.removed_nodes)}")
            print(f"  Attack log this iteration: {self.attack_log[-1]}")
            print(f"  Recovery log this iteration: {self.recovery_log[-1]}")

        # PLOTS
        self.plots()