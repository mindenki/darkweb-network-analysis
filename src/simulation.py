import networkx as nx
import numpy as np
import random
import copy

class NetworkAttackSimulation():
    def __init__(self, 
                 graph: nx.DiGraph, 
                 type_of_attack: str="random", 
                 type_of_recovery:str = None, 
                 num_of_iter: int=100, 
                 metrics: list=None, 
                 recovery_prob: float=None, 
                 random_seed: int=42
                 ):
        self.graph = graph
        self.type_of_attack = type_of_attack
        self.type_of_recovery = type_of_recovery
        self.num_of_iter = num_of_iter
        self.metrics = metrics
        self.recovery_prob = recovery_prob
        self.random_seed = random_seed
        self.history = list[dict]
        self.attack_log = list[list]
        self.recovery_log = list[list]
    
    def reset():
        pass
    def random_attack():
        pass
    def targeted_attack():
        pass


    def measure_network_state(self, G):
        """
        Compute a set of structural network metrics for the current graph state.
        Returns a dictionary that can be appended to `history`.
        """
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
        
        return metrics, change

    def uniform_recovery():
        pass
    def weighted_recovery():
        pass
    def plots():
        pass
    def run():
        pass