import networkx as nx
import random
import copy
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from tqdm import tqdm
from datetime import datetime

logger = logging.getLogger("NetworkAttackSimulation")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

class NetworkAttackSimulation():

    def __init__(self, 
                 graph: nx.DiGraph, 
                 type_of_attack: str="random", 
                 type_of_recovery:str = 'uniform', 
                 num_of_iter: int=100, 
                 metric: str="pagerank", 
                 random_seed: int= None,
                 recovery_scale: float=10,
                 recovery_prob: float=0.01,
                 recovery_edge_type: str="realistic",
                 recovery_edge_probability: float=0.001,
                 recovery_interval: int = 1,
                 metric_interval: int = 1,
                 ):
        #graph
        self.graph = graph
        self.original_graph: nx.DiGraph = copy.deepcopy(self.graph)
        # attack
        self.type_of_attack = type_of_attack
        # recovery
        self.type_of_recovery = type_of_recovery
        self.recovery_scale = recovery_scale
        self.recovery_prob = recovery_prob
        self.recovery_edge_type: str = recovery_edge_type
        self.recovery_edge_probability: float = recovery_edge_probability
        self.recovery_interval: int = recovery_interval
        
        # logs
        self.history:list[dict] = []
        self.change_history:list[dict] = []
        self.attack_log: list[dict] = []
        self.recovery_log: list[dict] = []
        self.removed_nodes: set = set()
        
        # misc
        self.num_of_iter = num_of_iter
        self.metric = metric
        self.metric_interval = metric_interval
        self.random_seed = random_seed or random.randint(0, 100000)
        self.iterations_completed: int = 0
        self.last_pagerank = None
        self.original_pagerank = nx.pagerank(self.original_graph)
        
        # set random seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
    
    def reset(self):
        """ Reset the simulation to its initial state."""
        # reset the graph
        self.graph = copy.deepcopy(self.original_graph)
        
        # clear the logs
        self.history.clear()
        self.change_history.clear()
        self.attack_log.clear()
        self.recovery_log.clear()
        self.removed_nodes.clear()
        self.last_pagerank = None

        # reseed just in case
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        logger.info("Simulation reset to initial state.")
    
    def get_reactive_pagerank(self):
        """Computes PageRank reactively using the previous state as a starting point."""
        if not self.graph.nodes:
            return {}
        
        # Filter the previous PageRank to only include nodes still in the graph
        if self.last_pagerank is not None:
            self.last_pagerank = {k: v for k, v in self.last_pagerank.items() if k in self.graph}
        
        self.last_pagerank = nx.pagerank(
            self.graph, 
            nstart=self.last_pagerank, 
            max_iter=100, 
            tol=1e-06
        )
        return self.last_pagerank
        
    
    # -----ATTACKS-----
    
    def random_attack(self):
        nodes = list(self.graph.nodes) 
        if not nodes:
            self.attack_log.append({"iteration": self.iterations_completed, "nodes": []})
            return
        node = random.choice(nodes)
        self.graph.remove_node(node)
        self.removed_nodes.add(node)
        
        self.attack_log.append({"iteration": self.iterations_completed, "nodes": [node]})
        logger.debug("Random attack removed node: %s", node)
        

    def targeted_attack(self, metric='pagerank', num_nodes_to_remove=1):
        G = self.graph
        if G.number_of_nodes() == 0:
            self.attack_log.append({"iteration": self.iterations_completed, "nodes": []})
            return

        if metric == "pagerank":
            centrality = self.get_reactive_pagerank()
        else:
            dict_metric = {
                "betweenness": lambda: nx.betweenness_centrality(G, k=min(len(G), 100)), # Sampled (it would be too slow without)
                "closeness": nx.closeness_centrality,
                "in_degree": lambda: dict(G.in_degree()),
                "out_degree": lambda: dict(G.out_degree()),
                "harmonic": nx.harmonic_centrality
            }
            metric_obj = dict_metric[metric]
            centrality = metric_obj() if callable(metric_obj) else metric_obj       
        
        if not isinstance(centrality, dict):
            raise TypeError(f"Centrality computation failed. Expected dict, got {type(centrality)}")
        
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        nodes_to_remove = [node for node, _ in sorted_nodes[:num_nodes_to_remove]]
        
        for node in nodes_to_remove:
            if G.has_node(node):
                G.remove_node(node)
                self.removed_nodes.add(node)
                if self.last_pagerank: # Remove from cache
                    self.last_pagerank.pop(node, None)
        
        self.attack_log.append({"iteration": self.iterations_completed, "nodes": nodes_to_remove})
        logger.debug("Targeted attack removed nodes %s by metric %s", nodes_to_remove, metric)
        
        return   

    # -----RECOVERY-----
    
    def recover_edges(self, node, recover_type):
        """ Recover edges for a given node based on different strategies."""
        
        recovered_in_edges = []
        recovered_out_edges = []
        
        if recover_type not in ["realistic", "original", "random", "none"]:
            raise ValueError("Invalid recover_type. Choose from 'realistic', 'original', 'random', 'none'.")
        
        if recover_type == "realistic":
            # add all outgoing edges from original graph and some random incoming edges
            if node in self.original_graph:
                for _, nbr in self.original_graph.out_edges(node):
                    if nbr in self.graph:
                        self.graph.add_edge(node, nbr)
                        recovered_out_edges.append((node, nbr))
                        
                # add some random incoming edges from the original graph
                for _, nbr in self.original_graph.in_edges(node):
                    if nbr in self.graph:
                        # 10% chance to readd each incoming edge
                        if np.random.rand() < 0.1:
                            self.graph.add_edge(nbr, node)
                            recovered_in_edges.append((nbr, node))
        elif recover_type == "original":
            # re-add all edges from original graph
            if node in self.original_graph:
                for _, nbr in self.original_graph.out_edges(node):
                    if nbr in self.graph:
                        self.graph.add_edge(node, nbr)
                        recovered_out_edges.append((node, nbr))
                
                for nbr, _ in self.original_graph.in_edges(node):
                    if nbr in self.graph:
                        self.graph.add_edge(nbr, node)
                        recovered_in_edges.append((nbr, node))            
                        
        elif recover_type == "random":
            possible_neighbors = list(self.graph.nodes)
            if not possible_neighbors: return

            n_out = np.random.binomial(len(possible_neighbors), self.recovery_edge_probability)
            n_in = np.random.binomial(len(possible_neighbors), self.recovery_edge_probability)

            if n_out > 0:
                targets = random.sample(possible_neighbors, min(n_out, len(possible_neighbors)))
                self.graph.add_edges_from([(node, t) for t in targets])
            if n_in > 0:
                sources = random.sample(possible_neighbors, min(n_in, len(possible_neighbors)))
                self.graph.add_edges_from([(s, node) for s in sources])
        
        logger.debug("Recovered %d outgoing edges and %d incoming edges for node %s using %s",
                     len(recovered_out_edges), len(recovered_in_edges), node, recover_type)
            

    def uniform_recovery(self):
        recovered_this_iter = []
        still_removed = set()
        for node in self.removed_nodes:
            if random.random() < self.recovery_prob: # removed node has a probabibility to be brought back.
                self.graph.add_node(node)
                self.recover_edges(node, self.recovery_edge_type)
                recovered_this_iter.append(node)
            else:
                still_removed.add(node)
        self.removed_nodes = still_removed
        self.recovery_log.append({"iteration": self.iterations_completed, "nodes": recovered_this_iter})
        logger.debug("Uniform recovery recovered nodes %s", recovered_this_iter)
        
        
    def weighted_recovery(self, metric):
        """
        Perform weighted recovery of removed nodes.
        Each removed node has a comeback probability weighted by a centrality metric.

        """
        G = self.graph
        Go = self.original_graph
        
        # No removed nodes → nothing to do
        if len(self.removed_nodes) == 0:
            self.recovery_log.append({"iteration": self.iterations_completed, "nodes": []})
            return
        
        # Compute metric weights
        if metric == "pagerank":
            weights = self.original_pagerank
        elif metric == "betweenness":
            weights = nx.betweenness_centrality(Go)
        elif metric == "closeness":
            weights = nx.closeness_centrality(Go)
        elif metric == "harmonic":
            weights = nx.harmonic_centrality(Go)
        elif metric == "in_degree":
            weights = dict(Go.in_degree())
        elif metric == "out_degree":
            weights = dict(Go.out_degree())
        else:
            raise ValueError("Invalid metric")
        
        #normalization
        normalized_weights = np.array(list(weights.values()))
        normalized_weights = normalized_weights / normalized_weights.sum()
        weights = dict(zip(weights.keys(), normalized_weights))
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
                self.recover_edges(node, self.recovery_edge_type)
            self.removed_nodes.discard(node)
        
        # Log this iteration
        self.recovery_log.append({"iteration": self.iterations_completed, "nodes": recovered_nodes})
        logger.debug("Weighted recovery recovered nodes %s", recovered_nodes)

   
    # ----MEASUREMENT AND PLOTTING-----
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
        pagerank = self.get_reactive_pagerank()
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

  

    def plots(self, figsize=(10, 6), save_fig_path='../results/sim_results'):
        
        if len(self.history) == 0:
            print("No simulation data to plot")
            return
        
        num_snapshots = len(self.history)
        
        # We assume the first snapshot is iteration 0, last snapshot is the final iteration
        iterations = [
            round(1 + i * (self.num_of_iter - 1) / (num_snapshots - 1)) for i in range(num_snapshots)
            ]
        
        if not os.path.exists(save_fig_path):
            raise FileNotFoundError(f"The folder '{save_fig_path}' does not exist. Cannot save figures.")
        
        # Unique subfolder name based on time + parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{self.type_of_attack}_{self.type_of_recovery}.{self.recovery_edge_type}"
        
        # Update the save path to be inside this new folder
        full_save_path = os.path.join(save_fig_path, folder_name)

        # Create this specific directory
        if not os.path.exists(full_save_path):
            os.makedirs(full_save_path, exist_ok=True)
            print(f"Created new directory for this run: {full_save_path}")

        
        #--------------- 1. Number of Nodes ---------------
        fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
        num_nodes = [h['num_nodes'] for h in self.history]
        
        ax1.plot(iterations, num_nodes, color='blue', marker='o', markersize=4, linewidth=2)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Number of Nodes', fontsize=12)
        ax1.set_title(f'Number of Nodes Over Time\nAttack: {self.type_of_attack} | Recovery: {self.type_of_recovery}', 
                    fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if full_save_path:
            fig1.savefig(os.path.join(full_save_path, "nodes.png"), dpi=300, bbox_inches='tight')

        #--------------- 2. Number of Edges ---------------
        fig2, ax2 = plt.subplots(1, 1, figsize=figsize)
        num_edges = [h['num_edges'] for h in self.history]
        
        ax2.plot(iterations, num_edges, color='red', marker='s', markersize=4, linewidth=2)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Number of Edges', fontsize=12)
        ax2.set_title(f'Number of Edges Over Time\nAttack: {self.type_of_attack} | Recovery: {self.type_of_recovery}', 
                    fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
        if full_save_path:
            fig2.savefig(os.path.join(full_save_path, "edges.png"), dpi=300, bbox_inches='tight')

        
        #--------------- 3. Density ---------------
        fig3, ax3 = plt.subplots(1, 1, figsize=figsize)
        density = [h['density'] for h in self.history]
        ax3.plot(iterations, density, color='green', marker='o', markersize=4, linewidth=2)
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title(f'Network Density\nAttack: {self.type_of_attack} | Recovery: {self.type_of_recovery}', 
                    fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if full_save_path:
            fig3.savefig(os.path.join(full_save_path, "density.png"), dpi=300, bbox_inches='tight')

        
        # --------------- 4. Largest Component Size ---------------
        fig4, ax4 = plt.subplots(1, 1, figsize=figsize)
        size_largest_scc = [h['size_largest_scc'] for h in self.history]
        size_largest_wcc = [h['size_largest_wcc'] for h in self.history]
    
        ax4.plot(iterations, size_largest_scc, label='Largest SCC', marker='o', markersize=4, linewidth=2)
        ax4.plot(iterations, size_largest_wcc, label='Largest WCC', marker='s', markersize=4, linewidth=2)
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('Size', fontsize=12)
        ax4.set_title(f'Largest Component Size\nAttack: {self.type_of_attack} | Recovery: {self.type_of_recovery}', 
                    fontsize=13, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
    
        plt.tight_layout()
        
        if full_save_path:
            fig4.savefig(os.path.join(full_save_path, "components.png"), dpi=300, bbox_inches='tight')

        
        #--------------- 5. Centrality Measures ---------------
        fig5, axes5 = plt.subplots(2, 2, figsize=(figsize[0]*1.5, figsize[1]*1.5))
        fig5.suptitle(f'Average Centrality Measures\nAttack: {self.type_of_attack} | Recovery: {self.type_of_recovery}', 
                    fontsize=14, fontweight='bold')
        axes5 = axes5.flatten()
        
        avg_betweenness = [h['avg_betweenness'] for h in self.history]
        avg_closeness = [h['avg_closeness'] for h in self.history]
        avg_pagerank = [h['avg_pagerank'] for h in self.history]
        avg_harmonic = [h['avg_harmonic'] for h in self.history]
        
        # Betweenness
        axes5[0].plot(iterations, avg_betweenness, color='purple', marker='o', markersize=4, linewidth=2)
        axes5[0].set_xlabel('Iteration', fontsize=11)
        axes5[0].set_ylabel('Average Betweenness', fontsize=11)
        axes5[0].set_title('Betweenness Centrality', fontsize=12)
        axes5[0].grid(True, alpha=0.3)
        
        # Closeness
        axes5[1].plot(iterations, avg_closeness, color='orange', marker='s', markersize=4, linewidth=2)
        axes5[1].set_xlabel('Iteration', fontsize=11)
        axes5[1].set_ylabel('Average Closeness', fontsize=11)
        axes5[1].set_title('Closeness Centrality', fontsize=12)
        axes5[1].grid(True, alpha=0.3)
        
        # PageRank
        axes5[2].plot(iterations, avg_pagerank, color='red', marker='^', markersize=4, linewidth=2)
        axes5[2].set_xlabel('Iteration', fontsize=11)
        axes5[2].set_ylabel('Average PageRank', fontsize=11)
        axes5[2].set_title('PageRank Centrality', fontsize=12)
        axes5[2].grid(True, alpha=0.3)
        
        # Harmonic
        axes5[3].plot(iterations, avg_harmonic, color='green', marker='d', markersize=4, linewidth=2)
        axes5[3].set_xlabel('Iteration', fontsize=11)
        axes5[3].set_ylabel('Average Harmonic', fontsize=11)
        axes5[3].set_title('Harmonic Centrality', fontsize=12)
        axes5[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if full_save_path:
            fig5.savefig(os.path.join(full_save_path, "centrality.png"), dpi=300, bbox_inches='tight')


        #--------------- 6. Path Length Metrics ---------------
        fig6, ax6 = plt.subplots(1, 1, figsize=figsize)
        avg_shortest_path = [h['avg_shortest_path_length'] for h in self.history]
        diameter = [h['diameter'] for h in self.history]
        
        ax6.plot(iterations, avg_shortest_path, label='Avg Path Length', marker='o', markersize=4, linewidth=2)
        ax6.plot(iterations, diameter, label='Diameter', marker='s', markersize=4, linewidth=2)
        ax6.set_xlabel('Iteration', fontsize=12)
        ax6.set_ylabel('Length', fontsize=12)
        ax6.set_title(f'Path Length Metrics (on Largest SCC)\nAttack: {self.type_of_attack} | Recovery: {self.type_of_recovery}', 
                    fontsize=13, fontweight='bold')
        ax6.legend(fontsize=11)
        ax6.grid(True, alpha=0.3)
    
        plt.tight_layout()
        
        if full_save_path:
            fig6.savefig(os.path.join(full_save_path, "path_length.png"), dpi=300, bbox_inches='tight')


        #--------------- 7. Clustering Coefficients ---------------
        fig7, ax7 = plt.subplots(1, 1, figsize=figsize)
        global_clustering = [h['global_clustering'] for h in self.history]
        avg_clustering = [h['avg_clustering'] for h in self.history]
        
        ax7.plot(iterations, global_clustering, label='Global', marker='o', markersize=4, linewidth=2)
        ax7.plot(iterations, avg_clustering, label='Average Local', marker='s', markersize=4, linewidth=2)
        ax7.set_xlabel('Iteration', fontsize=12)
        ax7.set_ylabel('Clustering Coefficient', fontsize=12)
        ax7.set_title(f'Clustering Coefficients\nAttack: {self.type_of_attack} | Recovery: {self.type_of_recovery}', 
                    fontsize=13, fontweight='bold')
        ax7.legend(fontsize=11)
        ax7.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if full_save_path:
            fig7.savefig(os.path.join(full_save_path, "clustering.png"), dpi=300, bbox_inches='tight')


        #--------------- 8. Attack and Recovery Activity ---------------
        fig8, ax8 = plt.subplots(1, 1, figsize=figsize)
        
        num_snapshots = len(self.history)   
        metric_interval = max(1, self.num_of_iter // num_snapshots)

        attacked_per_snapshot = []
        recovered_per_snapshot = []

        for i in range(num_snapshots):
            start_iter = i * metric_interval
            end_iter = (i + 1) * metric_interval
            
            attacked = sum(
                len(log["nodes"])
                for log in self.attack_log
                if start_iter <= log.get("iteration", 0) < end_iter
            )
            
            recovered = sum(
                len(log["nodes"])
                for log in self.recovery_log
                if start_iter <= log.get("iteration", 0) < end_iter
            )
            
            attacked_per_snapshot.append(attacked)
            recovered_per_snapshot.append(recovered)

        ax8.bar(iterations, attacked_per_snapshot, alpha=0.7, label='Attacked', color='red')
        ax8.bar(iterations, recovered_per_snapshot, alpha=0.7, label='Recovered', color='green', bottom=0)
        ax8.set_xlabel('Iteration', fontsize=12)
        ax8.set_ylabel('Number of Nodes', fontsize=12)
        ax8.set_title(f'Attack vs Recovery Activity\nAttack: {self.type_of_attack} | Recovery: {self.type_of_recovery}',
                    fontsize=13, fontweight='bold')
        ax8.legend(fontsize=11)
        ax8.grid(True, alpha=0.3, axis='y')
    
        plt.tight_layout()

        if full_save_path:
            fig8.savefig(os.path.join(full_save_path, "activity.png"), dpi=300, bbox_inches='tight')
        
        print(f"\nAll figures saved in directory: {full_save_path}")

        
        plt.show()

        # Summary statistics
        print("\n---Simulation Summary---")
        print(f"Initial nodes: {self.history[0]['num_nodes']}")
        print(f"Final nodes: {self.history[-1]['num_nodes']}")
        print(f"Initial edges: {self.history[0]['num_edges']}")
        print(f"Final edges: {self.history[-1]['num_edges']}")
        print(f"Total nodes attacked: {sum(len(log['nodes']) for log in self.attack_log)}")
        print(f"Total nodes recovered: {sum(len(log['nodes']) for log in self.recovery_log)}")
        print(f"Net nodes removed: {len(self.removed_nodes)}")
    
    #-----LOGGING AND RUNNING SIMULATION-----
    
    def log_iteration_summary(self):
        """ Log a summary of the current iteration's state."""
        recover_bool = 0
        if (self.iterations_completed - 1) % self.recovery_interval == 0:
            recover_bool = 1
        logger.info(
            "Iteration %d | nodes=%d edges=%d removed=%d attacked=%d recovered=%d",
            self.iterations_completed,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            len(self.removed_nodes),
            len(self.attack_log[-1]["nodes"]) if self.attack_log else 0,
            len(self.recovery_log[-1]["nodes"]) if self.recovery_log and recover_bool else 0
        )
        
        
    def run(self):
        """ Run the simulation over the specified number of iterations. 
        At each iteration, perform an attack, measure the network state, and optionally perform recovery.
        """
        
        self.reset() 
        
        # main loop
        for i in tqdm(range(self.num_of_iter), desc="Running network simulation", unit="iter",smoothing=0.1):

            self.iterations_completed = i + 1
            logger.info("Starting iteration %d/%d...", self.iterations_completed, self.num_of_iter)

            # ATTACK
            if self.type_of_attack == "random":
                self.random_attack()
            elif self.type_of_attack == "targeted":
                self.targeted_attack(metric=self.metric)
            else:
                raise ValueError(f"Unknown attack type: {self.type_of_attack}")
            
        
            # RECOVERY
            if i % self.recovery_interval == 0:
                if self.type_of_recovery == "uniform":
                    self.uniform_recovery()
                elif self.type_of_recovery == "weighted":
                    self.weighted_recovery(metric=self.metric)
                elif self.type_of_recovery == "none":
                    self.recovery_log.append({"iteration": i, "nodes": []})
                else:
                    raise ValueError(f"Unknown recovery type: {self.type_of_recovery}")
            
            # MEASURE
            if i % self.metric_interval == 0:
                 self.measure_network_state()
    
            # LOGGING
            self.log_iteration_summary()


        logger.info("Simulation completed after %d iterations.", self.num_of_iter)
        logger.info("Final network state: nodes=%d edges=%d removed_nodes=%d removed_edges=%d", 
                    self.graph.number_of_nodes(), self.graph.number_of_edges(), 
                    self.original_graph.number_of_nodes()-self.graph.number_of_nodes(), 
                    self.original_graph.number_of_edges()-self.graph.number_of_edges(),)
        # PLOTS
        self.plots()
        
        