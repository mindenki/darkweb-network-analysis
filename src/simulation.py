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
    def eval_metrics_and_metrics_change():
        pass
    def uniform_recovery():
        pass
    def weighted_recovery():
        pass
    def plots():
        pass
    def run():
        pass