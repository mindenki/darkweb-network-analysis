class NetworkAttackSimulation():
    def __init__(self, graph, typeofattack="random", typeofrecovery=None, num_of_iter=100, metrics=None, recovery_prob=None, random_seed=42, history=None, attack_log=None, recovery_log=None):
        self.graph = graph
        self.typeofattack = typeofattack
        self.typeofrecovery = typeofrecovery
        self.num_of_iter = num_of_iter
        self.metrics = metrics
        self.recovery_prob = recovery_prob
        self.random_seed = random_seed
        self.history = history
        self.attack_log = attack_log
        self.recovery_log = recovery_log
        
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