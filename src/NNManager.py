class NNManager:
    instance = None

    def __init__(self):
        self.array_NeuralNetwork = []

    @staticmethod
    def get_instance():
        if NNManager.instance is None:
            NNManager.instance = NNManager()
        print("Hello get_instance")
        return NNManager.instance

    def predicts(self, features):
        if not self.array_NeuralNetwork:
            return None
        return self.array_NeuralNetwork[0].predict(features)           # predict() est un placeholder pour la vraie method

    def predicts(self, index_NN, features):
        if index_NN < 0 or index_NN >= len(self.array_NeuralNetwork):
            return None
        return self.array_NeuralNetwork[index_NN].predict(features)    # predict() est un placeholder pour la vraie method

    def trains(self):
        for nn in self.array_NeuralNetwork:                            # Assumption que plusieurs NN seront entraînés en même temps
            nn.train()                                                 # train() est un placeholder pour la vraie method

    def trains(self, index_NN):
        if index_NN < 0 or index_NN >= len(self.array_NeuralNetwork):
            return
        self.array_NeuralNetwork[index_NN].train()                     # train() est un placeholder pour la vraie method

    def get_state(self):
        if NNManager.instance is None:
            return None
        return self.NeuralNetworkState                                 # self.NeuralNetworkState est un placeholder pour le vrai enum