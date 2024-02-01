from pipeline.feature.featureExtractionManager import FeatureExtractionManager


class NNManager:
    instance = None

    def __init__(self):
        self.array_NeuralNetwork = []
        self._featureExtractionManager = FeatureExtractionManager.get_instance()

    @staticmethod
    def get_instance():
        if NNManager.instance is None:
            NNManager.instance = NNManager()
        return NNManager.instance

    def predicts(self):
        features = self._featureExtractionManager.get_all_features()
        if not self.array_NeuralNetwork:
            return None
        # return self.array_NeuralNetwork[0].predict(features)           # predict() est un placeholder pour la vraie method
        return features

    def predicts_index(self, index_NN):
        features = self._featureExtractionManager.get_all_features(index_NN)
        if index_NN < 0 or index_NN >= len(self.array_NeuralNetwork):
            return None
        return self.array_NeuralNetwork[index_NN].predict(
            features
        )  # predict() est un placeholder pour la vraie method

    def trains(self):
        for (
            nn
        ) in (
            self.array_NeuralNetwork
        ):  # Assumption que plusieurs NN seront entraînés en même temps
            nn.train()  # train() est un placeholder pour la vraie method

    def trains_index(self, index_NN):
        if index_NN < 0 or index_NN >= len(self.array_NeuralNetwork):
            return
        self.array_NeuralNetwork[
            index_NN
        ].train()  # train() est un placeholder pour la vraie method

    def get_state(self):
        if NNManager.instance is None:
            return None
        return (
            self.NeuralNetworkState
        )  # self.NeuralNetworkState est un placeholder pour le vrai enum
    
    def add_models(self, models):
        if not isinstance(models, list):
            self.array_NeuralNetwork.append(models)
        else:
            for model in models:
                self.array_NeuralNetwork.append(model)

    def predict_imgs(self):
        Images = self._Da