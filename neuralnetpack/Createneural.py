import numpy as np
 
from .neural import NeuralNetwork

from .Decision import Decisiontree

class Createneural(NeuralNetwork):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        
        
    def generate(self):
        nn = NeuralNetwork(self.train_x, self.train_y)

        for i in range(1500):
            nn.feedforward()
            nn.backprop()
        print(nn.output.mean())
        return nn.output
    
    def neural_score(self):
        nn_score = self.generate().mean()
        
    def generate_dt(self):
        dt = Decisiontree(max_depth = 5)
        dt.fit(self.train_x, self.train_y)
        print('Decision Tree is now ready!')
        
    def dt_score(self):
        pass
    
                                
            