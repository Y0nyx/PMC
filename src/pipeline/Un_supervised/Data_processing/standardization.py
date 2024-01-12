#!/user/bin/env python

"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 01/11/2024
Description: Put the data in a value between 0 and 1 for the normalization of the value will be centered at 0 when 
             standardization will be used. 
            
Note:         It might not be usufull to use standardization or normalization if layers of batch normalization are 
              used in the network based on Francois Grondin experience. (I am personnaly not convinced of that one, 
              especially if the first activation function is used before the batch normalization layer. 
              The normalization/standardization is used in part to allow the activations functions to work properly. 
              Indeed their value varie between -1 to 1 in general.I will have to ask Julie Carreau a machine learning 
              professeur, with a strong math background at UdM that help me with my current project at ECCC).
"""

#####################################################################################################################
#                                                      Imports                                                      # 
#####################################################################################################################



#####################################################################################################################
#                                                     Variables                                                     #
#####################################################################################################################



#####################################################################################################################
#                                                     Functions                                                     #
#####################################################################################################################



######################################################################################################################
#                                                        Main                                                        #
######################################################################################################################
if __name__ == '__main__':
    pass