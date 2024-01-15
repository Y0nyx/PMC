#!/user/bin/env python

"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 01/11/2024
Description: Select channels that will be used in the Neural Network. The channels could be the choosen camera
             where the analysed piece have been taken in picture, it could be the RGB channels and eventually 
             it could be the selected lightning color. It is interesting to note that at anypoint during the
             project, other sensors could be used. This sensor could be and implement as new channels. 

Note:        At the moment it will be necessary that the selected channels are all related to the same angle of 
             of the analysed sensor. Indeed it is only possible to take information of the same sensors of the 
             analysed piece. For exemple, when analysing cameras1, the channels could be :
             - Picture of camera1 (RGB color)
             - Other lightning color
             - Other sensors of the same angle of camera1 (Based on Francois Grondin opinion)

Next itteration: I belive it is interesting to test what will happen if we test the five sensors at the same time
                 for the camera and use a 3D filter on it. Nothing prevent us from doing it and it do not add more
                 calculation. Also it is not commounly use in the litterature so it could be a nice add to.  
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
