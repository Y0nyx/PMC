#!/user/bin/env python

"""
Author: Jean-Sebastien Giroux
Contributor(s): 
Date: 01/11/2024
Description: Training the Neural Network, which is, in this case a GAN model. A GAN model is composed of a Generator
             and a Discriminator. The role of the generator is to generate images from random numbers and will try to 
             reproduce the images of interest. The discriminator will have to find in the image it receive is a real 
             image or an image that have been reproduced by the GAN. The Generator and Discriminator are playing a 
             game and when the generator win, the weights of the discriminator get updated for the next itteration and
             when the discriminator win, the weights of the generator get updated for the next itteration. 

Note:        In our case the generator will learn to reproduce images that look like the input data. For it part, the 
             discriminator will learn to know if the image is a real image or a fake one. Once both network became really
             good at what they do, the discriminator will be used to know if there is a welding error in a piece. Indeed, 
             it will know if there is an error if the input image do not look like the regenerate image of the generator. 

Next itteration: Try to use smaller images as the input. It could be possible to divise the original image in smaller pieces
                 like Mathias find in the litterature.
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