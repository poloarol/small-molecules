""" evaluation.py """

import numpy as np
from scipy.linalg import sqrtm


class Evaluations:
    """
    Provide methods to evaluate the effectiveness of
    Generative models used.
    """
    
    
    def inception_score(self) -> float:
        """"""
    
    def frechet_inception_distance(self, inputs_one, inputs_two) -> float:
        """ Calculate the Frechet Inception Distance """
                
        # calculate activations
        activation_one = self.model(inputs_one)
        activation_two = self.model(inputs_two)
        
        # calculate mean and covariance statistics
        mu1, sigma1 = activation_one[0].mean(axis=0), np.cov(activation_one[0], rowvar=False)
        mu2, sigma2 = activation_two[0].mean(axis=0), np.cov(activation_two[0], rowvar=False)
        
        # Calculate sqrt of product between cov
        ssdiff = np.sum((mu1 - mu2) ** 2)
        
        # Calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
    
    def structural_similarity_index(self) -> float:
        pass