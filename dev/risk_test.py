r da#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 17:19:10 2020

@author: ianich
"""

from scipy import integrate
from scipy.stats import norm
import numpy as np

mu = 0.03662779759944417
sigma = 0.49046873667527724

z = -1


#can't condition on loss b/c sum may contain wins
### can't use normal pdf because of truncation for win (loss) for short (long) at 100
#something isn't right
def pooled_risk(z, mu, sigma):
    
    def f(x1, x2):
        return norm.pdf(x1, mu, sigma)*norm.pdf(x2, mu, sigma)
    
    def bounds_x1(x2):
        return [-1*np.inf, z - x2]
    
    def bounds_x2():
        return [-1*np.inf, 0]
    
    result = integrate.nquad(f, [bounds_x1, bounds_x2])
    return result

print(pooled_risk(z, mu, sigma))

