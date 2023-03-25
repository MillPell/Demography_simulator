# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:44:08 2023

@author: willi
"""


import scipy.stats

inputt = 90
inputt2 = 95

pdf = scipy.stats.norm(82, 5).pdf(inputt)
cdf1 = scipy.stats.norm(82, 5).cdf(inputt)
cdf2 = scipy.stats.norm(82, 5).cdf(inputt2)
result = cdf2 - cdf1


print ('pdf: ' + str(pdf) + ' cdf: ' + str(cdf1))
print ('proba entre: ' + str(inputt) + ' et: ' + str(inputt2) + ' est égal à: ' + str(result))