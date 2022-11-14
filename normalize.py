# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 19:17:48 2022

@author: Sushanth
"""

def normalize(samples):
  samples_norm = samples/max(samples)
  return samples_norm