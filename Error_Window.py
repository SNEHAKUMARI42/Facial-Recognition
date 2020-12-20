# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:56:39 2019

@author: Shivam
"""

Error = 0
number = 0
percentage = 0

def Error_Allowed(number,percentage):
    
    Error = (number*percentage)/100
    
    Upper_Limit = number + Error
    Lower_Limit = number - Error
    
    return Upper_Limit,Lower_Limit