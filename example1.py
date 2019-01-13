# Python Program illustrating 
# numpy.dot() method 
  
import numpy as geek 
  
# 1D array 
vector_a = geek.array([[1, 4], [5, 6]]) 
vector_b = geek.array([[2, 4], [5, 2]]) 
  
product = geek.dot(vector_a, vector_b) 
print product
  
product = geek.dot(vector_b, vector_a) 
print product
  
"""  
Code 2 : as normal matrix multiplication 
"""