from model.symbols import *
from comparative_statics.canonical import price1_derivative, price2_derivative
ratio_derivative = (p2*price1_derivative(beta)-p1*price2_derivative(beta))/p2**2
