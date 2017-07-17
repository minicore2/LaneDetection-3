# -*- coding: utf-8 -*-
from sympy import *

ppm,f,tx,ty= symbols("ppm f tx ty")

Mint= Matrix([[ppm*f, 0, tx],[0, ppm*f, ty],[0,0,1]])

R_cv=Matrix()





