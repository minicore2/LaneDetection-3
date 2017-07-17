# -*- coding: utf-8 -*-
import numpy as np

def fu(xvec,sigma):
    out= (1/sigma**2)*np.exp(-xvec*xvec/2/sigma/sigma)*(1-xvec*xvec/sigma/sigma)
    out= out/ np.linalg.norm(out)
    return out
    
def fv(xvec,sigma):
    out= np.exp(-xvec*xvec/2/sigma/sigma)
    out= out/ np.linalg.norm(out)
    return out

xvec= np.array(range(-3,4))
kernelX= fu(xvec, 1)

print "X kernel:"
for x in kernelX:
    print x,","
    
yvec= np.array(range(-2,3))
kernelY= fv(yvec, 1)

print "\n\nY kernel:"
for x in kernelY:
    print x,","