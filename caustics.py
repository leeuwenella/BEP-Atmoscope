# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:51:21 2022

@author: Ella
"""
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt

plt.close('all')
eta= 0.000273
H=100
R= 3000

a= R+1000 
b= R
c= R+1000
theta = np.linspace(0,2*np.pi,100)

rx=[]
ry=[]
r2x=[]
r2y=[]
for i in theta:
    rx.append(a*cos(i) - (a**2 *sin(i)**2 + b**2*cos(i)**2)*cos(i)/a)
    ry.append( b*sin(i) - (a**2 * sin(i)**2 +b**2*cos(i)**2)*sin(i)/b)
    r2x.append((a**2-b**2)/a*cos(i)**3)
    r2y.append((b**2-a**2)/b*sin(i)**3)
fig, ax = plt.subplots()
ax.plot(rx,ry)
ax.plot(r2x,r2y)
ax.set_aspect('equal')    

#%%geometric plot
caust_theta = np.linspace(0,2*np.pi,100)
xell=[]
yell=[]

for i in caust_theta:
    [x,y] = [a*cos(i),b*sin(i)]
    norm = [x/a**2, y/b**2]
    t = -x/norm[0]
    [x2,y2]= [x + t*norm[0], y +t*norm[1]] 
    plt.plot([x,x2], [y,y2], color = 'black')
    plt.axis('off')
for i in theta:
    xell.append(a*cos(i))
    yell.append(b*sin(i))
plt.plot(xell,yell, color = (0.0,0.651,0.8392))   
rx=[(a**2-b**2)/a*(cos(i)**3) for i in theta]
ry=[(b**2-a**2)/b*(sin(i)**3) for i in theta]
plt.plot(rx,ry, color = (0.0,0.651,0.8392) )