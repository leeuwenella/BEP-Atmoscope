# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:27:33 2021

@author: Ella
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
R = 6.371e6
c=3e8
H = 8.5e3
eta=0.000273

def dlogn(r):
    return  (-eta/H)*(np.exp((R-r)/H)/(1+eta*np.exp((R-r)/H)))

def f(r,theta):
    return (r[1], 2*r[1]**2/r[0] +r[0]+ ((r[0]**2+r[1]**2)/r[0])**2*dlogn(r[0]))

rbegin= 2e7
end = 400
h=1e4
r0= [rbegin, -rbegin/(np.tan(np.arcsin((R+h)/rbegin))) ]
thetas= np.linspace(np.arcsin((R+h)/rbegin), np.pi+0.0001, 100000)

us=odeint(f,r0, thetas, full_output=1)
rs=np.array(us[0][:-end,0])
xs= rs*np.cos(thetas[:-end])
ys = rs*np.sin(thetas[:-end])

hoek = np.linspace(0,2*np.pi, 10000)
yearth=R*np.sin(hoek)
xearth=R*np.cos(hoek)

yatm=(R+H)*np.sin(hoek)
xatm=(R+H)*np.cos(hoek)
plt.plot(xs,ys)
plt.plot(xearth,yearth)
plt.plot(xatm,yatm)

#%%

class Lightray(object):
    
    def __init__(self, h, rbegin, steps):
        self.h = h
        self.rbegin = rbegin
        self.steps = steps
        
        
        self.r0 = [rbegin, -rbegin/(np.tan(np.arcsin((R+h)/rbegin))) ]
        self.thetas = np.linspace(np.arcsin((R+h)/rbegin), np.pi+0.0001,steps)
        
        self.us=odeint(f,self.r0, self.thetas)
        

        self.rs=np.array(self.us[:-end,0])
        
        self.min_r = min(self.rs)
        self.min_h = self.min_r - R
        
        self.xray = self.rs*np.cos(self.thetas[:-end])
        self.yray = self.rs*np.sin(self.thetas[:-end])
        self.path = [self.xray, self.yray]
        
        self.dtheta =2*eta*np.exp(-self.min_h/H)*np.sqrt(np.pi/2*(R+self.min_h)/H)
        self.a=(self.yray[-2] - self.yray[-1])/(self.xray[-2]-self.xray[-1])
        self.nu_num = np.arctan((self.yray[-2] - self.yray[-1])/(self.xray[-2]-self.xray[-1]))
        
        #calculating detector position
        self.b=self.yray[-2]-self.a*self.xray[-2]
        self.L=-self.b/self.a
        
        

    

        


        
class Planet(object):
    hoek = np.linspace(0,2*np.pi, 10000)
    def __init__(self, radius, atmosphere):
        self.R = radius
        self.H = atmosphere
        
        self.x = self.R*np.cos(self.hoek)
        self.y = self.R*np.sin(self.hoek)
        
        self.atmx = self.H*np.cos(self.hoek)
        self.atmy = self.H*np.sin(self.hoek)

#%%    
earth = Planet(6.371e6, 1e4+6.371e6)    
ray1 = Lightray(9e3, 2e7, int(1e6))

startheights = np.linspace(5e3, 1e4, 8)
rays =[]
for h in startheights:
    newray=Lightray(h, rbegin,100000 )
    rays.append(newray)
    plt.plot(newray.path[0], newray.path[1])
    
plt.plot(earth.x, earth.y, color= 'g')
plt.plot(earth.atmx, earth.atmy, color= 'b')

#%%
#figure of earth, atmosphere and 8 ightrays
earth = plt.Circle((0,0), 6.371e6, color = 'g')
fig, ax = plt.subplots()

start = 0.0
stop = .5
number_of_lines= 1000
cm_subsection = np.linspace(start, stop, number_of_lines) 

colors = [ plt.cm.Blues(x) for x in cm_subsection ]
colors.reverse()
rs= np.linspace(6.371e6,6.381e6,1000)
hoek = np.linspace(0,2*np.pi, 10000)
xs=[i*np.cos(hoek) for i in rs]
ys=[i*np.sin(hoek) for i in rs]

for i, color in enumerate(colors):
    ax.plot(xs[i],ys[i], color=color)
    
startheights = np.linspace(1743.4, 1.5e4, 8)
rays =[]
for h in startheights:
    newray=Lightray(h, rbegin,100000 )
    rays.append(newray)
    ax.plot(newray.path[0], newray.path[1], color = 'black')
ax.add_patch(earth)
    

#%%
#figure of earth, atmosphere, 1 lightray and the analytic angle
earth = plt.Circle((0,0), 6.371e6, color = 'g')
fig, ax = plt.subplots()

start = 0.0
stop = .5
number_of_lines= 1000
cm_subsection = np.linspace(start, stop, number_of_lines) 

colors = [ plt.cm.Blues(x) for x in cm_subsection ]
colors.reverse()
rs= np.linspace(6.371e6,6.381e6,1000)
hoek = np.linspace(0,2*np.pi, 10000)
xs=[i*np.cos(hoek) for i in rs]
ys=[i*np.sin(hoek) for i in rs]


ax=plt.gca()
ax.set_aspect(5)

for i, color in enumerate(colors):
    ax.plot(xs[i],ys[i], color=color)
ax.add_patch(earth)

h=1e4
ray=Lightray(h, 2e7, 100000)
ax.plot(ray.path[0],ray.path[1], color='black')

r_asymptote_x = np.linspace(-20000, 2e7, 3)
r_asymptote_y = np.array([ray.h+ R, ray.h + R, ray.h + R])
ax.plot(r_asymptote_x, r_asymptote_y, color= 'gray')

l_asymptote_x = np.linspace(-4e8,20000, 10000)
l_asymptote_y = np.tan(ray.dtheta)*l_asymptote_x + (ray.h+ R)/np.cos(ray.dtheta)

ax.plot(l_asymptote_x, l_asymptote_y, color= 'gray', linestyle = 'dashed')


#%% numerical vs analytic nu

harray=np.arange(1743.4,1e4,1)
numv=[]
analyticv=[]
minh=[]
for h in harray:
    ray=Lightray(h,rbegin,10000)
    numv.append(ray.nu_num*180/np.pi)
    analyticv.append(ray.dtheta*180/np.pi)
    minh.append(ray.min_h)

#%%
plt.plot(minh,  numv,  label='Numerical calculation')
plt.plot(minh, analyticv, label = 'Analytic calculation')
plt.legend()
plt.xlabel(r'Minimal height $h$ (m)')
plt.ylabel(r'Deflection angle $\nu$ (degrees)')
plt.savefig('Numeric vs analytic deflection angle.pdf')
plt.grid()


#%% plot for detector as minimal h

harray=np.arange(1743.4,1e4, 100)
minh=[]
L=[]
for h in harray:
    ray=Lightray(h,rbegin,10000)
    L.append(-ray.L*1e-3)
    minh.append(ray.min_h)

plt.plot(L, minh)
plt.ylabel(r'Minimal height $h$ (m)')
plt.xlabel(r'Detector distance (km)')
plt.tight_layout()
plt.grid()
plt.savefig('Detector distance as a function of minimal height.pdf')

#%%
ray= Lightray(1.5e4, 2e7, int(1e6))
plt.plot(ray.path[0],ray.path[1])