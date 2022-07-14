# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:36:26 2022

@author: Ella
"""
import numpy as np
from numpy import sqrt, exp, cos, sin
import matplotlib.pyplot as plt
from time import perf_counter
import winsound
from numba import jit
from tqdm import tqdm
from scipy.optimize import curve_fit
plt.close()
eta= 0.000273
H=8.5e3
R= 6e6

a= 6.378e6
b= 6.356e6
c= 6.378e6
def h(x,y,z):
    return (x**2/a**2 + y**2/b**2 +z**2/c**2 -1)/(2*sqrt(x**2/a**4+y**2/b**4 +z**2/c**4))


@jit
def TotalFunction(v,z): #the equation of motion we're trying to solve
    r1=sqrt(v[0]**2/a**2+v[1]**2/b**2+z**2/c**2)
    x=v[0]
    y=v[1]
    h1= (sqrt(x**2/a**2 +y**2/b**2 +z**2/c**2)-1)*a
    n1= 1+eta*exp(-(h1)/H)
    v1=v[2]
    v2=v[3]
    v3= sqrt(1-v1**2-v2**2)  
    dv11= -eta / H / r1 /n1 * exp(-(h1)/H)*(v3*x/a - v1*z/a + v2**2*x/v3/a - v1*v2*y*a/v3/b**2)
    dv22 = -eta / H / r1 /n1 * exp(-(h1)/H)*(v3*y*a/b**2 - v2*z/a + v1**2*y*a/v3/b**2 - v1*v2*x/v3/a)
    res = [v[2],v[3], dv11,dv22]
    return res

def LeapFrogSolve(dvdz, zspan, v0, n):#using leapfrog to solve the equations of motion
    
  z0 = zspan[0]
  zstop = zspan[1]
  dz = ( zstop - z0 ) / n

  z = np.zeros( n + 1 )
  v = np.zeros( [ n + 1, 4 ] )

  for i in range ( 0, n + 1 ):

    if ( i == 0 ):
      z[0]   = z0
      v[0,0] = v0[0]
      v[0,1] = v0[1]
      v[0,2] = v0[2]
      v[0,3] = v0[3]
      anew   = dvdz( v[i,:], z[i] )
    else:
      z[i]   = z[i-1] + dz
      aold   = anew
      v[i,0] = v[i-1,0] + dz * ( v[i-1,2] + 0.5 * dz * aold[2] )
      v[i,1] = v[i-1,1] + dz * ( v[i-1,3] + 0.5 * dz * aold[3] )
      anew   = dvdz ( v[i,:], z[i] )
      v[i,2] = v[i-1,2] + 0.5 * dz * ( aold[2] + anew[2] )
      v[i,3] = v[i-1,3] + 0.5 * dz * ( aold[3] + anew[3] )
  return v

#%% Testing one light ray
beginvalues= [6e6,0,0,0]

z0=-np.sqrt((R+10*H)**2-beginvalues[0]**2-beginvalues[1]**2)
n=15
zarray= np.linspace(z0, R+10*H,n+1) 
zspan= np.array([z0, zarray[-1]])
start= perf_counter()

res=LeapFrogSolve(TotalFunction, np.array([z0,zarray[-1]]), beginvalues, n )

end=perf_counter()
print("--- %s seconds" % (end-start))


#%% Creating the initial conditions
ringwidth = 110
rstepsize =2780
rpar=np.linspace(1.5e4-ringwidth, 1.5e4+ringwidth, rstepsize) 

theta= np.linspace(0,2*np.pi, 2*360)
ic=[]

for j in theta:
    for r in rpar:
        x=a*np.cos(j)*(1+r/a)
        y= b*np.sin(j)*(1+r/a)
        ic.append([x,y,0,0])
        
ringarea = np.pi*a*b*((1+rpar[-1]/a)**2-(1+rpar[0]/a)**2)
total_photons = len(ic)
#%% All lightrays calculation

start=perf_counter()
j=0

results=[LeapFrogSolve(TotalFunction, zspan, i, n )[-1] for i in tqdm(ic)]
end=perf_counter()
print("--- %s seconds ---" % (end-start))

duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)

#%% Making a line once the light ray leaves the atmosphere and plotting all the end points

midpoint_a = int(len(rpar)/2)
midpoint_b = int(len(rpar)*0.25*len(theta) + len(rpar)/2)

ta= -results[midpoint_a][0]/results[midpoint_a][2]
tb= -results[midpoint_b][1]/results[midpoint_b][3]
t= (ta+tb)/2
zeind= t+ R+10*H


x=[]
y=[]
for i in range(len(results)):
    x.append(t*(results[i][2])+results[i][0]) 
    y.append(t*(results[i][3])+results[i][1])

    
zplot=np.append(zarray,zeind)
fig, ax = plt.subplots()
ax.plot(x,y, ',')
ax.set_aspect('equal')
plt.savefig('figures/photonprop_LeapFrog_pixel_step' + \
            str(n) + '_' + str(len(ic)) + '_ring_' + str(ringwidth) +'.png')


#%%plot intensity

pixelsize=450
gridrange= 80000
gridx = np.arange(-gridrange,gridrange,pixelsize)
gridy = np.arange(-gridrange,gridrange,pixelsize)

amp_factor = ringarea/(total_photons* pixelsize**2)
grid, _, _ =np.histogram2d(x,y, bins=[gridx,gridy])

fig, ax= plt.subplots()
im = ax.pcolormesh(gridx/1e3, gridy/1e3, amp_factor*grid, cmap = 'hot')
fig.colorbar(im, ax = ax)
rx=[]
ry=[]

ax.set_aspect('equal')
ax.set_xlabel('km')
ax.set_ylabel('km')
for i in theta:
    rx.append(((a**2-b**2)/a*cos(i)**3)/1e3)
    ry.append(((b**2-a**2)/b*sin(i)**3)/1e3)
ax.plot(rx,ry)
   
#plt.savefig('figures/photonprop_LeapFrog_intensity_withcaustics' + \
#            str(n) + '_' + str(len(ic))  + '_ring_' + str(ringwidth) + '_pixelsize_' + str(pixelsize)+ '.png')
#%% Plotting the intensity for different pixelsizes
pix_size= [450,900,1250,1500]
fig, axs = plt.subplots(2,2)
i=0
titles = ['a)', 'b)', 'c)','d)']
for ax in axs.reshape(-1):
    pixelsize = pix_size[i]
    gridrange = 80000 + pixelsize/2
    gridx = np.arange(-gridrange,gridrange + pixelsize,pixelsize)
    gridy = np.arange(-gridrange,gridrange + pixelsize,pixelsize)

    amp_factor = ringarea/(total_photons* pixelsize**2)
    grid, _, _ =np.histogram2d(x,y, bins=[gridx,gridy])

    plotgrid = grid*amp_factor

    im = ax.pcolormesh(gridx/1e3, gridy/1e3, plotgrid, cmap = 'hot')
    fig.colorbar(im, ax = ax)
    ax.set_title(titles[i], loc='left')
    ax.set_aspect('equal')
    ax.set_xticks([-80,-60,-40,-20,0, 20, 40, 60 ,80])
    ax.set_yticks([-80,-60,-40,-20,0, 20, 40, 60 ,80])
    ax.set_xlabel('km')
    ax.set_ylabel('km')
    i+=1
plt.tight_layout()

#%% Looking at Amax for different pixelsizes
def A_max (pixelsize, gridrange):
    gridx = np.arange(-gridrange,gridrange,pixelsize)
    gridy = np.arange(-gridrange,gridrange,pixelsize)

    amp_factor = ringarea/(total_photons* pixelsize**2)
    grid, _, _ =np.histogram2d(x,y, bins=[gridx,gridy])

    plotgrid = grid*amp_factor
    return np.amax(plotgrid)

def f(x,a,b):
    return a*x**b
pix_size = np.linspace(50, 1000, 100)
amaxarray = [A_max(i, gridrange) for i in pix_size]
#%% analysing and plotting the data on a loglog plot
popt2, pcov2 = curve_fit(f, pix_size,amaxarray)
fig, ax= plt.subplots()
plt.loglog(pix_size,amaxarray, '.', color='black')
plt.loglog(pix_size, f(pix_size, *popt2), '-', color = (0.0,0.651,0.8392))
ax.set_xlabel('Pixelsize')
ax.set_ylabel('Maximum amplification')
plt.tight_layout()
#plt.savefig('figures/ellipsoidal_Amax_pixelsize_loglog_popt: ' +str(popt2) +'.jpg' )#%% R intensity
intensity = amp_factor*grid




