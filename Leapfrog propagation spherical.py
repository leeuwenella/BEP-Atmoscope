# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:27:42 2022

@author: Ella
"""
import numpy as np
from numpy import sqrt, exp
import matplotlib.pyplot as plt
from time import perf_counter
import winsound
from numba import jit
from tqdm import tqdm
from scipy.optimize import curve_fit

plt.close()
eta= 0.000273
H=8.5e3
R= 6.371e6

@jit
def TotalFunction(v,z):     #this is the differential equation for the propagation in z-direction of photons
    r1=sqrt(v[0]**2+v[1]**2+z**2)
    n1= 1+eta*exp(-(r1-R)/H)

    x=v[0]
    y=v[1]
    v1=v[2]
    v2=v[3]
    v3= np.sqrt(1-v1**2-v2**2)   
    dv11= -eta / H / r1 /n1 * exp(-(r1-R)/H)*(v3*x - v1*z + v2**2*x/v3 - v1*v2*y/v3)
    dv22 = -eta / H / r1 /n1 * exp(-(r1-R)/H)*(v3*y - v2*z + v1**2*y/v3 - v1*v2*x/v3)
    v = [v[2],v[3], dv11,dv22]
    return v


beginvalues= [6e6,0,0,0] #just a test beginvalues, and to create a begin z value

z0=-np.sqrt((R+10*H)**2-beginvalues[0]**2-beginvalues[1]**2)    #to make sure z starts far from the earth
n=15     #amount of steps taken in integration method
zarray= np.linspace(z0, R+10*H,n+1) 
zspan= np.array([z0, zarray[-1]])

def LeapFrogSolve(dvdz, zspan, v0, n):  # leapfrog integration method
    
  z0 = zspan[0]
  zstop = zspan[1]
  dz = ( zstop - z0 ) / n

  z = np.zeros( n + 1 )
  v = np.zeros( [ n + 1, 4 ] )

  for i in range ( 0, n + 1 ):

    if ( i == 0 ): #setting initial values
      z[0]   = z0
      v[0,0] = v0[0]
      v[0,1] = v0[1]
      v[0,2] = v0[2]
      v[0,3] = v0[3]
      anew   = dvdz( v[i,:], z[i] )
    else:   #updating all values using leapfrog
      z[i]   = z[i-1] + dz
      aold   = anew
      v[i,0] = v[i-1,0] + dz * ( v[i-1,2] + 0.5 * dz * aold[2] )
      v[i,1] = v[i-1,1] + dz * ( v[i-1,3] + 0.5 * dz * aold[3] )
      anew   = dvdz ( v[i,:], z[i] )
      v[i,2] = v[i-1,2] + 0.5 * dz * ( aold[2] + anew[2] )
      v[i,3] = v[i-1,3] + 0.5 * dz * ( aold[3] + anew[3] )
  
  return v

#%% testing 1 ray
start= perf_counter()

res= LeapFrogSolve(TotalFunction, np.array([z0,zarray[-1]]), beginvalues, n )

end=perf_counter()
print("--- %s seconds" % (end-start))

plt.plot(zarray,np.transpose(res)[0])

#%% Creating the initial conditions
ringwidth = 1
ringarea = np.pi*(2*(R+1.5e4)+ringwidth)*(2)
rsteps= 2780
thetasteps=2*360
total_photons= rsteps*thetasteps
rpar=np.linspace(R+1.5e4-ringwidth, R+1.5e4+ringwidth, rsteps) 
theta= np.linspace(0,2*np.pi, thetasteps)
ic=[]
z0 = np.sqrt((R+10*H)**2- rpar[0]**2)
zspan = [-z0,z0]
for i in rpar:
    for j in theta:
        x=i*np.cos(j)
        y=i*np.sin(j)
        ic.append([x,y,0,0])
#%% All lightrays calculation

start=perf_counter()
j=1
results=[LeapFrogSolve(TotalFunction, zspan, i, n )[-1] for i in tqdm(ic)]
end=perf_counter()
print("--- %s seconds ---" % (end-start))

duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)

#%% Making a line
midpoint = int(rsteps*thetasteps/2)
t= -results[midpoint][0]/results[midpoint][2]
zeind= sqrt(results[midpoint][2]**2)*t + R+10*H

#plot_array=results.copy()

x=[]
y=[]
for i in range(len(results)):
    #plot_array[i]=np.transpose(results[i])
    
    x.append(t*(results[i][2])+results[i][0]) 
    y.append(t*(results[i][3])+results[i][1])
    #plot_array[i]=np.append(plot_array[i],np.array([[x],[y],[0],[0]]), axis=1)
    #plottable_array[i]
    
zplot=np.append(zarray,zeind)
fig, ax = plt.subplots()
ax.plot(x,y, ',')
ax.set_aspect('equal')
# plt.savefig('figures/photonprop_LeapFrog_pixels' + \
#           str(n) + '_' + str(len(ic)) + '_ring_' + str(ringwidth) +'.jpg')

#%%plot intensity

pixelsize = 1
gridrange = 50
gridx = np.arange(-gridrange -pixelsize/2,gridrange +1.5*pixelsize,pixelsize)
gridy = np.arange(-gridrange -pixelsize/2 ,gridrange +1.5*pixelsize,pixelsize)

amp_factor = ringarea/(total_photons* pixelsize**2)
grid, _, _ =np.histogram2d(x,y, bins=[gridx,gridy])

plotgrid = grid*amp_factor
fig, ax= plt.subplots()
im = ax.pcolormesh(gridx, gridy, plotgrid, cmap = 'hot')
fig.colorbar(im, ax = ax)
ax.set_aspect('equal')

ax.set_xlabel('m')
ax.set_ylabel('m')
plt.savefig('figures/photonprop_LeapFrog_pixelsize' + \
            str(pixelsize)+'_Gridrange_'+str(gridrange) + '_' + str(len(ic)) \
            + '_ring_' + str(ringwidth) + '_L = '+str(round(zeind/1e9,2)) + '1e9km .pdf')
    
#%% Finding a_max for pixelsizes
def A_max (pixelsize, gridrange):
    gridx = np.arange(-gridrange -pixelsize/2,gridrange+ 1.5*pixelsize,pixelsize)
    gridy = np.arange(-gridrange -pixelsize/2,gridrange+1.5*pixelsize, pixelsize)

    amp_factor = ringarea/(total_photons* pixelsize**2)
    grid, _, _ =np.histogram2d(x,y, bins=[gridx,gridy])

    plotgrid = grid*amp_factor
    return np.amax(plotgrid)

def f(x,a,b):
    return a*1/x+b
pix_size = np.array([1,2,5,10])

fig, axs = plt.subplots(2,2)
i=0
titles = ['a)', 'b)', 'c)','d)']
for ax in axs.reshape(-1):
    pixelsize = pix_size[i]
    gridrange = 50 + pixelsize/2
    gridx = np.arange(-gridrange,gridrange + pixelsize,pixelsize)
    gridy = np.arange(-gridrange,gridrange + pixelsize,pixelsize)

    amp_factor = ringarea/(total_photons* pixelsize**2)
    grid, _, _ =np.histogram2d(x,y, bins=[gridx,gridy])

    plotgrid = grid*amp_factor

    im = ax.pcolormesh(gridx, gridy, plotgrid, cmap = 'hot')
    fig.colorbar(im, ax = ax)
    ax.set_title(titles[i], loc='left')
    ax.set_aspect('equal')
    ax.set_xticks([-50,-25,0,25,50])
    ax.set_yticks([-50,-25,0,25,50])
    ax.set_xlabel('m')
    ax.set_ylabel('m')
    i+=1
plt.tight_layout()
pix_size = np.linspace(1, 25, 100)
#%%
amaxarray = [A_max(i, gridrange) for i in tqdm(pix_size)]
#%% analysing the data 

popt, pcov = curve_fit(f, pix_size,amaxarray)
fig, ax= plt.subplots()
plt.plot(pix_size,amaxarray, '.', color='black')
plt.plot(pix_size, f(pix_size, *popt), '-', color = (0.0,0.651,0.8392))
ax.set_xlabel('Pixel-width')
ax.set_ylabel('Maximum amplification')
plt.tight_layout()
#plt.savefig('figures/spherical_Amax_pixelsize.pdf)
#%% using a different f for a loglogplot of amax
def f(x,a,b):
    return a*x**b


#%% analysing the data and plotting the loglog plot
popt2, pcov2 = curve_fit(f, pix_size[:],amaxarray[:])
fig, ax= plt.subplots()
plt.loglog(pix_size,amaxarray, '.', color='black')
plt.loglog(pix_size, f(pix_size, *popt2), '-', color = (0.0,0.651,0.8392))
ax.set_xlabel('Pixelsize')
ax.set_ylabel('Maximum amplification')
plt.tight_layout()
#plt.savefig('figures/ellipsoidal_Amax_pixelsize_loglog_popt: ' +str(popt2) +'.jpg' )#%% R intensity
#intensity = amp_factor*grid


        