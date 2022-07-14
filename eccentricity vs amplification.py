# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:41:32 2022

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
R= 6e6


@jit
def TotalFunction(v,z,a,b,c):
    r1=sqrt(v[0]**2/a**2+v[1]**2/b**2+z**2/c**2)
    x=v[0]
    y=v[1]
    h1= (sqrt(x**2/a**2 +y**2/b**2 +z**2/c**2)-1)*a
    #h1 = (x**2/a**2 + y**2/b**2 +z**2/c**2 -1)/(2*sqrt(x**2/a**4+y**2/b**4 +z**2/c**4))
    n1= 1+eta*exp(-(h1)/H)
    v1=v[2]
    v2=v[3]
    v3= sqrt(1-v1**2-v2**2)   
    dv11= -eta / H / r1 /n1 * exp(-(h1)/H)*(v3*x/a - v1*z/a + v2**2*x/v3/a - v1*v2*y*a/v3/b**2)
    dv22 = -eta / H / r1 /n1 * exp(-(h1)/H)*(v3*y*a/b**2 - v2*z/a + v1**2*y*a/v3/b**2 - v1*v2*x/v3/a)
    v = [v[2],v[3], dv11,dv22]
    return v



beginvalues= [6e6,0,0,0]

z0=-np.sqrt((R+10*H)**2-beginvalues[0]**2-beginvalues[1]**2)
n=15
zarray= np.linspace(z0, R+10*H,n+1) 
zspan= np.array([z0, zarray[-1]])


def LeapFrogSolve(dvdz, zspan, v0, n,a,b,c):
    
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
      v[0,2] = v0[3]
      anew   = dvdz( v[i,:], z[i],a,b,c )
    else:
      z[i]   = z[i-1] + dz
      aold   = anew
      v[i,0] = v[i-1,0] + dz * ( v[i-1,2] + 0.5 * dz * aold[2] )
      v[i,1] = v[i-1,1] + dz * ( v[i-1,3] + 0.5 * dz * aold[3] )
      anew   = dvdz ( v[i,:], z[i],a,b,c )
      v[i,2] = v[i-1,2] + 0.5 * dz * ( aold[2] + anew[2] )
      v[i,3] = v[i-1,3] + 0.5 * dz * ( aold[3] + anew[3] )
  
  return v

#%%
start= perf_counter()
a= 6.378e6
barray = np.array([-22000])+a
c= 6.378e6
res=[LeapFrogSolve(TotalFunction, zspan, beginvalues, n,a,b,c ) for b in barray]

end=perf_counter()
print("--- %s seconds" % (end-start))


#%% Creating the initial conditions
a= 6.378e6
earth_ecc = np.sqrt(1-(a-22000)**2/a**2)
ecc=np.linspace(0,earth_ecc, 5)
barray = np.sqrt((1-ecc**2)*a**2)
c= 6.378e6
#%%
ringwidth = 110
rstepsize =2780
rpar=np.linspace(1.5e4-ringwidth, 1.5e4+ringwidth, rstepsize) 

theta= np.linspace(0,2*np.pi, 2*360)
ic=[]
for i in range(len(barray)):
    ic.append([])
    for j in theta:
        for r in rpar:
            x=a*np.cos(j)*(1+r/a)
            y=barray[i]*np.sin(j)*(1+r/a)
            ic[i].append([x,y,0,0])   
ringarea = [np.pi*a*b*((1+rpar[-1]/a)**2-(1+rpar[0]/a)**2) for b in barray]
total_photons = len(ic[0])
#%% All lightrays calculation

start=perf_counter()
j=0

results=[]
i=0
for b in barray:
    results.append([])
    results[-1]=[LeapFrogSolve(TotalFunction, zspan, j, n,a,b,c )[-1] for j in tqdm(ic[i])]
    i+=1
end=perf_counter()
print("--- %s seconds ---" % (end-start))

duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)

#%% Making a line

midpoint_a = int(len(rpar)/2)
midpoint_b = int(len(rpar)*0.25*len(theta) + len(rpar)/2)

ta= np.array([-results[i][midpoint_a][0]/results[i][midpoint_a][2] for i in range(len(barray))])
tb= np.array([-results[i][midpoint_b][1]/results[i][midpoint_b][3] for i in range(len(barray))])
t= (ta+tb)/2
zeind= t + R+10*H


x=[]
y=[]
for i in range(len(results)):
    x.append([])
    y.append([])
    for j in range(len(results[i])):
                
        x[i].append(t[i]*(results[i][j][2])+results[i][j][0]) 
        y[i].append(t[i]*(results[i][j][3])+results[i][j][1])
        
        
zplot=np.append(zarray,zeind)
fig, ax = plt.subplots()
ax.plot(x[0],y[0], ',')
ax.set_aspect('equal')
#plt.savefig('figures/photonprop_LeapFrog_pixel_step' + \
#            str(n) + '_' + str(len(ic)) + '_ring_' + str(ringwidth) +'.jpg')


#%%plot intensity

pixelsize=450
gridrange= 80000
gridx = np.arange(-gridrange,gridrange,pixelsize)
gridy = np.arange(-gridrange,gridrange,pixelsize)


amp_factor1 = ringarea[1]/(total_photons* pixelsize**2)
grid1, _, _ =np.histogram2d(x[1],y[1], bins=[gridx,gridy])

amp_factor2 = ringarea[2]/(total_photons* pixelsize**2)
grid2, _, _ =np.histogram2d(x[2],y[2], bins=[gridx,gridy])

amp_factor3 = ringarea[3]/(total_photons* pixelsize**2)
grid3, _, _ =np.histogram2d(x[3],y[3], bins=[gridx,gridy])

amp_factor4 = ringarea[4]/(total_photons* pixelsize**2)
grid4, _, _ =np.histogram2d(x[4],y[4], bins=[gridx,gridy])

fig, axs= plt.subplots(2,2)
ax=axs[0,0]
im1 = ax.pcolormesh(gridx/1e3, gridy/1e3, amp_factor1*grid1, cmap = 'hot')
fig.colorbar(im1, ax = ax)
ax.set_aspect('equal')
#plt.title('Amplification of a ellipsoidal earth')
ax.set_xlabel('km')
ax.set_ylabel('km')
ax.set_title('a)', loc='left')
ax.set_xticks(np.arange(-80,120,40))
ax.set_yticks(np.arange(-80,120,40))
ax=axs[0,1]
im2 = ax.pcolormesh(gridx/1e3, gridy/1e3, amp_factor2*grid2, cmap = 'hot')
fig.colorbar(im2, ax = ax)
ax.set_aspect('equal')
#plt.title('Amplification of a ellipsoidal earth')
ax.set_xlabel('km')
ax.set_ylabel('km')
ax.set_title('b)', loc='left')
ax.set_xticks(np.arange(-80,120,40))
ax.set_yticks(np.arange(-80,120,40))
ax=axs[1,0]
im3 = ax.pcolormesh(gridx/1e3, gridy/1e3, amp_factor3*grid3, cmap = 'hot')
fig.colorbar(im3, ax = ax)
ax.set_aspect('equal')
#plt.title('Amplification of a ellipsoidal earth')
ax.set_xlabel('km')
ax.set_ylabel('km')
ax.set_title('c)', loc='left')
ax.set_xticks(np.arange(-80,120,40))
ax.set_yticks(np.arange(-80,120,40))
ax=axs[1,1]
im4 = ax.pcolormesh(gridx/1e3, gridy/1e3, amp_factor4*grid4, cmap = 'hot')
fig.colorbar(im4, ax = ax)
ax.set_aspect('equal')
ax.set_xticks(np.arange(-80,120,40))
ax.set_yticks(np.arange(-80,120,40))
#plt.title('Amplification of a ellipsoidal earth')
ax.set_title('d)', loc='left')
ax.set_xlabel('km')
ax.set_ylabel('km')
ax.set_aspect('equal')
#fig.colorbar(im1, ax=axs.ravel().tolist())
#plt.title('Amplification of a ellipsoidal earth')
ax.set_xlabel('km')
ax.set_ylabel('km')
plt.tight_layout()
   
plt.savefig('figures/Multiple different eccentricities in one plot.pdf')
#%%
def A_max (x,y,pixelsize, gridrange, ringarea):
    gridx = np.arange(-gridrange,gridrange,pixelsize)
    gridy = np.arange(-gridrange,gridrange,pixelsize)

    amp_factor = ringarea/(total_photons* pixelsize**2)
    grid, _, _ =np.histogram2d(x,y, bins=[gridx,gridy])

    plotgrid = grid*amp_factor
    return np.amax(plotgrid)



pix_size1 = np.logspace(2, 3, 100)
pix_size=np.copy(pix_size1)
amaxarray = [[A_max(x[i],y[i],j, gridrange, ringarea[i]) for j in tqdm(pix_size) ]for i in tqdm(range(len(x))) ]
#fig, ax=plt.subplots()
eccentricity = [np.sqrt(1- b**2/(a**2)) for b in barray]
#ax.plot(eccentricity, amaxarray, '.')
#%% analysing the data 
fig, ax= plt.subplots()
def f(x,a,b):
    return a*(x**b)
popts=[]
for i in range(len(amaxarray)):
    color = next(ax._get_lines.prop_cycler)['color']
    
    plt.loglog(pix_size,amaxarray[i], '.', label = str(round(ecc[i],3)), color= color)
    
    popt, pcov = curve_fit(f, pix_size[1:],amaxarray[i][1:], p0 =[8*H,-1])
    popts.append(popt)
    plt.loglog(pix_size, f(pix_size, *popt), '-', color = color)



ax.set_xlabel('Pixel-width (m)')
ax.set_ylabel('Maximum amplification')
plt.legend(title='Eccentricity')

plt.tight_layout()
plt.savefig('figures/eccentricity_amp_pixelsize.pdf')#%% R intensity
#intensity = amp_factor*grid
# with open('popts.txt', 'w') as f:
#     f.write('the curve fits for the eccentricity and pixelsize shizzle are as follows: \n' \
#             +str(popts))


