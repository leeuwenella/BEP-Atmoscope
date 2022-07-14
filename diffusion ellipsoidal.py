# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:07:17 2022

@author: Ella
"""
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
eta= 0.000273
H=8.5e3
R= 6e6
L0= 1e8 #graag de plaatjes maken voor L0 = H, L0 = np.sqrt(10*H*R) en L0 = 1e7
C0 = 9e-17

Diff = 9/20*C0/H**2*(L0**(5/3))
a= 6.378e6
b= 6.356e6
c= 6.378e6
def h(x,y,z):
    return (x**2/a**2 + y**2/b**2 +z**2/c**2 -1)/(2*sqrt(x**2/a**4+y**2/b**4 +z**2/c**4))


@jit
def TotalFunction(v,z):
    r1=sqrt(v[0]**2/a**2+v[1]**2/b**2+z**2/c**2)
    x=v[0]
    y=v[1]
    h1= (sqrt(x**2/a**2 +y**2/b**2 +z**2/c**2)-1)*a
    n1= 1+eta*exp(-(h1)/H)
    v1=v[2]
    v2=v[3]
    v3= sqrt(1-v1**2-v2**2)  
    a1 = (1-n1)/(v3*n1*H*r1)*(x/a -v1*(v1*x/a + v2*a*y/b**2 + v3*z/a))
    a2 = (1-n1)/(v3*n1*H*r1)*(y*a/(b**2) -v2*(v1*x/a + v2*a*y/b**2 + v3*z/a))
    res = [v[2],v[3], a1,a2]
    scalar = H*sqrt(Diff*n1*v3)*np.random.normal()/n1
    b1 = a1*scalar
    b2 = a2*scalar
    barray = [b1,b2]
    return res, barray



beginvalues= [6e6,0,0,0]

z0=-np.sqrt((R+10*H)**2-beginvalues[0]**2-beginvalues[1]**2)
n=25
zarray= np.linspace(z0, R+10*H,n+1) 
zspan= np.array([z0, zarray[-1]])

# Diff = 0
def LeapFrogSolve(dvdz, zspan, v0, n):
    
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
      anew,bnew   = dvdz( v[i,:], z[i] )

    else:
      z[i]   = z[i-1] + dz
      aold   = anew
      bold   = bnew
      v[i,0] = v[i-1,0] + dz * ( v[i-1,2] + 0.5 * dz * aold[2] )
      v[i,1] = v[i-1,1] + dz * ( v[i-1,3] + 0.5 * dz * aold[3] )
      anew,bnew   = dvdz ( v[i,:], z[i] )
      
      v[i,2] = v[i-1,2] + 0.5 *  ( aold[2] + anew[2] ) *dz \
               + 0.5 * (bold[0] + bnew[0]) *sqrt(dz)
      v[i,3] = v[i-1,3] + 0.5 * ( aold[3] + anew[3] ) * dz \
              + 0.5 * (bold[1] + bnew[1])*sqrt(dz)
  return v


#%% Creating the initial conditions
ringwidth = 110
rstepsize =2780
rpar=np.linspace(1.5e4-ringwidth, 1.5e4+ringwidth, rstepsize) 

theta= np.linspace(0,2*np.pi, 360*2)
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
results=[LeapFrogSolve(TotalFunction, zspan, i, n )[-1] for i in tqdm(ic)]
end=perf_counter()
print("--- %s seconds ---" % (end-start))

duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)

#%% Making a line

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
#plt.title('Amplification of a ellipsoidal earth')
ax.set_xlabel('km')
ax.set_ylabel('km')
# for i in theta:
#     rx.append(((a**2-b**2)/a*cos(i)**3)/1e3)
#     ry.append(((b**2-a**2)/b*sin(i)**3)/1e3)
# ax.plot(rx,ry)
plotgrid = amp_factor*grid
plt.savefig('figures/diffusion figures/photonprop_LeapFrog_intensity_withcausticsL0' + str(L0) + \
            str(n) + '_' + str(len(ic))  + '_ring_' + str(ringwidth) + '_pixelsize_' + str(pixelsize)+ 'amax' + str(round(np.amax(plotgrid),3)) +'.pdf')
#%%
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

#%%
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
#%% analysing the data 
popt2, pcov2 = curve_fit(f, pix_size,amaxarray)
fig, ax= plt.subplots()
plt.loglog(pix_size,amaxarray, '.', color='black')
plt.loglog(pix_size, f(pix_size, *popt2), '-', color = (0.0,0.651,0.8392))
ax.set_xlabel('Pixelsize')
ax.set_ylabel('Maximum amplification')
plt.tight_layout()
plt.savefig('figures/ellipsoidal_Amax_pixelsize_loglog_popt: ' +str(popt2) +'.jpg' )#%% R intensity
intensity = amp_factor*grid




