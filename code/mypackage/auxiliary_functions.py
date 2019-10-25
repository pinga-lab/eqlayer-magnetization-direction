import numpy as np
from fatiando.gridder import regular
from fatiando.utils import ang2vec, vec2ang
from fatiando.constants import CM, T2NT, G, SI2MGAL, SI2EOTVOS
from scipy.optimize import nnls

def sph2cart(mag,inc,dec):
    '''
    Calculate the Cartesian coordinates of a vector

    input

    mag: float - Magnetization intensity
    inc: float - Inclination
    dec: float - Declination

    return

    mx,my,mz - float - Cartesian coordinates
    '''
    mx = mag*np.cos(np.deg2rad(inc))*np.cos(np.deg2rad(dec))
    my = mag*np.cos(np.deg2rad(inc))*np.sin(np.deg2rad(dec))
    mz = mag*np.sin(np.deg2rad(inc))

    mag = np.array([mx,my,mz])

    return mag


def derivative_inclination(mag,inc,dec):
    '''
    Calculate the Cartesian coordinates of th derivative of a vector in respect to inclination

    input

    mag: float - Magnetization intensity
    inc: float - Inclination
    dec: float - Declination

    return

    mx,my,mz - float - Cartesian coordinates
    '''
    inc_rad = np.deg2rad(inc)
    dec_rad = np.deg2rad(dec)

    dmx = - mag*np.sin(inc_rad)*np.cos(dec_rad)
    dmy = - mag*np.sin(inc_rad)*np.sin(dec_rad)
    dmz =   mag*np.cos(inc_rad)

    dmag = np.array([dmx,dmy,dmz])

    return dmag

def derivative_declination(mag,inc,dec):
    '''
    Calculate the Cartesian coordinates of th derivative of a vector in respect to declination

    input

    mag: float - Magnetization intensity
    inc: float - Inclination
    dec: float - Declination

    return

    mx,my,mz - float - Cartesian coordinates
    '''

    inc_rad = np.deg2rad(inc)
    dec_rad = np.deg2rad(dec)

    dmx = - mag*np.cos(inc_rad)*np.sin(dec_rad)
    dmy =   mag*np.cos(inc_rad)*np.cos(dec_rad)
    dmz = 1e-10

    dmag = np.array([dmx,dmy,dmz])

    return dmag

def kernelxx(x,y,z,xs,ys,zs):
    '''
    Calculate the second derivative in relation x of a function 1/r.

    input
    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources

    output
    phi_xx : numpy array - Values with the second derivatives at one point.
    '''
    assert xs.size == ys.size and ys.size == zs.size and xs.size == zs.size, \
    'All arrays must have the same size'

    r = np.sqrt((x - xs)**2 + (y - ys)**2 + (z - zs)**2)
    r2 = r*r
    phi_xx = (((3.*(x - xs)**2)/(r2*r2*r))-(1./(r2*r)))

    return phi_xx

def kernelxy(x,y,z,xs,ys,zs):
    '''
    Calculate the second derivative in relation x and y of a function 1/r.

    input
    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources

    output
    phi_xy : numpy array - Values with the second derivatives.
    '''
    assert x.size == y.size and y.size == z.size, \
    'All arrays must have the same size'

    r = np.sqrt((x - xs)**2 + (y - ys)**2 + (z - zs)**2)
    r2 = r*r
    phi_xy = 3.*(((x - xs)*(y - ys))/(r2*r2*r))

    return phi_xy

def kernelxz(x,y,z,xs,ys,zs):
    '''
    Calculate the second derivative in relation x and z of a function 1/r.

    input
    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources

    output
    phi_xz : numpy array - Values with the second derivatives.
    '''
    assert x.size == y.size and y.size == z.size, \
    'All arrays must have the same size'

    r = np.sqrt((x - xs)**2 + (y - ys)**2 + (z - zs)**2)
    r2 = r*r
    phi_xz = 3.*(((x - xs)*(z - zs))/(r2*r2*r))

    return phi_xz

def kernelyy(x,y,z,xs,ys,zs):
    '''
    Calculate the second derivative in relation y of a function 1/r.

    input
    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources

    output
    phi_yy : numpy array - Values with the second derivatives.
    '''
    assert x.size == y.size and y.size == z.size and x.size == z.size, \
    'All arrays must have the same size'

    r = np.sqrt((x - xs)**2 + (y - ys)**2 + (z - zs)**2)
    r2 = r*r
    phi_yy = ((3.*(y - ys)**2)/(r2*r2*r))-(1./(r2*r))

    return phi_yy

def kernelyz(x,y,z,xs,ys,zs):
    '''
    Calculate the second derivative in relation y and z of a function 1/r.

    input
    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources

    output
    phi_yz : numpy array - Values with the second derivatives.
    '''
    assert x.size == y.size and y.size == z.size, \
    'All arrays must have the same size'

    r = np.sqrt((x - xs)**2 + (y - ys)**2 + (z - zs)**2)
    r2 = r*r
    phi_yz = 3.*((y - ys)*(z - zs))/(r2*r2*r)

    return phi_yz

def kernelzz(x,y,z,xs,ys,zs):
    '''
    Calculate the second derivative in relation z of a function 1/r.

    input
    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources

    output
    phi_zz : numpy array - Values with the second derivatives at one point.
    '''
    assert xs.size == ys.size and ys.size == zs.size and xs.size == zs.size, \
    'All arrays must have the same size'

    r = np.sqrt((x - xs)**2 + (y - ys)**2 + (z - zs)**2)
    r2 = r*r
    phi_zz = (((3.*(z - zs)**2)/(r2*r2*r))-(1./(r2*r)))

    return phi_zz

def tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p,inc,dec):
    '''
    Calculate the Total Field Anomaly produced by a layer

    input

    x,y,z : float - Cartesian coordinates (in m) of the i-th observation point
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources
    sinc,sdec: float - Main field direction
    p: numpy array - Vector composed by magnetic moment and magnetization direction of the
                     equivalent sources

    return
    tfa : numpy array - total field anomaly of the equivalent layer
    '''
    N = x.size
    M = xs.size

    tfa = np.empty(N,dtype=float)

    F0x,F0y,F0z = ang2vec(1.,sinc,sdec)
    mx,my,mz = ang2vec(1.,inc,dec)

    for i in range(N):
        phi_xx = kernelxx(x[i],y[i],z[i],xs,ys,zs)
        phi_yy = kernelyy(x[i],y[i],z[i],xs,ys,zs)
        phi_xy = kernelxy(x[i],y[i],z[i],xs,ys,zs)
        phi_xz = kernelxz(x[i],y[i],z[i],xs,ys,zs)
        phi_yz = kernelyz(x[i],y[i],z[i],xs,ys,zs)
        phi_zz = -phi_xx - phi_yy

        gi = (F0x*mx - F0z*mz)*phi_xx + (F0y*my - F0z*mz)*phi_yy + (F0x*my + F0y*mx)*phi_xy +\
             (F0x*mz + F0z*mx)*phi_xz + (F0y*mz + F0z*my)*phi_yz

        tfa[i] = np.dot(p.T,gi)

    tfa *= CM*T2NT
    return tfa


def sensitivity_mag(x,y,z,xs,ys,zs,sinc,sdec,inc,dec):
    '''
    Calculate the sensitivity matrix

    input

    return
    '''
    N = x.size # number of data
    M = xs.size # number of parameters

    #pM = p[:M] # magnetic moment of eqsources
    #inc,dec = p[M],p[M+1] # magnetization direction of eqsources

    A = np.empty((N,M)) # sensitivity matrix

    F0x,F0y,F0z = ang2vec(1.,sinc,sdec) # main field
    mx,my,mz = ang2vec(1.,inc,dec) # magnetization direction in Cartesian coordinates

    for i in range(N):
        phi_xx = kernelxx(x[i],y[i],z[i],xs,ys,zs)
        phi_yy = kernelyy(x[i],y[i],z[i],xs,ys,zs)
        phi_xy = kernelxy(x[i],y[i],z[i],xs,ys,zs)
        phi_xz = kernelxz(x[i],y[i],z[i],xs,ys,zs)
        phi_yz = kernelyz(x[i],y[i],z[i],xs,ys,zs)
        phi_zz = - phi_xx - phi_yy
        gi = (F0x*mx - F0z*mz)*phi_xx + (F0y*my - F0z*mz)*phi_yy + (F0x*my + F0y*mx)*phi_xy +\
             (F0x*mz + F0z*mx)*phi_xz + (F0y*mz + F0z*my)*phi_yz
        A[i,:] = gi

    A *= CM*T2NT
    return A



def sensitivity_dir(x,y,z,xs,ys,zs,sinc,sdec,p,inc,dec):
    '''
    Calculate the sensitivity matrix

    input

    return
    '''
    N = x.size # number of data
    M = xs.size # number of parameters

    F0x,F0y,F0z = ang2vec(1.,sinc,sdec) # main field
    mx,my,mz = ang2vec(1.,inc,dec) # magnetization direction in Cartesian coordinates

    dmx_I,dmy_I,dmz_I = derivative_inclination(1.,inc,dec)
    dmx_D,dmy_D,dmz_D = derivative_declination(1.,inc,dec)

    dgi_I = np.empty(N,dtype=float)
    for i in range(N):
        phi_xx = kernelxx(x[i],y[i],z[i],xs,ys,zs)
        phi_yy = kernelyy(x[i],y[i],z[i],xs,ys,zs)
        phi_xy = kernelxy(x[i],y[i],z[i],xs,ys,zs)
        phi_xz = kernelxz(x[i],y[i],z[i],xs,ys,zs)
        phi_yz = kernelyz(x[i],y[i],z[i],xs,ys,zs)
        phi_zz = - phi_xx - phi_yy
        gi = (F0x*dmx_I - F0z*dmz_I)*phi_xx + (F0y*dmy_I - F0z*dmz_I)*phi_yy + (F0x*dmy_I + F0y*dmx_I)*phi_xy +\
             (F0x*dmz_I + F0z*dmx_I)*phi_xz + (F0y*dmz_I + F0z*dmy_I)*phi_yz
        dgi_I[i] = np.dot(p.T,gi)

    dgi_D = np.empty(N,dtype=float)
    for i in range(N):
        phi_xx = kernelxx(x[i],y[i],z[i],xs,ys,zs)
        phi_yy = kernelyy(x[i],y[i],z[i],xs,ys,zs)
        phi_xy = kernelxy(x[i],y[i],z[i],xs,ys,zs)
        phi_xz = kernelxz(x[i],y[i],z[i],xs,ys,zs)
        phi_yz = kernelyz(x[i],y[i],z[i],xs,ys,zs)
        phi_zz = - phi_xx - phi_yy
        gi = (F0x*dmx_D - F0z*dmz_D)*phi_xx + (F0y*dmy_D - F0z*dmz_D)*phi_yy + (F0x*dmy_D + F0y*dmx_D)*phi_xy +\
             (F0x*dmz_D + F0z*dmx_D)*phi_xz + (F0y*dmz_D + F0z*dmy_D)*phi_yz
        dgi_D[i] = np.dot(p.T,gi)


    A = np.column_stack([dgi_I,dgi_D])
    A *= CM*T2NT

    return A

############################# Implementing the inverse problem ########################################

def levenberg_marquardt(dobs,x,y,z,xs,ys,zs,sinc,sdec,p,inc0,dec0,lamb,dlamb,itext,itmarq,eps_i,mu,f0):
    '''
    Apply the Levenberg-Marquardt Method to solve a non-linear problem.

    input

    dobs: numpy array - Observed Total field anomaly vector
    x,y,z : numpy arrays - Cartesian coordinates of observations
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources
    sinc,sdec : float - Direction of Main field
    p : numpy array - magnetic moment of equivalent sources
    inc0,dec0: float - magnetization direction of equivalent sources
    itmax : integer - number of iterations

    return

    inc,dec : float - magnetization direction

    '''
    
    for i in range(itext):
            tf0 = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p,inc0,dec0)
            r0 = dobs - tf0
            phi0 = np.linalg.norm(r0) + mu*f0*np.linalg.norm(p)

            G_dir = sensitivity_dir(x,y,z,xs,ys,zs,sinc,sdec,p,inc0,dec0)
            J = -np.dot(G_dir.T,r0)
            H =  np.dot(G_dir.T,G_dir)
            
            for j in range(itmarq):
                dp = np.linalg.solve(H + lamb*np.identity(2),-J)

                dp_inc = np.rad2deg(dp[0])
                dp_dec = np.rad2deg(dp[1])

                inc = inc0 + dp_inc
                dec = dec0 + dp_dec

                tf = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p,inc,dec)
                r = dobs - tf
                phi = np.linalg.norm(r) + mu*f0*np.linalg.norm(p)

                dphi = phi-phi0

                if (dphi > 0.):
                    lamb *= dlamb
                else:
                    lamb /= dlamb
                    break

            condition_1 = np.abs(dphi)/phi0
            if (condition_1 < eps_i):
                break
            else:
                r0[:] = r[:]
                phi0 = phi
                tf0[:] = tf[:]
                inc0 = inc
                dec0 = dec
    
    return inc,dec,phi0
          

def LM_NNLS(dobs,x,y,z,xs,ys,zs,sinc,sdec,inc0,dec0,lamb,dlamb,imax,itext,itmarq,eps_e,eps_i,mu):
    '''
    (REFAZER ESTA DESCRICAO)
    Apply the Levenberg-Marquardt Method to solve a non-linear problem.

    input

    dobs: numpy array - Observed Total field anomaly vector
    x,y,z : numpy arrays - Cartesian coordinates of observations
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources
    sinc,sdec : float - Direction of Main field
    p : numpy array - magnetic moment of equivalent sources
    inc,dec: float - magnetization direction of equivalent sources
    itmax : integer - number of iterations

    return

    inc0,dec0 : float - magnetic moment estimation

    '''
    N = x.size
    M = xs.size
    null = np.zeros(M)
    I = np.identity(M)
    do = np.hstack([dobs,null])
    
    phi_it = []
    iteration = []
    pest = []
    incs = [inc0]
    decs = [dec0]
    
    for i in range(imax):
        print 'i =', i
        
        G_mag = sensitivity_mag(x,y,z,xs,ys,zs,sinc,sdec,inc0,dec0)
        f0 = np.trace(np.dot(G_mag.T,G_mag))/M
        GI = np.vstack([G_mag,mu*f0*I])
        p0,_ = nnls(GI,do)
        tf_ext = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)
        r_ext = dobs - tf_ext
        phi_ext = np.linalg.norm(r_ext) + mu*f0*np.linalg.norm(p0)
        
        ## Levenberg-Marquardt method ##
        inc0,dec0,phi0 = levenberg_marquardt(dobs,x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0,lamb,dlamb,itext,itmarq,eps_i,mu,f0)
        ################################
        print inc0,dec0
        
        phi_it.append(phi0)
        iteration.append(i)
        pest.append(p0)
        incs.append(inc0)
        decs.append(dec0)
        
        dphi_ext = phi_ext - phi0
        condition_2 = np.abs(dphi_ext)/phi0
        print condition_2
        if (condition_2 < eps_e):
            break
                       
    return p0,inc0,dec0,phi_it,i,pest,incs,decs

def residual(do,dp):
    r = do - dp
    r_mean = np.mean(r)
    r_std = np.std(r)
    r_norm = (r - r_mean)/r_std
    return r_norm, r_mean, r_std

