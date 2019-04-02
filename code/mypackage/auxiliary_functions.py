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
    #dmz =   1e-8
    dmz = 0.

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



def sensitivity_mag_dir(x,y,z,xs,ys,zs,sinc,sdec,p,inc,dec):
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

    A = np.empty((N,M)) # sensitivity matrix
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

    A = np.column_stack((A,dgi_I,dgi_D))
    A *= CM*T2NT
    return A

############################# Implementing the inverse problem ########################################

def gauss_newton_LB_linear(dobs,x,y,z,xs,ys,zs,sinc,sdec,p0,inc,dec,itmax,lamb,eps):
    '''
    Apply the Gauss-Newton for solving a linear problem with log barrier.

    input

    dobs: numpy array - Observed Total field anomaly vector
    x,y,z : numpy arrays - Cartesian coordinates of observations
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources
    sinc,sdec : float - Direction of Main field
    p : numpy array - magnetic moment of equivalent sources
    inc,dec: float - magnetization direction of equivalent sources
    itmax : integer - number of iterations
    lamb : float - barrier parameter
    eps: float - stop criterion for convergence

    return

    p0 : array - magnetic moment estimation

    '''
    eps = 1e-8
    N = x.size
    M = xs.size

    p_est = []
    pred = []
    phi_it = []
    iteration  = []

    G = sensitivity_mag(x,y,z,xs,ys,zs,sinc,sdec,inc,dec)

    for i in range(itmax):
        print i

        tf0 = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p0,inc,dec)
        r0 = dobs - tf0
        phi0 = np.sum(r0*r0) - 2.*lamb*(np.sum(np.log(p0+1e-8)))

        G = sensitivity_mag(x,y,z,xs,ys,zs,sinc,sdec,inc,dec)

        # Solving System for magnetic moment

        X1 = np.diag(1./(p0+1e-8))
        e = np.ones_like((p0+1e-8))
        J = -np.dot(G.T,r0) - lamb*(np.dot(X1,e)) # M+2 x 1

        X2 = np.diag(1./((p0*p0+1e-8)))
        H = np.dot(G.T,G) + lamb*X2

        gamma = 0.925
        dp = np.linalg.solve(H,-J)
        p0 += gamma*dp

        print np.max(p0)
        print np.min(p0)

        tf = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p0,inc,dec)
        r = dobs - tf
        phi = np.sum(r*r) - 2.*lamb*(np.sum(np.log(p0)))

        r0 = r[:]
        phi0 = phi
        tf0 = tf[:]
        lamb = gamma*lamb

        p_est.append(p0)
        pred.append(tf0)
        iteration.append(i)
        phi_it.append(phi0)

    return p0,p_est,pred,iteration,phi_it



def gauss_newton_LB_non_linear(dobs,x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0,itmax,lamb):
    '''
    Apply the Levenberg-Marquardt Method to solve a non-linerar problem with positivity constraint.

    input

    dobs: numpy array - Observed Total field anomaly vector
    x,y,z : numpy arrays - Cartesian coordinates of observations
    xs,ys,zs : numpy arrays - Cartesian coordinates of equivalent sources
    sinc,sdec : float - Direction of Main field
    p : numpy array - magnetic moment of equivalent sources
    inc,dec: float - magnetization direction of equivalent sources
    itmax : integer - number of iterations
    lamb : float - barrier parameter

    return

    p0 : array - magnetic moment estimation

    '''
    eps = 1e-8
    N = x.size
    M = xs.size

    p_est = []
    incs = []
    decs = []
    pred = []
    phi_it = []
    iteration  = []

    for i in range(itmax):
        print i

        tf0 = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0) # N x 1
        r0 = dobs - tf0 # N x 1
        phi0 = np.sum(r0*r0) - 2.*lamb*(np.sum(np.log(p0))) # escalar

        G = sensitivity_mag_dir(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0) # N x M+2
        print G.shape
        G_m = G[:,:M]
        G_dir = G[:,M:M+2]

        print G_dir.shape

        # Solving System for magnetic moment

        X1 = np.diag(1./(p0))
        e = np.ones_like((p0))
        J_m = -np.dot(G_m.T,r0) - lamb*(np.dot(X1,e)) # M+2 x 1

        X2 = np.diag(1./((p0*p0)))
        H_m = np.dot(G_m.T,G_m) + lamb*X2

        dp_m = np.linalg.solve(H_m,-J_m)

        gamma = 0.925

        p0 += gamma*dp_m

        # Solving for the layer direction

        J_dir = -np.dot(G_dir.T,r0)
        H_dir = np.dot(G_dir.T,G_dir)

        dp_dir = np.linalg.solve(H_dir,-J_dir)

        print dp_dir.shape

        dp_inc = np.rad2deg(dp_dir[0])
        dp_dec = np.rad2deg(dp_dir[1])

        inc0 += dp_inc
        dec0 += dp_dec

        print inc0,dec0
        print np.max(p0)
        print np.min(p0)

        tf = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)
        r = dobs - tf
        phi = np.sum(r*r) - 2.*lamb*(np.sum(np.log(p0)))

        #print phi0
        #print phi
        #dphi = phi-phi0
        #condition = np.abs(dphi)/(phi0+1e-8)
        #if (condition <= eps):
        #    break
        #else:
        r0 = r[:]
        phi0 = phi
        tf0 = tf[:]
        lamb = gamma*lamb

        p_est.append(p0)
        incs.append(inc0)
        decs.append(dec0)
        pred.append(tf0)
        iteration.append(i)
        phi_it.append(phi0)

    return p0,inc0,dec0,p_est,incs,decs,pred,iteration,phi_it



def gauss_newton_dir(dobs,x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0,itmax):
    '''
    Apply the Gauss-Newton Method to solve a non-linear problem.

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
    eps = 1e-8
    N = x.size

    incs = []
    decs = []
    for i in range(itmax):
        print i
        tf0 = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)
        r0 = dobs - tf0
        phi0 = np.sum(r0*r0)

        G = sensitivity_dir(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)

        J = -np.dot(G.T,r0)
        H = np.dot(G.T,G)

        dp = np.linalg.solve(H,-J)

        dp_inc = np.rad2deg(dp[0])
        dp_dec = np.rad2deg(dp[1])

        inc0 += dp_inc
        dec0 += dp_dec

        print inc0,dec0

        tf = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)
        r = dobs - tf
        phi = np.sum(r*r)

        r0 = r[:]
        phi0 = phi
        tf0 = tf[:]

        incs.append(inc0)
        decs.append(dec0)

    return inc0,dec0,incs,decs

def gauss_newton_NNLS_1(dobs,x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0,itmax):
    '''
    Apply the Gauss-Newton Method to solve a non-linear problem.

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
    eps = 1e-8
    N = x.size
    M = xs.size


    pest = []
    incs = []
    decs = []
    for i in range(itmax):
        print i
        tf0 = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)
        r0 = dobs - tf0
        phi0 = np.sum(r0*r0)

        G = sensitivity_mag_dir(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)

        dp,_ = nnls(G,r0)
        p0 += dp[:M]
        dp_inc = np.rad2deg(dp[M])
        dp_dec = np.rad2deg(dp[M+1])

        inc0 += dp_inc
        dec0 += dp_dec

        print inc0,dec0

        tf = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)
        r = dobs - tf
        phi = np.sum(r*r)

        r0 = r[:]
        phi0 = phi
        tf0 = tf[:]

        pest.append(p0)
        incs.append(inc0)
        decs.append(dec0)

    return p0,inc0,dec0,pest,incs,decs

def gauss_newton_NNLS_1(dobs,x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0,itmax):
    '''
    Apply the Gauss-Newton Method to solve a non-linear problem.

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


    pest = []
    incs = []
    decs = []
    for i in range(itmax):
        print i
        tf0 = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)
        r0 = dobs - tf0
        phi0 = np.sum(r0*r0)

        G = sensitivity_mag_dir(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)

        dp,_ = nnls(G,r0)
        p0 += dp[:M]
        dp_inc = np.rad2deg(dp[M])
        dp_dec = np.rad2deg(dp[M+1])

        inc0 += dp_inc
        dec0 += dp_dec

        print inc0,dec0

        tf = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)
        r = dobs - tf
        phi = np.sum(r*r)

        r0 = r[:]
        phi0 = phi
        tf0 = tf[:]

        pest.append(p0)
        incs.append(inc0)
        decs.append(dec0)

    return p0,inc0,dec0,pest,incs,decs


def gauss_newton_NNLS_2(dobs,x,y,z,xs,ys,zs,sinc,sdec,inc0,dec0,itmax):
    '''
    Apply the Gauss-Newton Method to solve a non-linear problem.

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
    eps = 1e-6
    N = x.size
    M = xs.size


    pest = []
    incs = []
    decs = []
    for i in range(itmax):
        print i
        ## Step for magnetic moment estimation
        G_mag = sensitivity_mag(x,y,z,xs,ys,zs,sinc,sdec,inc0,dec0)
        p0,_ = nnls(G_mag,dobs)
        p0 = p0 + 1e-8

        ## Calculating residual and misfit
        tf0 = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)
        r0 = dobs - tf0
        phi0 = np.sum(r0*r0)

        ## Step for calculating the increment for direction

        G_dir = sensitivity_dir(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)
        J = -np.dot(G_dir.T,r0)
        H = np.dot(G_dir.T,G_dir)

        dp = np.linalg.solve(H,-J)
        dp_inc = np.rad2deg(dp[0])
        dp_dec = np.rad2deg(dp[1])

        inc0 += dp_inc
        dec0 += dp_dec

        print inc0,dec0

        tf = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)
        r = dobs - tf
        phi = np.sum(r*r)

        dphi = phi-phi0
        condition = np.abs(dphi)/(phi0)
        if (condition <= eps):
            break
        else:
            r0 = r[:]
            phi0 = phi
            tf0 = tf[:]

            pest.append(p0)
            incs.append(inc0)
            decs.append(dec0)

    return p0,inc0,dec0,pest,incs,decs

def levenberg_marquardt_NNLS(dobs,x,y,z,xs,ys,zs,sinc,sdec,inc0,dec0,lamb,dlamb,imax,itext,itmarq,eps_e,eps_i,mu):
    '''
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

        phi_ext = np.sum(r_ext*r_ext)
        
        for j in range(itext):
            tf0 = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)
            r0 = dobs - tf0
            phi0 = np.sum(r0*r0)

            G_dir = sensitivity_dir(x,y,z,xs,ys,zs,sinc,sdec,p0,inc0,dec0)
            J = -np.dot(G_dir.T,r0)
            H =  np.dot(G_dir.T,G_dir)
            
            for k in range(itmarq):
                dp = np.linalg.solve(H + lamb*np.identity(2),-J)

                dp_inc = np.rad2deg(dp[0])
                dp_dec = np.rad2deg(dp[1])

                inc = inc0 + dp_inc
                dec = dec0 + dp_dec

                tf = tfa_layer(x,y,z,xs,ys,zs,sinc,sdec,p0,inc,dec)
                r = dobs - tf
                phi = np.sum(r*r)

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
                        
        phi_it.append(phi0)
        iteration.append(i)
        pest.append(p0)
        incs.append(inc0)
        decs.append(dec0)

        print inc0,dec0
        
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



#mask = (dp>0)
        #if dp.size == dp[mask].size:
        #    beta = 1.
        #else:
        #    beta= np.min(p0/(np.sum(dp*dp)))
        #lamb = lamb*(1.- np.min([gamma,beta]))
        #d = tf_layer(x,y,z,layer,p0 + gamma*beta*dp,sinc,sdec,inc,dec)
