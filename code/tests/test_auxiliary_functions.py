import numpy as np
from numpy.testing import assert_almost_equal
from pytest import raises
import eql_functions as func
import auxiliary_functions as fc
from fatiando.gridder import regular
from fatiando.utils import ang2vec, vec2ang
from fatiando.constants import CM, T2NT

def test_matrix_sensitivity_magnetic_moment():
    'Test for calculation of sensitivity matrix for magnetic moment layer'
    ### Area of observation
    area = [-1500.,1500.,-1500,.1500]
    
    ### Number of point on each axis
    shape = (20,20)
    Nx,Ny = shape[0],shape[1]
    
    ### Observation Height
    z_obs = 0.
    
    ### Inclination and declination main field
    inc_mf,dec_mf = -50.,20.
    
    ### Inclination and declination body
    inc,dec = -60.,30.
    
    ### Observation coordinates
    x,y,z = regular(area,shape,z_obs)
    
    ### Layer depth
    h = 120.
    
    ### Equivalent sources coordinates/model
    xs,ys,zs = regular(area,shape,h)
    layer = func.PointGrid(area,h,shape)
    
    ### Sensitivity eql_functions
    G_eql= func.sensitivity_mag(x,y,z,layer,inc_mf,dec_mf,inc,dec) ## column made
    
    ### Sensitivity auxiliary_functions
    G_aux= fc.sensitivity_mag(x,y,z,xs,ys,zs,inc_mf,dec_mf,inc,dec) ## row made
    
    result = np.allclose(G_eql,G_aux)
    
    assert_almost_equal(result, True, decimal=15)
    
def test_predicted_data():
    'Test for calculation of predicted data eqlayer'
    ### Area of observation
    area = [-1500.,1500.,-1500,.1500]
    
    ### Number of point on each axis
    shape = (20,20)
    Nx,Ny = shape[0],shape[1]
    
    ### Observation Height
    z_obs = 0.
    
    ### Inclination and declination main field
    inc_mf,dec_mf = -50.,20.
    
    ### Inclination and declination body
    inc,dec = -60.,30.
    
    ### Observation coordinates
    x,y,z = regular(area,shape,z_obs)
    
    ### Layer depth
    h = 120.
    
    ### Equivalent sources coordinates/model
    xs,ys,zs = regular(area,shape,h)
    layer = func.PointGrid(area,h,shape)
    
    ### Number of data
    N = x.size
    
    ### Number of parameters
    M = xs.size
    
    ### Array magnetic moment of equivalent sources
    p = np.ones(Nx*Ny)
    
    ### Predicted data using eql_functions
    G_eql= func.sensitivity_mag(x,y,z,layer,inc_mf,dec_mf,inc,dec) ## column made
    tfa_eql = np.dot(G_eql,p)
    
    ### Predicted data using auxiliary_functions
    tfa_aux= fc.tfa_layer(x,y,z,xs,ys,zs,inc_mf,dec_mf,p,inc,dec) ## row made
    
    result = np.allclose(tfa_eql,tfa_aux)
    
    assert_almost_equal(result, True)

def test_coordinate_transformation():
    'Test for calculation transformation coordinate'
    ### Directions in Spherical coordinates
    inc,dec = -60.,30.
    
    ### Generating a vector with ang2vec
    F = ang2vec(1.,inc,dec)
    
    ### Generating a vector using a function within auxiliary function
    F_aux = fc.sph2cart(1.,inc,dec)
    
    
    result = np.allclose(F,F_aux)
    
    assert_almost_equal(result, True)

def test_matrix_sensitivity_sliced_1():
    'Test for calculation of sensitivity matrix for magnetic moment layer'
    ### Area of observation
    area = [-1500.,1500.,-1500,.1500]
    
    ### Number of point on each axis
    shape = (20,20)
    Nx,Ny = shape[0],shape[1]
    
    ### Observation Height
    z_obs = 0.
    
    ### Inclination and declination main field
    inc_mf,dec_mf = -50.,20.
    
    ### Inclination and declination body
    inc,dec = -60.,30.
    
    ### Observation coordinates
    x,y,z = regular(area,shape,z_obs)
    
    ### Layer depth
    h = 120.
    
    ### Equivalent sources coordinates/model
    xs,ys,zs = regular(area,shape,h)
      
    ### parameter vector
    p = np.ones(Nx*Ny)
    
    N = x.size
    M = xs.size
    
    ### Sensitivity eql_functions
    G_m = fc.sensitivity_mag(x,y,z,xs,ys,zs,inc_mf,dec_mf,inc,dec) ## column made
    
    ### Sensitivity auxiliary_functions
    G = fc.sensitivity_mag_dir_positivity(x,y,z,xs,ys,zs,inc_mf,dec_mf,p,inc,dec) ## row made
    
    G_sliced = G[:,:M]
    
    result = np.allclose(G_m,G_sliced)
    
    assert_almost_equal(result, True, decimal=15)

    
def test_matrix_sensitivity_sliced_2():
    'Test for calculation of sensitivity matrix for magnetic moment layer'
     
    ### Area of observation
    area = [-1500.,1500.,-1500,.1500]
    
    ### Number of point on each axis
    shape = (20,20)
    Nx,Ny = shape[0],shape[1]
    
    ### Observation Height
    z_obs = 0.
    
    ### Inclination and declination main field
    inc_mf,dec_mf = -50.,20.
    
    ### Inclination and declination body
    inc,dec = -60.,30.
    
    ### Observation coordinates
    x,y,z = regular(area,shape,z_obs)
    
    ### Layer depth
    h = 120.
    
    ### Equivalent sources coordinates/model
    xs,ys,zs = regular(area,shape,h)
      
    ### parameter vector
    p = np.ones(Nx*Ny)
    
    N = x.size
    M = xs.size
    
    ### Sensitivity eql_functions
    G_dir = fc.sensitivity_dir(x,y,z,xs,ys,zs,inc_mf,dec_mf,p,inc,dec) ## column made
    
    ### Sensitivity auxiliary_functions
    G = fc.sensitivity_mag_dir_positivity(x,y,z,xs,ys,zs,inc_mf,dec_mf,p,inc,dec) ## row made
    
    G_sliced = G[:,M:M+2]
    
    result = np.allclose(G_dir,G_sliced)
    
    assert_almost_equal(result, True, decimal=15)





  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    