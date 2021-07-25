import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import space_filling_decomp_new as sfc
import sys
import vtk, vtktools
import numpy as np
import time

# calculate the space-filling curve for the mesh and convert to csv format

#----------------------------------------------------------------------------------------------
# get a mesh from the results vtus
#path = '/home/cheaney/Results/nirom_test_fpc_nPOD_20_nSnap_100/snapshots/'

def convert_vtu_to_csv(path):
    # path = '../data/FPC_Re3900_DG_old'
    filename = path + 'Flowpast_2d_Re3900_0.vtu' # this file will do (all files have the same mesh)
    vtu_data = vtktools.vtu(filename)
    coords = vtu_data.GetLocations()
    nNodes = coords.shape[0]
    
    # sfc settings    
    findm,colm,ncolm = sfc.form_spare_matric_from_pts( coords, nNodes )
    ncurve = 2
    graph_trim = -10  # has always been set at -10
    starting_node = 0 # =0 do not specifiy a starting node, otherwise, specify the starting node
    colm1 = np.ones((ncolm),dtype = 'int')
    colm1 = colm[0:ncolm]
        
    # call the sfc fortran code (space_filling_decomp_new.so)
    whichd, space_filling_curve_numbering = sfc.ncurve_python_subdomain_space_filling_curve(colm1, findm, starting_node, graph_trim, ncurve, nNodes, ncolm)

    N = len(space_filling_curve_numbering)
    inverse_numbering = np.zeros((N, ncurve), dtype=np.int)

    # fortran numbers from 1 to N :-) whereas python numbers from 0 to N-1
    space_filling_curve_numbering = space_filling_curve_numbering - 1
    
    inverse_numbering[:, 0] = np.argsort(space_filling_curve_numbering[:, 0])
    inverse_numbering[:, 1] = np.argsort(space_filling_curve_numbering[:, 1])
    
    # read in data and save, with sfc ordering, to csv file   

    t_save_to_csv = 0
    t_read_in = 0

    # data for training, validation and test
    nTotalExamples = 1000 # maximum is 1000 for this data set

    #cwd = os.getcwd()
    if not os.path.isdir('csv_data'):
        os.mkdir('csv_data')  
    #os.chdir('csv_data') # will overwrite files in results

    for data_i in range(nTotalExamples):
        t0 = time.time()    
        # path = '/home/cheaney/Results/nirom_test_fpc_nPOD_20_nSnap_100/snapshots/'
        filename = path + 'Flowpast_2d_Re3900_' + str(data_i) + '.vtu'
        vtu_data = vtktools.vtu(filename)
        # D[:,0] and D[:,1] store order1 and order2, D[:,2] store velocity

        t1 = time.time()
        t_read_in = t_read_in + t1 - t0

        D = np.zeros((nNodes, 5))

        D[:, :2] = inverse_numbering # this seems to be transformed to reals - should be integers

        velocity = vtu_data.GetField('Velocity')
        D[:, 2] = np.sqrt(velocity[:nNodes, 0]**2 + velocity[:nNodes, 1]**2) # not needed
        D[:, 3] = velocity[:nNodes, 0]
        D[:, 4] = velocity[:nNodes, 1]

        t0 = time.time()
        np.savetxt('csv_data/data_' + str(data_i) + '.csv', D , delimiter=',')
        t1 = time.time()
        t_save_to_csv = t_save_to_csv + t1 - t0

        print("data loaded", data_i)



    print('Time loading data', t_read_in)
    print('Time to write to csv', t_save_to_csv)

