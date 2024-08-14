import os
from pathlib import Path
# import logging
# logger = logging.getLogger(__name__)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--numrows", type=int, dest='numrows', help="Number of rows in the response matrix")
parser.add_argument("--numcols", type=int, dest='numcols', help="Number of columns in the response matrix")
parser.add_argument("--iteration_max", type=int, dest='iteration_max', help="Maximum number of iterations in RL deconvolution")
parser.add_argument("--data_dir", type=str, dest='data_dir', help="Directory where data lies")
parser.add_argument("--save_results", type=bool, dest='save_results', help="Should the results be saved?")
parser.add_argument("--results_dir", type=str, dest='results_dir', help="Directory to save results (only enabled if --save_results is set to True)")
args = parser.parse_args()

# Import third party libraries
import numpy as np
from mpi4py import MPI
import h5py

# Define the number of rows and columns
NUMROWS = args.numrows        # TODO: Ideally, for row-major form to exploit caching, NUMROWS must be smaller than NUMCOLS
NUMCOLS = args.numcols

# Define MPI and iteration misc variables
MASTER = 0                      # Indicates master process
MAXITER = args.iteration_max    # Maximum number of iterations

# FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = Path(args.data_dir)
RESULTS_DIR = Path(args.result_dir)

'''
Response matrix
'''
def load_response_matrix(comm, start_row, end_row, filename='psr_gal_flattened_Ti44_E_1150_1164keV_DC2.h5'):
    with h5py.File(DATA_DIR / filename, "r", driver="mpio", comm=comm) as f1:
        # Assuming the dataset name is "response_matrix"
        dataset = f1["response_matrix"]
        R = dataset[start_row:end_row, :]
    return R

'''
Response matrix transpose
'''
def load_response_matrix_transpose(comm, start_col, end_col, filename='psr_gal_flattened_Ti44_E_1150_1164keV_DC2.h5'):
    with h5py.File(DATA_DIR / filename, "r", driver="mpio", comm=comm) as f1:
        # Assuming the dataset name is "response_matrix"
        dataset = f1["response_matrix"]
        RT = dataset[:, start_col:end_col]
    return RT

'''
Response matrix summed along axis=i
'''
def load_axis0_summed_response_matrix(filename='psr_gal_flattened_Ti44_E_1150_1164keV_DC2.h5'):
    with h5py.File(DATA_DIR / filename, "r") as f1:
        # Assuming the dataset name is "response_vector"
        dataset = f1["response_vector"]
        Rj = dataset[:]
    return Rj

'''
Sky model
'''
def initial_sky_model():
    M0 = np.ones(NUMCOLS, dtype=np.float64) * 1e-4                 # Initial guess according to image_deconvolution.py
    return M0

'''
Background model
'''
def load_bg_model(filename='data/total_bg_dense.hdf5'):
    with h5py.File(DATA_DIR / filename) as hf_bkg:
        bkg = hf_bkg['contents'][:]
    return bkg

'''
Observed data
'''
def load_signal_counts(filename='data/Ti44_CasA_x50_dense.hdf5'):
    with h5py.File(DATA_DIR / filename) as hf_signal:
        signal = hf_signal['contents'][:]
    return signal

def main():
    # Set up MPI
    comm = MPI.COMM_WORLD
    numtasks = comm.Get_size()
    taskid = comm.Get_rank()

    # Initialise vectors required by all processes
    epsilon = np.zeros(NUMROWS)                 # All gatherv-ed. Explicit variable declaration.
    epsilon_fudge = 1e-12                       # To prevent divide-by-zero and underflow errors

    # Initialise epsilon_slice and C_slice. Explicit variable declarations. 
    epsilon_slice = np.zeros(end_row - start_row)
    C_slice = np.zeros(end_col - start_col)

    # Calculate the indices in Rij that the process has to parse. My hunch is that calculating these scalars individually will be faster than the MPI send broadcast overhead.
    averow = NUMROWS // numtasks
    extra_rows = NUMROWS % numtasks
    start_row = taskid * averow
    end_row = (taskid + 1) * averow if taskid < (numtasks - 1) else NUMROWS

    # Calculate the indices in Rji, i.e., Rij transpose, that the process has to parse.
    avecol = NUMCOLS // numtasks
    extra_cols = NUMCOLS % numtasks
    start_col = taskid * avecol
    end_col = (taskid + 1) * avecol if taskid < (numtasks - 1) else NUMCOLS

    # Load R and RT into memory (single time if response matrix doesn't 
    # change with time)
    R = load_response_matrix(comm, start_row, end_row, filename='psr_gal_flattened_511_DC2.h5')
    RT = load_response_matrix_transpose(comm, start_col, end_col, filename='psr_gal_flattened_511_DC2.h5')

    M = np.empty(NUMCOLS, dtype=np.float64)     # Loaded and broadcasted by master. TODO: Correctly link variables to relevant object inputs
    d = np.empty(NUMROWS, dtype=np.float64)     # Loaded and broadcasted by master. 
    bkg = np.zeros(NUMROWS)                     # Loaded and broadcasted by master.

# ****************************** MPI ******************************

# **************************** Part I *****************************

    '''*************** Master ***************'''

    if taskid == MASTER:
        # Pretty print definitions
        linebreak_stars = '**********************'
        linebreak_dashes = '----------------------'

        # Load Rj vector (response matrix summed along axis=i)
        Rj = load_axis0_summed_response_matrix(filename='psr_gal_flattened_511_DC2.h5')

        # Load sky model input
        M = initial_sky_model()     # TODO: Correctly link variables to relevant object inputs

        # Load observed data counts
        signal = load_signal_counts(filename='511_thin_disk_dense.h5')
        bkg = load_bg_model(filename='albedo_bg_dense.h5')     # TODO: Correctly link variables to relevant object inputs
        d = signal + bkg            # TODO: Correctly link variables to relevant object inputs

        # Sanity check: print d
        print()
        print('Observed data-space d vector:')
        print(d)
        ## Pretty print
        print()
        print(linebreak_stars)

        # Initialise C vector. Only master requires full length. Explicit variable declaration.
        C = np.empty(NUMCOLS, dtype=np.float64)

        # Initialise update delta vector. Explicit variable declaration.
        delta = np.empty(NUMCOLS, dtype=np.float64)

    '''*************** Worker ***************'''

    if taskid > MASTER:
        # Only separate if... clause for NON-MASTER processes. 
        # Initialise C vector to None. Only master requires full length.
        C = None

    # Broadcast d vector
    comm.Bcast([d, MPI.DOUBLE], root=MASTER)

    # Scatter bkg vector to epsilon_BG
    comm.Bcast([bkg, MPI.DOUBLE], root=MASTER)
    # comm.Scatter(bkg, [epsilon_BG, recvcounts, displacements, MPI.DOUBLE])

    # print(f"TaskID {taskid}, gathered broadcast")

    # Sanity check: print epsilon
    # if taskid == MASTER:
    #     print('epsilon_BG')
    #     print(bkg)
    #     print()

# **************************** Part IIa *****************************

    '''***************** Begin Iterative Segment *****************'''
    # Set up initial values for iterating variables.
    # Exit if:
    ## 1. Max iterations are reached
    ## 2. M vector converges
    for iter in range(MAXITER):

        '''*************** Master ***************'''
        if taskid == MASTER:
            # Pretty print - starting
            print(f"Starting iteration {iter + 1}")
            # logger.info(f"## Iteration {self.iteration_count}/{self.iteration_max} ##")
            # logger.info("<< E-step >>")


    # Calculate epsilon vector and all gatherv

        '''**************** All *****************'''

        '''Synchronization Barrier 1'''
        # Broadcast M vector
        comm.Bcast([M, MPI.DOUBLE], root=MASTER)

        # Calculate epsilon slice
        epsilon_BG = bkg[start_row:end_row]             # TODO: Change the way epsilon_BG is loaded. Make it taskID dependent through MPI.Scatter for example. Use `recvcounts`
        epsilon_slice = np.dot(R, M) + epsilon_BG + epsilon_fudge

        '''Synchronization Barrier 2'''
        # All vector gather epsilon slices
        recvcounts = [averow] * (numtasks-1) + [averow + extra_rows]
        displacements = np.arange(numtasks) * averow
        comm.Allgatherv(epsilon_slice, [epsilon, recvcounts, displacements, MPI.DOUBLE])

        # Sanity check: print epsilon
        # if taskid == MASTER:
        #     print('epsilon')
        #     print(epsilon)
        #     print(epsilon.min(), epsilon.max())
        #     print()

# **************************** Part IIb *****************************

    # Calculate C vector and gatherv
    
        '''**************** All *****************'''

        # Calculate C slice
        C_slice = np.dot(RT.T, d/epsilon)

        '''Synchronization Barrier 3'''
        # All vector gather C slices
        recvcounts = [avecol] * (numtasks-1) + [avecol + extra_cols]
        displacements = np.arange(numtasks) * avecol
        comm.Gatherv(C_slice, [C, recvcounts, displacements, MPI.DOUBLE], root=MASTER)

# **************************** Part IIc *****************************

    # Iterative update of model-space M vector

        if taskid == MASTER:

            # logger.info("<< M-step >>")

            # Sanity check: print C
            # print('C')
            # print(C)
            # print(C.min(), C.max())
            # print()

            delta = C / Rj - 1
            M = M + delta * M           # Allows for optimization features presented in Siegert et al. 2020

            # Sanity check: print M
            # print('M')
            # print(np.round(M, 5))
            # print(np.round(M.max(), 5))

            # Sanity check: print delta
            # print('delta')
            # print(delta)

            # Pretty print - completion
            print(f"Done")
            print(linebreak_dashes)

            # Save iteration
            # np.savetxt(RESULTS_DIR / f'Mstep{iter+1}.csv', M)

            # MAXITER
            if iter == (MAXITER - 1):
                print(f'Reached maximum iterations = {MAXITER}')
                print(linebreak_stars)
                print()
  
    '''****************** End Iterative Segment ******************'''

    # Print converged M
    if taskid == MASTER:
        # logger.info("<< Registering Result >>")
        print('Converged M vector:')
        print(np.round(M, 5))
        print(np.round(M.max(), 5))
        print(np.sum(M))
        print()

        # Save final output
        # np.savetxt(RESULTS_DIR / f'ConvergedM.csv', M)

    # MPI Shutdown
    MPI.Finalize()
    
if __name__ == "__main__":
    main()
