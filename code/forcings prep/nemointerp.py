import netCDF4 as nc
import numpy as np
import scipy.sparse as sp

def build_matrix(weightsfile, opsfile, xname='x', yname='y'):
    """
    Given a NEMO weights file and an operational surface forcing file, we
    assemble the weights into a sparse interpolation matrix that interpolates
    the surface forcing data from the operational grid to the NEMO grid. This
    function returns the matrix and the NEMO grid shape.

    :arg weightsfile: Path to NEMO weights file.
    :type weightsfile: str

    :arg opsfile: Path to an operational file.
    :type opsfile: str

    :returns: Sparse interpolation matrix and NEMO grid shape
    :rtype: (:class:`~scipy:scipy.sparse.csr_matrix`, :class:`tuple`)
    """
    # Weights
    with nc.Dataset(weightsfile) as f:
        s1 = f.variables['src01'][:]-1  # -1 for fortran-to-python indexing
        s2 = f.variables['src02'][:]-1
        s3 = f.variables['src03'][:]-1
        s4 = f.variables['src04'][:]-1
        w1 = f.variables['wgt01'][:]
        w2 = f.variables['wgt02'][:]
        w3 = f.variables['wgt03'][:]
        w4 = f.variables['wgt04'][:]

    with nc.Dataset(opsfile) as f:
        NO = f.dimensions[xname].size * f.dimensions[yname].size   # number of operational grid points
    NN, nemoshape = s1.size, s1.shape   # number of NEMO grid points and shape of NEMO matrix

    # Build matrix
    n = np.array([x for x in range(0, NN)])
    M1 = sp.csr_matrix((w1.flatten(), (n, s1.flatten())), (NN, NO))
    M2 = sp.csr_matrix((w2.flatten(), (n, s2.flatten())), (NN, NO))
    M3 = sp.csr_matrix((w3.flatten(), (n, s3.flatten())), (NN, NO))
    M4 = sp.csr_matrix((w4.flatten(), (n, s4.flatten())), (NN, NO))
    M = M1+M2+M3+M4
    return M,nemoshape



def use_matrix(matrix, nemoshape, odata=None, opsfile=None, variable=None, time=None):
    """
    Given an operational surface forcing file, an interpolation matrix and
    NEMO grid shape (produced by grid_tools.build_matrix), a variable name and
    a time index, we return the operational data interpolated onto the NEMO
    grid.

    :arg matrix: Interpolation matrix (from build_matrix)
    :type matrix: :class:`~scipy:scipy.sparse.csr_matrix`

    :arg nemoshape: NEMO grid shape (from build_matrix)
    :type nemoshape: tuple

    Now you can provide odata, or, provide opsfile&variable&time

    :arg odata: array of operational surface forcing data
    :rtype: :class:`~numpy:numpy.ndarray`

    :arg opsfile: Path to operational file to interpolate.
    :type opsfile: str

    :arg variable: Specified variable in ops file.
    :type variable: str

    :arg time index: time index in ops file.
    :type time index: integer

    :returns: Operational data interpolated onto the NEMO grid
    :rtype: :class:`~numpy:numpy.ndarray`
    """
    if odata is None:
        with nc.Dataset(opsfile) as f:
            odata = f.variables[variable][time, ...]   # Load the 2D field
  
    # Interpolate by matrix multiply - quite fast
    ndata = matrix*odata.flatten()

    # Reshape to NEMO shaped array
    ndata = ndata.reshape(nemoshape)

    return ndata

