import gzip
import os
import tempfile

import numpy as np
from scipy.sparse import coo_matrix

from ..io import read_10x_mtx


# def test_read_10x_mtx():
#     # create an example 10x mtx file
#     data = np.array([1, 2, 3, 4, 5])
#     row = np.array([0, 0, 1, 2, 2])
#     col = np.array([1, 1, 1, 0, 1])
#     mtx = coo_matrix((data, (row, col)))
#     # write the mtx file to disk
#     mtx_dir = tempfile.mkdtemp()
#     mtx = os.path.join(mtx_dir, "matrix.mtx")
#     compressed = mtx + ".gz"
#     with open(mtx, "wb") as f:
#         np.savetxt(f, np.array([3, 2, 5]), fmt="%d")
#         np.savetxt(f, np.column_stack((row, col, data)), fmt="%d")
#     # compress the mtx filem, remove the uncompressed mtx file
#     with open(mtx, "rb") as f_in:
#         with gzip.open(compressed, "wb") as f_out:
#             f_out.writelines(f_in)
#     os.remove(mtx)
#     # read the mtx file using the read_10x_mtx function
#     adata = read_10x_mtx(mtx_dir)
#     # check that the resulting AnnData object has the correct shape
#     assert adata.shape == (3, 2)
#     # check that the resulting AnnData object has the correct values
#     expected_X = np.array([[1, 2], [0, 3], [4, 5]])
#     assert np.allclose(adata.X.toarray(), expected_X)
#     # clean up the test files
#     os.remove(compressed)
