import numpy as np

import pandas as pd
from pandas.testing import assert_series_equal

from anndata import AnnData

from ..tools.cellhashes import _bw_silverman, positive_feature_cutoff, demultiplex


def test_bw_silverman():
    # create an example array
    x = np.array([1, 2, 3, 4, 5])
    # calculate the bandwidth using the _bw_silverman function
    bw = _bw_silverman(x)
    # check that the bandwidth is a float
    assert isinstance(bw, float)
    # check that the bandwidth is greater than zero
    assert bw > 0


def test_positive_feature_cutoff():
    # create an example array
    vals = np.array([1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 5, 5])
    # calculate the positive feature cutoff using the positive_feature_cutoff function
    cutoff = positive_feature_cutoff(
        vals,
        threshold_maximum=5.0,
        threshold_minimum=2.0,
        kde_minimum=0.0,
        kde_maximum=10.0,
        debug=False,
        show_cutoff_value=False,
        cutoff_text="cutoff",
        debug_figfile=None,
    )
    # check that the cutoff is a float or None
    assert isinstance(cutoff, float)
    # check that the cutoff is within the specified range
    assert cutoff >= 2.0 and cutoff <= 5.0
    # if cutoff is not None:
    #     assert cutoff >= 2.0 and cutoff <= 8.0


def test_positive_feature_cutoff_no_minima():
    # create an example array without any minima
    vals = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
    # calculate the positive feature cutoff using the positive_feature_cutoff function
    cutoff = positive_feature_cutoff(
        vals,
        threshold_maximum=5.0,
        threshold_minimum=2.0,
        kde_minimum=0.0,
        kde_maximum=10.0,
        debug=False,
        show_cutoff_value=False,
        cutoff_text="cutoff",
        debug_figfile=None,
    )
    # check that no cutoff was found
    assert cutoff is None


def test_positive_feature_cutoff_out_of_bounds():
    # create an example array with a cutoff that is out of bounds
    vals = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 8, 8, 8])
    # calculate the positive feature cutoff using the positive_feature_cutoff function
    cutoff = positive_feature_cutoff(
        vals,
        threshold_maximum=5.0,
        threshold_minimum=2.0,
        kde_minimum=0.0,
        kde_maximum=15.0,
        debug=False,
        show_cutoff_value=False,
        cutoff_text="cutoff",
        debug_figfile=None,
    )
    # check that the cutoff is None
    assert cutoff is None


def test_demultiplex():
    # create an example AnnData object with multiple hashes
    obs = pd.DataFrame(
        {
            "cellhash1": [1, 2, 1, 8, 9],
            "cellhash2": [6, 7, 1, 1, 7],
            "batch": ["A", "A", "A", "B", "B"],
        }
    )
    adata = AnnData(obs=obs)
    # run the demultiplex function
    result = demultiplex(
        adata,
        threshold_minimum=1.0,
        threshold_maximum=10.0,
        debug=False,
    )
    # check that the resulting AnnData object has the correct shape
    assert result.obs.shape == (5, 4)
    # check that the resulting batch assignments are correct
    expected_assignments = pd.Series(
        ["cellhash2", "cellhash2", "unassigned", "cellhash1", "doublet"],
        index=result.obs_names,
        name="cellhash_assignment",
    )
    assert_series_equal(result.obs["cellhash_assignment"], expected_assignments)
