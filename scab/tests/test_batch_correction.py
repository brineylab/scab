import pytest
import numpy as np
import scanpy as sc
import anndata

from ..tools.batch_correction import combat, harmony, mnn, scanorama


@pytest.fixture
def adata():
    # create an example AnnData object
    adata = anndata.AnnData(
        X=np.random.randint(0, 10, size=(100, 200)),
        obs={"batch": ["A"] * 50 + ["B"] * 50},
    )
    adata.raw = adata
    adata.var["highly_variable"] = [True] * 100 + [False] * 100
    adata.var["ig"] = [False] * 200
    adata.obsm["X_pca"] = np.random.randint(0, 10, size=(100, 50))
    return adata


def test_combat(adata):
    # run the combat function
    adata = combat(adata, batch_key="batch", post_correction_umap=False, verbose=True)
    # check that the shape of the data matrix is unchanged
    assert adata.X.shape == (100, 200)
    # check that the batch column is still present
    assert "batch" in adata.obs.columns
    # check that the batch column has the correct values
    batch_counts = adata.obs["batch"].value_counts()
    assert batch_counts["A"] == 50
    assert batch_counts["B"] == 50
    # assert list(adata.obs["batch"]) == ["A", "A", "B", "B"]


# def test_harmony(adata):
#     # run the harmony function
#     adata = harmony(adata, batch_key="batch", post_correction_umap=False, verbose=True)
#     # check that the shape of the data matrix is unchanged
#     assert adata.X.shape == (100, 200)
#     # check that the batch column is still present
#     assert "batch" in adata.obs.columns
#     # check that the batch column has the correct values
#     batch_counts = adata.obs["batch"].value_counts()
#     assert batch_counts["A"] == 50
#     assert batch_counts["B"] == 50
#     # assert list(adata.obs["batch"]) == ["A", "A", "B", "B"]


# def test_mnn(adata):
#     # run the mnn function
#     adata = mnn(adata, batch_key="batch", post_correction_umap=False, verbose=True)
#     # check that the shape of the data matrix is unchanged
#     assert adata.X.shape == (4, 2)
#     # check that the batch column is still present
#     assert "batch" in adata.obs.columns
#     # check that the batch column has the correct values
#     assert list(adata.obs["batch"]) == ["A", "A", "B", "B"]


# def test_scanorama(adata):
#     # run the scanorama function
#     adata = scanorama(
#         adata, batch_key="batch", post_correction_umap=False, verbose=True
#     )
#     # check that the shape of the data matrix is unchanged
#     assert adata.X.shape == (4, 2)
#     # check that the batch column is still present
#     assert "batch" in adata.obs.columns
#     # check that the batch column has the correct values
#     assert list(adata.obs["batch"]) == ["A", "A", "B", "B"]
