import pytest
import anndata

import numpy as np

from ..tools.embeddings import pca, umap


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
    return adata


def test_pca(adata):
    # run the pca function
    adata = pca(adata, solver="arpack", n_pcs=2, ignore_ig=True, verbose=False)
    # check that the shape of the data matrix is unchanged
    assert adata.X.shape == (100, 200)
    # check that the PCA coordinates are present in the obsm attribute
    assert "X_pca" in adata.obsm.keys()
    # check that the PCs are present in the varm attribute
    assert "PCs" in adata.varm.keys()
    # check that the variance ratio is present in the uns attribute
    assert "variance_ratio" in adata.uns["pca"].keys()
    # check that the variance is present in the uns attribute
    assert "variance" in adata.uns["pca"].keys()


def test_umap(adata):
    # run the umap function
    adata = umap(
        adata,
        solver="arpack",
        n_neighbors=2,
        n_pcs=2,
        force_pca=True,
        ignore_ig=True,
        paga=False,
        use_rna_velocity=False,
        use_rep=None,
        random_state=42,
        resolution=1.0,
        verbose=False,
    )
    # check that the shape of the data matrix is unchanged
    assert adata.X.shape == (4, 2)
    # check that the PCA coordinates are present in the obsm attribute
    assert "X_pca" in adata.obsm.keys()
    # check that the neighbor graph is present in the uns attribute
    assert "neighbors" in adata.uns.keys()
    # check that the leiden clustering is present in the obs attribute
    assert "leiden" in adata.obs.keys()
    # check that the UMAP coordinates are present in the obsm attribute
    assert "X_umap" in adata.obsm.keys()
