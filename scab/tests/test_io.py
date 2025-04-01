import os

import pytest

from ..io import read_10x_mtx

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture
def sc5p_mtx_path():
    return os.path.join(TEST_DATA_DIR, "sc5p_v2_hs_PBMC_1k/filtered_feature_bc_matrix")


@pytest.fixture
def sc5p_bcr_fasta_path():
    return os.path.join(TEST_DATA_DIR, "sc5p_v2_hs_PBMC_1k/bcr/filtered_contig.fasta")


@pytest.fixture
def sc5p_bcr_fastq_path():
    return os.path.join(TEST_DATA_DIR, "sc5p_v2_hs_PBMC_1k/bcr/filtered_contig.fastq")


@pytest.fixture
def sc5p_bcr_airr_path():
    return os.path.join(TEST_DATA_DIR, "sc5p_v2_hs_PBMC_1k/bcr/airr_rearrangement.tsv")


@pytest.fixture
def sc5p_bcr_parquet_path():
    return os.path.join(
        TEST_DATA_DIR, "sc5p_v2_hs_PBMC_1k/bcr/airr_rearrangement.parquet"
    )


@pytest.fixture
def sc5p_bcr_annot_path():
    return os.path.join(
        TEST_DATA_DIR, "sc5p_v2_hs_PBMC_1k/bcr/filtered_contig_annotations.csv"
    )


@pytest.fixture
def sc5p_tcr_fasta_path():
    return os.path.join(TEST_DATA_DIR, "sc5p_v2_hs_PBMC_1k/bcr/filtered_contig.fasta")


@pytest.fixture
def sc5p_tcr_fastq_path():
    return os.path.join(TEST_DATA_DIR, "sc5p_v2_hs_PBMC_1k/bcr/filtered_contig.fastq")


@pytest.fixture
def sc5p_tcr_annot_path():
    return os.path.join(
        TEST_DATA_DIR, "sc5p_v2_hs_PBMC_1k/bcr/filtered_contig_annotations.csv"
    )


def test_read_10x_mtx(sc5p_mtx_path):
    # create an example 10x mtx file
    adata = read_10x_mtx(sc5p_mtx_path)
    assert adata is not None
    assert adata.shape[0] > 0
    assert adata.shape[1] > 0
    assert adata.X.shape == (adata.shape[0], adata.shape[1])


def test_read_10x_mtx_with_bcr_fasta(
    sc5p_mtx_path, sc5p_bcr_fasta_path, sc5p_bcr_annot_path
):
    adata = read_10x_mtx(
        mtx_path=sc5p_mtx_path,
        bcr_file=sc5p_bcr_fasta_path,
        bcr_annot=sc5p_bcr_annot_path,
    )
    assert adata is not None
    assert adata.shape[0] > 0
    assert adata.shape[1] > 0
    assert adata.X.shape == (adata.shape[0], adata.shape[1])
    assert "bcr" in adata.obs.columns


def test_read_10x_mtx_with_bcr_fastq(
    sc5p_mtx_path, sc5p_bcr_fastq_path, sc5p_bcr_annot_path
):
    adata = read_10x_mtx(
        mtx_path=sc5p_mtx_path,
        bcr_file=sc5p_bcr_fastq_path,
        bcr_annot=sc5p_bcr_annot_path,
    )
    assert adata is not None
    assert adata.shape[0] > 0
    assert adata.shape[1] > 0
    assert adata.X.shape == (adata.shape[0], adata.shape[1])
    assert "bcr" in adata.obs.columns


def test_read_10x_mtx_with_bcr_airr(
    sc5p_mtx_path, sc5p_bcr_airr_path, sc5p_bcr_annot_path
):
    adata = read_10x_mtx(
        mtx_path=sc5p_mtx_path,
        bcr_file=sc5p_bcr_airr_path,
        bcr_annot=sc5p_bcr_annot_path,
        bcr_format="airr",
    )
    assert adata is not None
    assert adata.shape[0] > 0
    assert adata.shape[1] > 0
    assert adata.X.shape == (adata.shape[0], adata.shape[1])
    assert "bcr" in adata.obs.columns


def test_read_10x_mtx_with_bcr_parquet(
    sc5p_mtx_path, sc5p_bcr_parquet_path, sc5p_bcr_annot_path
):
    adata = read_10x_mtx(
        mtx_path=sc5p_mtx_path,
        bcr_file=sc5p_bcr_parquet_path,
        bcr_annot=sc5p_bcr_annot_path,
        bcr_format="parquet",
    )
    assert adata is not None
    assert adata.shape[0] > 0
    assert adata.shape[1] > 0
    assert adata.X.shape == (adata.shape[0], adata.shape[1])
    assert "bcr" in adata.obs.columns
