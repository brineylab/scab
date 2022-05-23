#!/usr/bin/env python
# filename: vdj.py


#
# Copyright (c) 2021 Bryan Briney
# License: The MIT license (http://opensource.org/licenses/MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#


from collections import Counter
import itertools
import sys

import pandas as pd
import numpy as np

from Levenshtein import distance

import fastcluster as fc
from scipy.cluster.hierarchy import fcluster

from mnemonic import Mnemonic

import dnachisel as dc

from natsort import natsorted

from abutils.core.sequence import Sequence
from abutils.utils.cluster import cluster
from abutils.utils.utilities import nested_dict_lookup

from .models.lineage import Lineage


def clonify(
    adata,
    distance_cutoff=0.32,
    shared_mutation_bonus=0.65,
    length_penalty_multiplier=2,
    preclustering=False,
    preclustering_threshold=0.65,
    preclustering_field="cdr3_nt",
    lineage_field="lineage",
    lineage_size_field="lineage_size",
    annotation_format="airr",
    return_assignment_dict=False,
):
    """
    Assigns BCR sequences to clonal lineages using the clonify_ [Briney16]_ algorithm.

    .. seealso::
       | Bryan Briney, Khoa Le, Jiang Zhu, and Dennis R Burton  
       | Clonify: unseeded antibody lineage assignment from next-generation sequencing data.
       | *Scientific Reports* 2016. https://doi.org/10.1038/srep23901

    Parameters  
    ----------
    adata : anndata.AnnData  
        ``AnnData`` object containing annotated sequence data at ``adata.obs.bcr``. If
        data was read using ``scab.read_10x_mtx()``, BCR data should already be in the 
        correct location.

    distance_cutoff : float, default=0.32  
        Distance threshold for lineage clustering.  

    shared_mutation_bonus : float, default=0.65  
        Bonus applied for each shared V-gene mutation.  

    length_penalty_multiplier : int, default=2  
        Multiplier for the CDR3 length penalty. Default is ``2``, resulting in CDR3s that 
        differ by ``n`` amino acids being penalized ``n * 2``.

    preclustering : bool, default=False  
        If ``True``, V/J groups are pre-clustered on the `preclustering_field` sequence, 
        which can potentially speed up lineage assignment and reduce memory usage. 
        If ``False``, each V/J group is processed in its entirety without pre-clustering. 

    preclustering_threshold : float, default=0.65  
        Identity threshold for pre-clustering the V/J groups prior to lineage assignment. 

    preclustering_field : str, default='cdr3_nt'  
        Annotation field on which to pre-cluster sequences.  

    lineage_field : str, default='lineage'  
        Name of the lineage assignment field.  

    lineage_size_field : str, default='lineage_size'  
        Name of the lineage size field.  

    annotation_format : str, default='airr'  
        Format of the input sequence annotations. Choices are ``'airr'`` or ``'json'``.  
        
    return_assignment_dict : bool, default=False  
        If ``True``, a dictionary linking sequence IDs to lineage names will be returned. 
        If ``False``, the input ``anndata.AnnData`` object will be returned, with lineage 
        annotations included.  

    Returns
    -------
    output : ``anndata.AnnData`` or ``dict``
        By default (``return_assignment_dict == False``), an updated `adata` object is \
        returned with two additional columns populated - ``adata.obs.bcr_lineage``, \
        which contains the lineage assignment, and ``adata.obs.bcr_lineage_size``, \
        which contains the lineage size. If ``return_assignment_dict == True``, \
        a ``dict`` mapping droplet barcodes (``adata.obs_names``) to lineage \
        names is returned. 


    .. _clonify: https://github.com/briney/clonify    
    """
    # select the appropriate data fields
    if annotation_format.lower() == "airr":
        vgene_key = "v_gene"
        jgene_key = "j_gene"
        cdr3_key = "cdr3_aa"
        muts_key = "v_mutations"
    elif annotation_format.lower() == "json":
        vgene_key = "v_gene.gene"
        jgene_key = "j_gene.gene"
        cdr3_key = "cdr3_aa"
        muts_key = "var_muts_nt.muts"
    else:
        error = "ERROR: "
        error += f'annotation_format must be either "airr" or "json", but you provided {annotation_format}'
        print("\n")
        print(error)
        print("\n")
        sys.exit()
    # group sequences by V/J genes
    vj_group_dict = {}
    for p in adata.obs.bcr:
        if p.heavy is None:
            continue
        # build new Sequence objects using just the data we need
        h = p.heavy
        s = Sequence(h.sequence, id=p.name)
        s["v_gene"] = nested_dict_lookup(h, vgene_key.split("."))
        s["j_gene"] = nested_dict_lookup(h, jgene_key.split("."))
        s["cdr3"] = nested_dict_lookup(h, cdr3_key.split("."))
        if annotation_format.lower() == "json":
            muts = nested_dict_lookup(h, muts_key.split("."), [])
            s["mutations"] = [f"{m['position']}:{m['was']}>{m['is']}" for m in muts]
        else:
            muts = nested_dict_lookup(h, muts_key.split("."), "").split("|")
            s["mutations"] = [m for m in muts if m.strip()]
        required_fields = ["v_gene", "j_gene", "cdr3", "mutations"]
        if preclustering:
            s["preclustering"] = nested_dict_lookup(h, preclustering_field.split("."))
            required_fields.append("preclustering")
        if any([s[v] is None for v in required_fields]):
            continue
        # group sequences by VJ gene use
        vj = f"{s['v_gene']}__{s['j_gene']}"
        if vj not in vj_group_dict:
            vj_group_dict[vj] = []
        vj_group_dict[vj].append(s)
    # assign lineages
    assignment_dict = {}
    mnemo = Mnemonic("english")
    for vj_group in vj_group_dict.values():
        # preclustering
        if preclustering:
            seq_dict = {s.id: s for s in vj_group}
            cluster_seqs = [Sequence(s[preclustering_field], id=s.id) for s in vj_group]
            clusters = cluster(cluster_seqs, threshold=preclustering_threshold)
            groups = [[seq_dict[i] for i in c.seq_ids] for c in clusters]
        else:
            groups = [
                vj_group,
            ]
        for group in groups:
            if len(group) == 1:
                seq = group[0]
                assignment_dict[seq.id] = "_".join(
                    mnemo.generate(strength=128).split()[:6]
                )
                continue
            # build a distance matrix
            dist_matrix = []
            for s1, s2 in itertools.combinations(group, 2):
                d = _clonify_distance(
                    s1, s2, shared_mutation_bonus, length_penalty_multiplier
                )
                dist_matrix.append(d)
            # cluster
            linkage_matrix = fc.linkage(
                dist_matrix, method="average", preserve_input=False
            )
            cluster_list = fcluster(
                linkage_matrix, distance_cutoff, criterion="distance"
            )
            # rename clusters
            cluster_ids = list(set(cluster_list))
            cluster_names = {
                c: "_".join(mnemo.generate(strength=128).split()[:6])
                for c in cluster_ids
            }
            renamed_clusters = [cluster_names[c] for c in cluster_list]
            # assign sequences
            for seq, name in zip(vj_group, renamed_clusters):
                assignment_dict[seq.id] = name
            lineage_size_dict = Counter(assignment_dict.values())
    # return assignments
    if return_assignment_dict:
        return assignment_dict
    lineage_assignments = [assignment_dict.get(n, np.nan) for n in adata.obs_names]
    lineage_sizes = [lineage_size_dict.get(l, np.nan) for l in lineage_assignments]
    adata.obs[lineage_field] = lineage_assignments
    adata.obs[lineage_size_field] = lineage_sizes
    return adata


def _clonify_distance(s1, s2, shared_mutation_bonus, length_penalty_multiplier):
    if len(s1["cdr3"]) == len(s2["cdr3"]):
        dist = sum([i != j for i, j in zip(s1["cdr3"], s2["cdr3"])])
    else:
        dist = distance(s1["cdr3"], s2["cdr3"])
    length_penalty = abs(len(s1["cdr3"]) - len(s2["cdr3"])) * length_penalty_multiplier
    length = min(len(s1["cdr3"]), len(s2["cdr3"]))
    shared_mutations = list(set(s1["mutations"]) & set(s2["mutations"]))
    mutation_bonus = len(shared_mutations) * shared_mutation_bonus
    score = (dist + length_penalty - mutation_bonus) / length
    return max(score, 0.001)  # distance values can't be negative


def build_synthesis_constructs(
    adata,
    overhang_5=None,
    overhang_3=None,
    annotation_format="airr",
    sequence_key=None,
    locus_key=None,
    name_key=None,
    bcr_key="bcr",
    sort=True,
):
    """
    Builds codon-optimized synthesis constructs, including Gibson overhangs suitable 
    for cloning IGH, IGK and IGL variable region constructs into antibody expression 
    vectors.


    .. seealso:: 
        | Thomas Tiller, Eric Meffre, Sergey Yurasov, Makoto Tsuiji, Michel C Nussenzweig, Hedda Wardemann 
        | Efficient generation of monoclonal antibodies from single human B cells by single cell RT-PCR and expression vector cloning
        | *Journal of Immunological Methods* 2008, doi: 10.1016/j.jim.2007.09.017  


    Parameters
    ----------
    adata : anndata.AnnData  
        An ``anndata.AnnData`` object containing annotated BCR sequences.

    overhang_5 : dict, optional  
        A ``dict`` mapping the locus name to 5' Gibson overhangs. By default, Gibson
        overhangs corresponding to the expression vectors in Tiller et al, 2008:  

            | **IGH:** ``catcctttttctagtagcaactgcaaccggtgtacac``
            | **IGK:** ``atcctttttctagtagcaactgcaaccggtgtacac``
            | **IGL:** ``atcctttttctagtagcaactgcaaccggtgtacac``

        To produce constructs without 5' Gibson overhangs, provide an empty dictionary.

    overhang_3 : dict, optional  
        A ``dict`` mapping the locus name to 3' Gibson overhangs. By default, Gibson
        overhangs corresponding to the expression vectors in Tiller et al, 2008: 

            | **IGH:** ``gcgtcgaccaagggcccatcggtcttcc``
            | **IGK:** ``cgtacggtggctgcaccatctgtcttcatc``
            | **IGL:** ``ggtcagcccaaggctgccccctcggtcactctgttcccgccctcgagtgaggagcttcaagccaacaaggcc``

        To produce constructs without 3' Gibson overhangs, provide an empty dictionary.

    sequence_key : str, default='sequence_aa'  
        Field containing the sequence to be codon optimized. Default is ``'sequence_aa'`` if
        ``annotation_format == 'airr'`` or ``'vdj_aa'`` if ``annotation_format == 'json'``. 
        Either nucleotide or amino acid sequences are acceptable.

    locus_key : str, default='locus'  
        Field containing the sequence locus. Default is ``'locus'`` if ``annotation_key == 'airr'``,
        or ``'chain'`` if ``annotation_key == 'json'``. Note that values in ``locus_key`` should match
        the keys in ``overhang_5`` and ``overhang_3``.

    name_key : str, optional  
        Field (in ``adata.obs``) containing the name of the BCR pair. If not provided, the
        droplet barcode will be used.

    bcr_key : str, default='bcr'  
        Field (in ``adata.obs``) containing the annotated BCR pair.  

    sort : bool, default=True  
        If ``True``, output will be sorted by sequence name.  


    Returns
    -------
    sequences : ``list`` of ``Sequence`` objects 
        A ``list`` of ``abutils.Sequence`` objects. Each ``Sequence`` object has the following
        descriptive properties:  

            | *id*: The sequence ID, which includes the pair name and the locus.  
            | *sequence*: The codon-optimized sequence, including Gibson overhangs.  

        If ``sort == True``, the output ``list``  will be sorted by 
        `name_key` using ``natsort.natsorted()``.

    """
    # if any([locus_key is None, sequence_key is None]):
    #     if annotation_format.lower() == "airr":
    #         sequence_key = sequence_key if sequence_key is not None else "sequence_aa"
    #         locus_key = locus_key if locus_key is not None else "locus"
    #     elif annotation_format.lower() == "json":
    #         sequence_key = sequence_key if sequence_key is not None else "vdj_aa"
    #         locus_key = locus_key if locus_key is not None else "chain"
    #     else:
    #         err = '\nERROR: annotation format must be either "json" or "airr". '
    #         err += f"You provided {annotation_format}\n"
    #         print(err)
    #         sys.exit()
    # get overhangs
    overhang_3 = overhang_3 if overhang_3 is not None else GIBSON3
    overhang_5 = overhang_5 if overhang_5 is not None else GIBSON5
    # parse sequences
    sequences = []
    for i, r in adata.obs.iterrows():
        bcr = r[bcr_key]
        pair_name = r[name_key] if name_key is not None else bcr.name
        for seq in [bcr.heavy, bcr.light]:
            if seq is None:
                continue
            l = seq[locus_key]
            n = f"{pair_name}_{l}"
            optimized = _optimize_codons(seq, sequence_key)
            s = overhang_5.get(l, "") + optimized.sequence + overhang_3.get(l, "")
            opt_seq = Sequence(s, id=n)
            opt_seq[sequence_key] = seq[sequence_key]
            opt_seq[locus_key] = l
            opt_seq["obs_name"] = i
            sequences.append(opt_seq)
    if sort:
        sequences = natsorted(sequences, key=lambda x: x.id)
    return sequences


def _optimize_codons(sequence, sequence_key="vdj_aa"):
    if all(
        [
            res.upper() in ["A", "C", "G", "T", "N", "-"]
            for res in sequence[sequence_key]
        ]
    ):
        dna_seq = sequence[sequence_key]
    else:
        dna_seq = dc.reverse_translate(sequence[sequence_key])
    problem = dc.DnaOptimizationProblem(
        sequence=dna_seq,
        constraints=[
            dc.EnforceTranslation(),
            dc.EnforceGCContent(maxi=0.56),
            dc.EnforceGCContent(maxi=0.64, window=100),
            dc.UniquifyAllKmers(10),
        ],
        objectives=[dc.CodonOptimize(species="h_sapiens")],
        logger=None,
    )
    problem.resolve_constraints(final_check=True)
    problem.optimize()
    return problem


def bcr_summary_csv(
    adata,
    leading_fields=None,
    include=None,
    exclude=None,
    rename=None,
    annotation_format="airr",
    output_file=None,
):
    """
    docstring for bcr_summary_csv.

    Parameters
    ----------
    adata : anndata.AnnData
        An ``anndata.AnnData`` object containing annotated BCR sequences.

    leading_fields : iterable object, optional  
        A list of fields in ``adata.obs`` that should be at the start
        of the output data. By defauolt, the existing column order in 
        ``adata.obs`` is used.  

    include : iterable object, optional 
        A list of columns in ``adata.obs`` that should be included in the 
        summary output. By default, all columns in ``adata.obs`` are used.  

    exclude : iterable object, optional  
        A list of columns in ``adata.obs`` that should be excluded from the 
        summary output. By default, no columns in ``adata.obs`` are excluded.  

    rename : dict, optional  
        A ``dict`` mapping ``adata.obs`` columns to new column names. Any column
        names not included in `rename` will not be renamed.

    annotation_format : str, default='airr'  
        Format of the input sequence annotations. Choices are ``['airr', 'json']``.

    output_file : str, optional  
        Path to the output file. If not provided, the summary output will
        be returned as a Pandas ``DataFrame``.

    
    Returns
    -------
    If ``output_file`` is provided, the summary output will be written to the file in CSV \
    format and noting is returned. If ``output_file`` is not provided, the summary data will \
    be returned as a Pandas ``DataFrame``.

    
    """
    # data fields
    if rename is None:
        rename = {}
    if leading_fields is None:
        leading_fields = []
    if exclude is None:
        exclude = []
    if include is None:
        include = adata.obs.columns
    include = [c for c in include if c not in ["bcr", "tcr"]]
    cols = leading_fields
    cols += [c for c in include if c not in leading_fields + exclude]
    # BCR fields
    if annotation_format.lower() == "airr":
        bcr_fields = AIRR_SUMMARY_FIELDS
    elif annotation_format.lower() == "json":
        bcr_fields = JSON_SUMMARY_FIELDS
    else:
        err = '\nERROR: annotation format must be either "json" or "airr". '
        err += f"You provided {annotation_format}\n"
        print(err)
        sys.exit()
    # parse row data
    data = []
    for i, r in adata.obs.iterrows():
        d = {}
        d["barcode"] = i
        for c in cols:
            d[rename.get(c, c)] = r[c]
        h = r["bcr"].heavy
        l = r["bcr"].light
        for n, seq in [h, l]:
            if seq is None:
                continue
            for f in bcr_fields:
                d[f"{f}:{n}"] = seq.annotations.get(f, "")
        data.append(d)
    df = pd.DataFrame(data)
    if output_file is not None:
        df.to_csv(output_file, index=False)
    else:
        return df


def group_lineages(adata, lineage_names=None, sort_by_size=True, lineage_key="lineage"):
    if lineage_key not in adata.obs:
        err = f"\nERROR: {lineage_key} was not found in adata.obs\n"
        print(err)
        sys.exit()
    if lineage_names is None:
        lineage_names = adata.obs[lineage_key].unique()
    lineages = []
    for lname in lineage_names:
        if pd.isnull(lname):
            continue
        _adata = adata[adata.obs[lineage_key] == lname]
        lineages.append(Lineage(_adata))
    if sort_by_size:
        lineages = sorted(lineages, key=lambda x: x.size, reverse=True)
    return lineages


GIBSON5 = {
    "IGH": "catcctttttctagtagcaactgcaaccggtgtacac",
    "IGK": "atcctttttctagtagcaactgcaaccggtgtacac",
    "IGL": "atcctttttctagtagcaactgcaaccggtgtacac",
    "heavy": "catcctttttctagtagcaactgcaaccggtgtacac",
    "kappa": "atcctttttctagtagcaactgcaaccggtgtacac",
    "lambda": "atcctttttctagtagcaactgcaaccggtgtacac",
}

GIBSON3 = {
    "IGH": "gcgtcgaccaagggcccatcggtcttcc",
    "IGK": "cgtacggtggctgcaccatctgtcttcatc",
    "IGL": "ggtcagcccaaggctgccccctcggtcactctgttcccgccctcgagtgaggagcttcaagccaacaaggcc",
    "heavy": "gcgtcgaccaagggcccatcggtcttcc",
    "kappa": "cgtacggtggctgcaccatctgtcttcatc",
    "lambda": "ggtcagcccaaggctgccccctcggtcactctgttcccgccctcgagtgaggagcttcaagccaacaaggcc",
}


AIRR_SUMMARY_FIELDS = [
    "v_gene",
    "d_gene",
    "j_gene",
    "junction_aa",
    "cdr3_length",
    "fr1_aa",
    "cdr1_aa",
    "fr2_aa",
    "cdr2_aa",
    "fr3_aa",
    "cdr3_aa",
    "fr4_aa",
    "v_identity",
    "v_identity_aa",
    "v_mutations",
    "v_mutations_aa",
    "v_insertions",
    "v_deletions",
    "isotype",
    "locus",
    "sequence",
    "sequence_aa",
    "raw_input",
]

JSON_SUMMARY_FIELDS = [
    "v_gene.gene",
    "d_gene.gene",
    "j_gene.gene",
    "junction_aa",
    "cdr3_len",
    "fr1_aa",
    "cdr1_aa",
    "fr2_aa",
    "cdr2_aa",
    "fr3_aa",
    "cdr3_aa",
    "fr4_aa",
    "nt_identity.v",
    "aa_identity.v",
    "v_insertions",
    "v_deletions",
    "isotype",
    "chain",
    "vdj_nt",
    "vdj_aa",
    "raw_input",
]
