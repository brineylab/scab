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
import random
import string
import sys

import pandas as pd
import numpy as np

from Levenshtein import distance

import fastcluster as fc
from scipy.cluster.hierarchy import fcluster

from abutils.core.pair import Pair
from abutils.core.sequence import Sequence
from abutils.utils.cluster import cluster
from abutils.utils.utilities import nested_dict_lookup



def merge_vdj(adata, vdj_file, tenx_annotation_csv=None, high_confidence=True,
              vdj_delimiter='\t', vdj_id_key='seq_id', vdj_sequence_key='vdj_nt'):
    '''
    Merges VDJ information into an AnnData object containing gene expression and/or feature barcode data.

    Args:
    -----

        adata (anndata.AnnData): AnnData object containing gene expression and/or feature barcode count data.

        vdj_file (str): Path to a file containing annotated VDJ data. Abstar's ``tabular`` 
                        output format is assumed, but any delimited format can be accomodated. Required.

        tenx_annotation_csv (str): Path to CellRanger's contig annotation file (in CSV format). Note that CellRanger
                                   also outputs a JSON-formatted annotation file, which is not supported. Default is ``None``,
                                   which results in CellRanger annotations not being included. Supplying the annotation CSV
                                   is encouraged, as the UMI counts are used to identify potential background contigs.

        high_confidence (bool): If ``True``, only sequences that have been identified as "high confidence" by CellRanger are
                                merged. Default is ``True``. This option is ignored if ``tenx_annotation_csv`` is ``None``.

        vdj_delimiter (str): Delimiter used in ``vdj_file``. Default is ``'\t'``.

        vdj_id_key (str): Name of the field in ``vdj_file`` that corresponds to the sequence ID. Default is ``'seq_id'``.

        vdj_sequence_key (str): Name of the field in ``vdj_file`` that corresponds to the sequence. Default is ``'vdj_nt'``.

    
    Returns:
    --------

        Returns an ``anndata.AnnData`` object with merged VDJ data.

    '''
    # read sequences
    vdj_df = pd.read_csv(vdj_file, sep=vdj_delimiter)
    seqs = [Sequence(row.where(pd.notnull(row), None).to_dict(), id_key=vdj_id_key, seq_key=vdj_sequence_key) for index, row in vdj_df.iterrows()]
    # read 10xG annotation file
    if tenx_annotation_csv is not None:
        annot_df = pd.read_csv(tenx_annotation_csv)
        key = 'contig_id' if 'contig_id' in annot_df.columns.values else 'consensus_id'
        annots = {row[key]: row.to_dict() for index, row in annot_df.iterrows()}
        for s in seqs:
            s.tenx = annots.get(s[vdj_id_key], {})
        if high_confidence:
            seqs = [s for s in seqs if s.tenx.get('high_confidence', False)]
    # identify pairs
    pdict = {}
    for s in seqs:
        pname = s[vdj_id_key].split('_')[0]
        if pname not in pdict:
            pdict[pname] = [s, ]
        else:
            pdict[pname].append(s)
    pair_dict = {n: Pair(pdict[n],
                         name=n,
                         h_selection_func=umi_selector,
                         l_selection_func=umi_selector) 
             for n in pdict.keys()}
    # update adata
    vdjs = [pair_dict.get(o, Pair([], name=o)) for o in adata.obs_names]
    adata.obs['vdj'] = vdjs
    return adata


def umi_selector(seqs):
    sorted_seqs = sorted(seqs, key=lambda x: int(x.tenx.get('unis'), 0), reverse=True)
    return sorted_seqs[0]



def assign_bcr_lineages(adata, distance_cutoff=0.32, shared_mutation_bonus=0.65, length_penalty_multiplier=2,
                        preclustering_threshold=0.65, preclustering_field='cdr3_nt', preclustering=False,
                        annotation_format='airr', return_assignment_dict=False):
    '''
    Assigns BCR lineages using the clonify algorithm.

    Args:
    -----

        adata (anndata.AnnData): ``AnnData`` object containing annotated sequence data at ``adata.obs.bcr``. If
            data was read using ``scab.read_10x_mtx()``, BCR data should already be in the correct location.

        distance_cutoff (float): Distance threshold for lineage clustering. Default is ``0.32``.

        shared_mutation_bonus (float): Bonus applied for each shared V-gene mutation. Default is ``0.65``.

        length_penalty_multiplier (int): Multiplier for the CDR3 length penalty. Default is ``2``, resulting in
            CDR3s that differ by ``n`` amino acids being penalized ``n * 2``.

        preclustering_threshold (float): Identity threshold for pre-clustering the V/J groups prior to lineage 
            assignment. Default is ``0.65``.

        preclustering_field (str): Annotation field on which to pre-cluster sequences. Default is ``'cdr3_nt'``.

        preclustering (bool): If ``True``, V/J groups are pre-clustered, which can potentially speed up lineage assignment
            and reduce memory usage. If ``False``, each V/J group is processed in its entirety without pre-clustering. 
            Default is ``False``.

        annotation_format (str): Format of the input sequence annotations. Choices are ``['airr', 'json']``.
            Default is ``'airr'``.
        
        return_assignment_dict (bool): If ``True``, a dictionary linking sequence IDs to lineage names will be returned.
            If ``False``, the input ``anndata.AnnData`` object will be returned, with lineage annotations included.
            Default is ``False``.

    
    '''
    # select the appropriate data fields
    if annotation_format.lower() == 'airr':
        vgene_key = 'v_call'
        jgene_key = 'j_call'
        cdr3_key = 'cdr3_aa'
        muts_key = 'v_mutations'
    elif annotation_format.lower() == 'json':
        vgene_key = 'v_gene.gene'
        jgene_key = 'j_gene.gene'
        cdr3_key = 'cdr3_aa'
        muts_key = 'var_muts_nt.muts'
    else:
        error = 'ERROR: '
        error += f'annotation_format must be either "airr" or "json", but you provided {annotation_format}'
        print('\n')
        print(error)
        print('\n')
        sys.exit()
    # group sequences by V/J genes
    vj_group_dict = {}
    for p in adata.obs.bcr:
        if p.heavy is None:
            continue
        # build new Sequence objects using just the data we need
        h = p.heavy
        s = Sequence(h.sequence, id=p.name)
        s['v_call'] = nested_dict_lookup(h, vgene_key.split('.'))
        s['j_call'] = nested_dict_lookup(h, jgene_key.split('.'))
        s['cdr3'] = nested_dict_lookup(h, cdr3_key.split('.'))
        if annotation_format.lower() == 'json':
            muts = nested_dict_lookup(h, muts_key.split('.'), [])
            s['mutations'] = [f"{m['position']}:{m['was']}>{m['is']}" for m in muts]
        else:
            nested_dict_lookup(h, muts_key.split('.'), '').split('|')
            s['mutations'] = [m for m in muts if m.strip()]
        required_fields = ['v_call', 'j_call', 'cdr3', 'mutations']
        if preclustering:
            s['preclustering'] = nested_dict_lookup(h, preclustering_field.split('.'))
            required_fields.append('preclustering')
        if any([s[v] is None for v in required_fields]):
            continue
        # group sequences by VJ gene use
        vj = f"{s['v_call']}__{s['j_call']}"
        if vj not in vj_group_dict:
            vj_group_dict[vj] = []
        vj_group_dict[vj].append(s)
    # assign lineages
    assignment_dict = {}
    for vj_group in vj_group_dict.values():
        # preclustering
        if preclustering:
            seq_dict = {s.id: s for s in vj_group}
            cluster_seqs = [Sequence(s[preclustering_field], id=s.id) for s in vj_group]
            clusters = cluster(cluster_seqs, threshold=preclustering_threshold)
            groups = [[seq_dict[i] for i in c.seq_ids] for c in clusters]
        else:
            groups = [vj_group, ]
        for group in groups:
            if len(group) == 1:
                seq = group[0]
                assignment_dict[seq.id] = ''.join(random.sample(characters, 12))
                continue
            # build a distance matrix
            dist_matrix = []
            for s1, s2 in itertools.combinations(group, 2):
                d = _clonify_distance(s1, s2,
                                        shared_mutation_bonus,
                                        length_penalty_multiplier)
                dist_matrix.append(d)
            # cluster
            linkage_matrix = fc.linkage(dist_matrix, 
                                        method='average',
                                        preserve_input=False)
            cluster_list = fcluster(linkage_matrix,
                                    distance_cutoff,
                                    criterion='distance')
            # rename clusters
            cluster_ids = list(set(cluster_list))
            characters = string.ascii_letters + string.digits
            cluster_names = {c: ''.join(random.sample(characters, 12)) for c in cluster_ids}
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
    adata.obs['bcr_lineage'] = lineage_assignments
    adata.obs['bcr_lineage_size'] = lineage_sizes
    return adata

        
def _clonify_distance(s1, s2, shared_mutation_bonus, length_penalty_multiplier):
    if len(s1['cdr3']) == len(s2['cdr3']):
        dist = sum([i != j for i, j in zip(s1['cdr3'], s2['cdr3'])])
    else:
        dist = distance(s1['cdr3'], s2['cdr3'])
    length_penalty = abs(len(s1['cdr3']) - len(s2['cdr3'])) * length_penalty_multiplier
    length = min(len(s1['cdr3']), len(s2['cdr3']))
    shared_mutations = list(set(s1['mutations']) & set(s2['mutations']))
    mutation_bonus = len(shared_mutations) * shared_mutation_bonus
    score = (dist + length_penalty - mutation_bonus) / length
    return max(score, 0.001) # distance values can't be negative





