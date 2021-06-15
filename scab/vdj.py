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


import pandas as pd

from abutils.core.pair import Pair
from abutils.core.sequence import Sequence


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










