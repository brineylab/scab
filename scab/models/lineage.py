#!/usr/bin/env python
# filename: lineage.py


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


import copy
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from natsort import natsorted

import prettytable as pt

from abstar.core.germline import get_imgt_germlines
from abstar.utils.regions import (
    IMGT_REGION_START_POSITIONS_AA,
    IMGT_REGION_END_POSITIONS_AA,
)

import abutils


class Lineage:
    """
    docstring for Lineage
    """

    def __init__(self, adata, name=None):
        self.adata = adata.copy()
        self.obs = self.adata.obs
        self._name = name

    def __iter__(self):
        for bcr in self.bcrs:
            yield bcr

    @property
    def name(self):
        if self._name is None:
            if "bcr_lineage" in self.adata.obs:
                self._name = self.adata.obs.bcr_lineage.value_counts().index[0]
            elif "lineage" in self.adata.obs:
                try:
                    self._name = self.adata.obs.lineage.value_counts().index[0]
                except IndexError:
                    pass
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def size(self):
        return self.adata.shape[0]

    @property
    def bcrs(self):
        return self.adata.obs.bcr

    @property
    def pairs(self):
        return [b for b in self.bcrs if b.is_pair]

    @property
    def heavies(self):
        return [b.heavy for b in self.bcrs if b.heavy is not None]

    @property
    def lights(self):
        return [b.light for b in self.bcrs if b.light is not None]

    def summarize(
        self,
        agbcs=None,
        specificities=None,
        extra_blocks=None,
        dot_alignment=True,
        in_color=True,
        pairs_only=False,
        padding_width=2,
    ):
        if isinstance(extra_blocks, str):
            extra_blocks = [
                extra_blocks,
            ]
        return LineageSummary(
            self,
            agbcs=agbcs,
            specificities=specificities,
            extra_blocks=extra_blocks,
            dot_alignment=dot_alignment,
            in_color=in_color,
            pairs_only=pairs_only,
            padding_width=padding_width,
        )


class LineageAssignment:
    """
    docstring for LineageAssignment
    """

    def __init__(
        self, pair: abutils.Pair, assignment_dict: dict,
    ):
        self.pair = pair
        self.assignment_dict = assignment_dict
        self.all_lineages = list(assignment_dict.values())

    def __eq__(self, lineage_name) -> bool:
        return lineage_name in self.all_lineages

    def get(self, lineage_name) -> Optional[abutils.Pair]:
        """
        Gets the ``Pair`` that was assigned to a lineage. Note that if
        the original ``Pair`` object contained multiple heavy chains,
        a modified ``Pair`` is returned containing all light chains from the
        original ``Pair`` but only the heavy chain assigned to the lineage.

        Parameters
        ----------
        lineage_name : str
            Lineage name. Required

        Returns
        -------
        abutils.Pair
            A ``Pair`` that was assigned to the specified lineage. Note that if
            the original ``Pair`` object contained multiple heavy chains,
            a modified ``Pair`` is returned containing all light chains from the
            original ``Pair`` but only the heavy chain assigned to the lineage.
        """
        if lineage_name not in self.all_lineages:
            return None
        lookup = {v: k for k, v in self.assignment_dict.items()}
        identifier = lookup[lineage_name]
        i = int(identifier.split("__")[-1])
        p = copy.deepcopy(self.pair)
        p.heavy = p.heavies[i]
        p.heavies = [p.heavy]
        return p


class LineageSummary:
    """
    docstring for LineageSummary
    """

    def __init__(
        self,
        lineage,
        agbcs=None,
        specificities=None,
        extra_blocks=None,
        dot_alignment=True,
        in_color=True,
        pairs_only=False,
        padding_width=1,
    ):
        self.lineage = lineage
        self.name = lineage.name
        self.adata = lineage.adata
        self.obs = lineage.obs
        self.bcrs = lineage.bcrs
        self.agbcs = agbcs
        self.specificities = specificities
        self.extra_blocks = extra_blocks if extra_blocks is not None else []
        self.dot_alignment = dot_alignment
        self.in_color = in_color
        self.pairs_only = pairs_only
        self.padding_width = padding_width

        self.COLOR_PREFIX = {
            "yellow": "\033[93m",
            "green": "\033[92m",
            "red": "\033[91m",
            "blue": "\033[94m",
            "pink": "\033[95m",
            "black": "\033[90m",
        }

        self.REGION_COLORS = {
            "cdr1": "red",
            "cdr2": "yellow",
            "junction_v": "green",
            "junction_j": "blue",
            "junction": "black",
        }

    def __repr__(self):
        return self._assemble_summary()

    def __str__(self):
        s = self._assemble_summary()
        return s.get_string()

    def show(self, in_color=None):
        s = self._assemble_summary(in_color=in_color)
        print(s)

    def to_string(self, in_color=False):
        s = self._assemble_summary(in_color=in_color)
        return s.get_string()

    def _assemble_summary(self, in_color=None):
        in_color = in_color if in_color is not None else self.in_color
        # group identical AA sequences
        identical_dict = {}
        for bcr in self.bcrs:
            aa = ""
            if bcr.heavy is not None:
                aa += bcr.heavy["sequence_aa"]
            if bcr.light is not None:
                aa += bcr.light["sequence_aa"]
            if aa not in identical_dict:
                identical_dict[aa] = []
            identical_dict[aa].append(bcr)
        identical_groups = sorted(
            identical_dict.values(), key=lambda x: len(x), reverse=True
        )
        h_standards = [ig[0].heavy for ig in identical_groups]
        l_standards = [ig[0].light for ig in identical_groups]

        x = pt.PrettyTable()
        # frequencies block
        freq_block = self._frequencies_block(identical_groups)
        x.add_column("frequencies", ["", freq_block], align="r")

        # specificities_block
        if self.specificities is not None:
            spec_block = self._specificities_block(identical_groups)
            x.add_column("specificities", ["", spec_block], align="r")

        # AgBCs block
        if self.agbcs is not None:
            agbc_block = self._agbc_block(identical_groups)
            x.add_column("agbcs", ["", agbc_block], align="r")

        # extra blocks
        for xblock in self.extra_blocks:
            extra_block = self._extra_block(xblock, identical_groups)
            x.add_column(xblock, ["", extra_block], align="r")

        # HC and LC blocks
        hgene_block = self._germline_block(self.lineage.heavies)
        hcdr_block = self._cdr_alignment_block(h_standards, color=in_color)
        x.add_column("heavy", [hgene_block, hcdr_block], align="l")

        lgene_block = self._germline_block(self.lineage.lights)
        lcdr_block = self._cdr_alignment_block(l_standards, color=in_color)
        x.add_column("light", [lgene_block, lcdr_block], align="l")

        # style
        x.set_style(pt.DEFAULT)
        # x.set_style(pt.MARKDOWN)
        # x.set_style(pt.PLAIN_COLUMNS)
        x.header = False
        x.hrules = pt.ALL
        x.left_padding_width = self.padding_width
        x.right_padding_width = self.padding_width
        x.title = self.name
        return x

    def _frequencies_block(self, identical_groups):
        # id column
        ids = ["", "all"] + [f"{i}" for i in range(len(identical_groups))]
        # sequence count
        n_seqs = ["n", len(self.bcrs)] + [len(ig) for ig in identical_groups]
        # isotypes
        isotypes = natsorted(set([h["isotype"] for h in self.lineage.heavies]))
        isotype_dict = {}
        for i in isotypes:
            isotype_dict[i] = []
            for ig in identical_groups:
                heavies = [b.heavy for b in ig if b is not None]
                isotype_dict[i].append(sum([h["isotype"] == i for h in heavies]))
            isotype_dict[i] = [i, sum(isotype_dict[i])] + isotype_dict[i]
        # build the block
        x = pt.PrettyTable()
        x.add_column("ids", ids)
        x.add_column("n_seqs", n_seqs)
        for i in isotypes:
            x.add_column(i, isotype_dict[i])
        x.set_style(pt.PLAIN_COLUMNS)
        x.header = False
        x.left_padding_width = 0
        x.right_padding_width = self.padding_width
        x.align = "r"
        table_string = x.get_string()
        unpadded = "\n".join([l.strip() for l in table_string.split("\n")])
        return unpadded

    def _specificities_block(self, identical_groups):
        specificities_dict = {}
        for s in self.specificities:
            counts = []
            for ig in identical_groups:
                names = [b.name for b in ig]
                positives = sum(self.obs[f"is_{s}"][names])
                counts.append(positives)
            specificities_dict[s] = [s, sum(counts)] + counts
        # build the block
        x = pt.PrettyTable()
        for s in self.specificities:
            x.add_column(s, specificities_dict[s])
        x.set_style(pt.PLAIN_COLUMNS)
        x.header = False
        x.left_padding_width = 0
        x.right_padding_width = self.padding_width
        x.align = "r"
        table_string = x.get_string()
        unpadded = "\n".join([l.strip() for l in table_string.split("\n")])
        return unpadded

    def _agbc_block(self, identical_groups):
        agbc_dict = {}
        for a in self.agbcs:
            vals = []
            for ig in identical_groups:
                names = [b.name for b in ig]
                mean = np.mean(self.obs[f"{a}"][names])
                std = np.std(self.obs[f"{a}"][names])
                vals.append(f"{mean:.1f} ({std:.1f})")
            agbc_dict[a] = [a, ""] + vals
        # build the block
        x = pt.PrettyTable()
        for a in self.agbcs:
            x.add_column(a, agbc_dict[a])
        x.set_style(pt.PLAIN_COLUMNS)
        x.header = False
        x.left_padding_width = 0
        x.right_padding_width = self.padding_width
        x.align = "r"
        table_string = x.get_string()
        unpadded = "\n".join([l.strip() for l in table_string.split("\n")])
        return unpadded

    def _extra_block(self, column, identical_groups):
        categories = natsorted(self.adata.obs[column].unique())
        extra_dict = {}
        for c in categories:
            extra_dict[c] = []
            for ig in identical_groups:
                names = [b.name for b in ig]
                _adata = self.adata[names]
                extra_dict[c].append(_adata[_adata.obs[column] == c].shape[0])
            extra_dict[c] = [c, sum(extra_dict[c])] + extra_dict[c]
        # build the block
        x = pt.PrettyTable()
        for c in categories:
            x.add_column(c, extra_dict[c])
        x.set_style(pt.PLAIN_COLUMNS)
        x.header = False
        x.left_padding_width = 0
        x.right_padding_width = self.padding_width
        x.align = "r"
        table_string = x.get_string()
        unpadded = "\n".join([l.strip() for l in table_string.split("\n")])
        return unpadded

    def _germline_block(self, sequences):
        block_strings = []
        # chain
        sequences = [s for s in sequences if s is not None]
        if all([s["locus"] == "IGH" for s in sequences]):
            chain = "HEAVY CHAIN\n"
        else:
            chain = "LIGHT CHAIN\n"
        # germline genes and CDR3 length
        v_call = Counter([s["v_call"] for s in sequences]).most_common(1)[0][0]
        if any([s["d_call"] for s in sequences]):
            d_call = Counter(
                [s["d_call"] for s in sequences if s["d_call"]]
            ).most_common(1)[0][0]
            dcount = len(list(set([s["d_call"] for s in sequences])))
            if len(list(set([s["d_call"] for s in sequences]))) > 1:
                d_call = f"{d_call}{'.' * (dcount-1)}"
        else:
            d_call = None
        j_call = Counter([s["j_call"] for s in sequences]).most_common(1)[0][0]
        cdr3_length = Counter([s["cdr3_length"] for s in sequences]).most_common(1)[0][
            0
        ]
        if d_call is not None:
            block_strings.append(
                f"{chain}{v_call} | {d_call} | {j_call} | {cdr3_length}"
            )
        else:
            block_strings.append(f"{chain}{v_call} | {j_call} | {cdr3_length}")
        # identities
        ident_string = "mutation: "
        # nt_mut_counts = [int(s['v_mutation_count']) for s in sequences]
        # min_nt = np.min(nt_mut_counts)
        # max_nt = np.max(nt_mut_counts)
        # if min_nt == max_nt:
        #     ident_string += f'{min_nt}nt'
        # else:
        #     ident_string += f'{min_nt}-{max_nt}nt'
        # ident_string += ' | mutations | '
        nt_identities = [100.0 * (1 - float(s["v_identity"])) for s in sequences]
        # mean_nt = round(np.mean(nt_identities), 1)
        min_nt = round(np.min(nt_identities), 1)
        max_nt = round(np.max(nt_identities), 1)
        if f"{min_nt:.0f}" == f"{max_nt:.0f}":
            # ident_string += f"nt: {min_nt:.0f}%"
            ident_string += f"{min_nt:.0f}% nt"
        else:
            # ident_string += f"nt: {min_nt:.0f}-{max_nt:.0f}%"
            ident_string += f"{min_nt:.0f}-{max_nt:.0f}% nt"
        # aa_mut_counts = [int(s['v_mutation_count_aa']) for s in sequences]
        # min_aa = np.min(aa_mut_counts)
        # max_aa = np.max(aa_mut_counts)
        # if min_aa == max_aa:
        #     ident_string += f'{min_aa}aa'
        # else:
        #     ident_string += f'{min_aa}-{max_aa}aa'
        aa_identities = [100.0 * (1 - float(s["v_identity_aa"])) for s in sequences]
        mean_aa = round(np.mean(aa_identities), 1)
        min_aa = round(np.min(aa_identities), 1)
        max_aa = round(np.max(aa_identities), 1)
        # ident_string += ' | muts | '
        ident_string += " | "
        if f"{min_aa:.0f}" == f"{max_aa:.0f}":
            # ident_string += f"aa: {min_aa:.0f}%"
            ident_string += f"{min_aa:.0f}% aa"
        else:
            # ident_string += f"aa: {min_aa:.0f}-{max_aa:.0f}%"
            ident_string += f"{min_aa:.0f}-{max_aa:.0f}% aa"
        block_strings.append(ident_string)
        # multiple seqeunces for a single chain
        if chain.startswith("H"):
            multiples = sum([len(b.heavies) > 1 for b in self.lineage.bcrs])
            block_strings.append(f"multiple heavies: {multiples}")
        else:
            multiples = sum([len(b.lights) > 1 for b in self.lineage.bcrs])
            block_strings.append(f"multiple lights: {multiples}")

        return "\n".join(block_strings)

    def _cdr_alignment_block(self, sequences, dot_alignment=True, color=True):
        aln_data = []
        # get a representative sequence for parsing junction germline info
        notnone_sequences = [s for s in sequences if s is not None]
        ref = notnone_sequences[0]
        # get region positions for germline
        cdr1_start = IMGT_REGION_START_POSITIONS_AA["CDR1"] - 1
        cdr1_end = IMGT_REGION_END_POSITIONS_AA["CDR1"]
        cdr2_start = IMGT_REGION_START_POSITIONS_AA["CDR2"] - 1
        cdr2_end = IMGT_REGION_END_POSITIONS_AA["CDR2"]
        # junction_start = IMGT_REGION_END_POSITIONS_AA['FR3'] - 1
        # junction_end = IMGT_REGION_START_POSITIONS_AA['FR4']
        # get data for germline line
        d = {"name": "germ", "order": 0}
        dbname = Counter(
            [s["germline_database"] for s in notnone_sequences]
        ).most_common(1)[0][0]
        v_gene = Counter([s["v_call"] for s in notnone_sequences]).most_common(1)[0][0]
        j_gene = Counter([s["j_call"] for s in notnone_sequences]).most_common(1)[0][0]
        v_germ = get_imgt_germlines(dbname, "V", gene=v_gene)
        # j_germ = get_imgt_germlines(dbname, 'J', gene=j_gene)
        junction_v = ref["junction_germ_v_aa"]
        junction_j = ref["junction_germ_j_aa"]
        junction_x = "X" * (len(ref["junction_aa"]) - len(junction_v) - len(junction_j))
        d["cdr1"] = v_germ.gapped_aa_sequence[cdr1_start:cdr1_end].replace(".", "")
        d["cdr2"] = v_germ.gapped_aa_sequence[cdr2_start:cdr2_end].replace(".", "")
        d["junction"] = junction_v + junction_x + junction_j
        aln_data.append(d)
        # get data for each sequence line
        for i, seq in enumerate(sequences, 1):
            d = {"name": i, "order": i}
            for region in ["cdr1", "cdr2", "junction"]:
                if seq is not None:
                    d[region] = seq[f"{region}_aa"]
                else:
                    d[region] = "X"
            aln_data.append(d)
        # make a dataframe
        df = pd.DataFrame(aln_data).sort_values(by="order")
        # do the alignments
        names = df["name"]
        for region in ["cdr1", "cdr2", "junction"]:
            aln_seqs = [abutils.Sequence(s, id=n) for s, n in zip(df[region], names)]
            aln = abutils.tl.muscle(aln_seqs)
            # aln_dict = {rec.id: str(rec.seq) for rec in aln}
            aln_dict = {}
            for rec in aln:
                if not str(rec.seq).replace("-", "").replace("X", ""):
                    aln_dict[rec.id] = " " * len(str(rec.seq))
                else:
                    aln_dict[rec.id] = str(rec.seq)
            df[f"{region}_aligned"] = [aln_dict[str(n)] for n in names]
        # make a dot alignment
        for region in ["cdr1", "cdr2"]:
            dot_alns = []
            dot_ref = df[f"{region}_aligned"][0]
            dot_alns.append(dot_ref)
            for name, reg in zip(df["name"], df[f"{region}_aligned"]):
                if name == "germ":
                    continue
                dot_aln = ""
                for r1, r2 in zip(dot_ref, reg):
                    if r1 == r2 == "-":
                        dot_aln += "-"
                    elif r1 == r2:
                        dot_aln += "."
                    else:
                        dot_aln += r2
                # dot_aln = ''.join(['.' if r1 == r2 else r2 for r1, r2 in zip(dot_ref, reg)])
                dot_alns.append(dot_aln)
            df[f"{region}_dot_aligned"] = dot_alns
        df["junction_dot_aligned"] = df["junction_aligned"]
        # make the block
        block_data = []
        header = []
        sep = " " * self.padding_width
        regions = ["cdr1", "cdr2", "junction"]
        is_dot = "_dot" if dot_alignment else ""
        for _, row in df.iterrows():
            d = []
            if row["name"] == "germ":
                for region in regions:
                    c = self.COLOR_PREFIX[self.REGION_COLORS[region]] if color else ""
                    r = row[f"{region}{is_dot}_aligned"]
                    if region == "junction":
                        vc = (
                            self.COLOR_PREFIX[self.REGION_COLORS["junction_v"]]
                            if color
                            else ""
                        )
                        jc = (
                            self.COLOR_PREFIX[self.REGION_COLORS["junction_j"]]
                            if color
                            else ""
                        )
                        r = r.replace("X", ".")
                        v = r.split(".")[0]
                        j = r.split(".")[-1]
                        n = "." * (len(r) - len(v) - len(j))
                        r = f"{vc}{v}{c}{n}{jc}{j}{c}"
                        # build the junction header
                        vheader = "v"
                        vheader += "-" * (len(v) - 1)
                        # vheader += '>'
                        vheader = vheader[: len(v)]
                        # jheader = '<'
                        jheader = "-" * (len(j) - 1)
                        jheader += "j"
                        jheader = jheader[-len(j) :]
                        header.append(f"{vheader}{' ' * len(n)}{jheader}")
                    else:
                        rname = region if len(region) <= len(r) else region[-1:]
                        h_dashes = max((len(r) - len(rname)) / 2, 0)
                        prefix_dashes = "-" * int(np.floor(h_dashes))
                        suffix_dashes = "-" * int(np.ceil(h_dashes))
                        header.append(f"{prefix_dashes}{rname}{suffix_dashes}")
                        r = f"{c}{r}"
                    d.append(r)
            else:
                d.extend([row[f"{region}{is_dot}_aligned"] for region in regions])
            block_data.append([str(_d) for _d in d])
        # assemble the alignment table
        x = pt.PrettyTable()
        for bd in [header] + block_data:
            x.add_row(bd)
        x.set_style(pt.PLAIN_COLUMNS)
        x.header = False
        x.left_padding_width = self.padding_width
        x.right_padding_width = self.padding_width
        table_string = x.get_string()
        unpadded = "\n".join([l.strip() for l in table_string.split("\n")])
        return unpadded
