#!/usr/bin/env python
# filename: batch_cellranger.py


#
# Copyright (c) 2022 Bryan Briney
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


from argparse import ArgumentParser
import csv
from datetime import datetime
import humanize
from natsort import natsorted
import os
import pathlib
import re
import shutil
import subprocess as sp
import sys
import time
from unicodedata import name
import urllib

from typing import Any, Callable, Collection, Dict, Mapping, Optional, Sequence, Union

import yaml

from natsort import natsorted

from sample_sheet import SampleSheet

from abutils.utils import log
from abutils.utils.pipeline import list_files, make_dir


from ..version import __version__


def parse_arguments(print_help=False):
    parser = ArgumentParser(
        prog="batch_cellranger",
        description="Batch CellRanger processing of one or more 10x Genomics samples.",
    )
    parser.add_argument(
        "-p",
        "--project-directory",
        dest="project_dir",
        required=True,
        help="The project directory, where run data will be downloaded \
                        and output files will be written. Required.",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        required=True,
        help="The config file, in YML format. Required.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="If set, logs are much more detailed, including the stdout and stderr \
                        from all commands. Default is ``False``, which produces more consise logs.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    if print_help:
        parser.print_help()
    else:
        args = parser.parse_args()
        args.project_dir = os.path.abspath(args.project_dir)
        return args


class Args:
    """
    docstring for Args()
    """

    def __init__(
        self,
        project_dir: Union[str, pathlib.Path, None] = None,
        config_file: Union[str, pathlib.Path, None] = None,
        debug: bool = False,
    ):
        super(Args, self).__init__()
        self.project_dir = os.path.abspath(project_dir)
        self.config_file = os.path.abspath(config_file)
        self.debug = debug


class Config:
    """
    Config provides the following attributes:

      - config_file: path to the configuration file, in YAML format.
      - runs: a list of ``Run`` objects
      - samples: a list of `Sample` objects.
      - [gex|vdj|feature]_reference: dictionary mapping sample names to
            GEX, VDJ or Feature references. Each reference tuype must include
            a ``default`` reference, which will be used for all samples not
            specifically named in the dictionary.
      - uiport: port for the cellranger UI. Default is 72647.
      - cellranger: path to the cellranger binary. Default is ``"cellranger"``, which
            assumes that the cellranger binary is on your $PATH.

    """

    def __init__(self, config_file: Union[str, pathlib.Path]):
        self.config_file = os.path.abspath(config_file)
        self.gex_reference = None
        self.vdj_reference = None
        self.feature_reference = None
        self.uiport = None
        self.cellranger = None
        self._runs = None
        self._samples = None
        self._parse_config_file()

    def __repr__(self) -> str:
        rlist = ["BATCH CELLRANGER CONFIGURATION"]
        rlist.append("------------------------------")
        rlist.append("config file: {}".format(self.config_file))
        rlist.append("VDJ reference:")
        rlist.append("  - default: {}".format(self.reference["default"]))
        for k, v in self.reference.items():
            if k != "default":
                rlist.append("  - {}: {}".format(k, v))
        rlist.append("transcriptome:")
        rlist.append("  - default: {}".format(self.transcriptome["default"]))
        for k, v in self.transcriptome.items():
            if k != "default":
                rlist.append("  - {}: {}".format(k, v))
        rlist.append("feature reference:")
        rlist.append("  - default: {}".format(self.feature_reference["default"]))
        for k, v in self.feature_reference.items():
            if k != "default":
                rlist.append("  - {}: {}".format(k, v))
        rlist.append("UI port: {}".format(self.uiport))
        rlist.append("cellranger binary: {}".format(self.cellranger))
        rlist.append("runs: {}".format([r.name for r in self.runs]))
        rlist.append("samples: {}".format([s.name for s in self.samples]))
        return "\n".join(rlist)

    @property
    def runs(self) -> Sequence:
        if self._runs is None:
            return []
        return self._runs

    @runs.setter
    def runs(self, runs: Sequence):
        self._runs = runs

    @property
    def samples(self) -> Sequence:
        if self._samples is None:
            return []
        return self._samples

    @samples.setter
    def samples(self, samples: Sequence):
        self._samples = samples

    @staticmethod
    def get_ref(ref_dict: Mapping, key: str) -> Union[str, pathlib.Path, None]:
        """
        Looks for the reference that corresponds to `key`. If it doesn't exist,
        looks for the `'default'` reference. If that doesn't exist either,
        return `None`.
        """
        if key in ref_dict:
            return ref_dict[key]
        if "default" in ref_dict:
            return ref_dict["default"]
        return None

    def get_multi_cli_options(self, sample_name: str) -> str:
        if sample_name in self.cli_options["multi"]:
            return self.cli_options["multi"][sample_name]
        return self.cli_options["multi"].get("default", "")

    def get_mkfastq_cli_options(self, run_name: str) -> str:
        if run_name in self.cli_options["mkfastq"]:
            return self.cli_options["mkfastq"][run_name]
        return self.cli_options["mkfastq"].get("default", "")

    def _parse_config_file(self):
        """
        Parses the user-provided YAML configuration file.
        """
        with open(self.config_file) as f:
            config = yaml.load(f, Loader=yaml.BaseLoader)
        # runs
        if "sequencing_runs" in config:
            self.runs = [
                Run(name, cfg) for name, cfg in config["sequencing_runs"].items()
            ]
        # references
        self.gex_reference = config.get("gex_reference", {})
        self.vdj_reference = config.get("vdj_reference", {})
        self.feature_reference = config.get("feature_reference", {})
        # samples
        sample_dict = config.get("samples", {})
        self.samples = [
            Sample(
                name,
                library_dict,
                gex_reference=Config.get_ref(self.gex_reference, name),
                vdj_reference=Config.get_ref(self.vdj_reference, name),
                feature_reference=Config.get_ref(self.feature_reference, name),
            )
            for name, library_dict in sample_dict.items()
        ]
        # general config options
        self.uiport = config.get("uiport", 72647)
        self.cellranger = config.get("cellranger", "cellranger")

        # cli options
        self.cli_options = config.get("cli_options", {})
        if "mkfastq" not in self.cli_options:
            self.cli_options["mkfastq"] = {"default": ""}
        if "multi" not in self.cli_options:
            self.cli_options["multi"] = {"default": ""}

            # for name, lib_dict in config['samples'].itmes()
        # collect samples from runs
        # libraries= []
        # for run in self.runs:
        #     if run.libraries is not None:
        #         libraries += run.libraries
        # self.libraries = list(set(libraries))
        # # # assign runs to each sample:
        # # for run in self.runs:
        # #     for s in samples:
        # #         if s.name in [s.name for s in run.samples]:
        # #             s.add_run(run.name)
        # # parse ops
        # self.ops = {}
        # self.ops['vdj'] = config.get('vdj', [])
        # self.ops['count'] = config.get('count', {})
        # self.ops['aggr'] = config.get('aggr', {})
        # # assign ops to each sample
        # for op, samples in self.ops.items():
        #     if op in ['count']:
        #         samples = [k for subject_dict in samples for k in subject_dict.keys()]
        #     for s in self.samples:
        #         if s.name in samples:
        #             s.add_op(op)

        # assign references/transcriptomes to each sample:
        # for s in self.samples:
        #     s.gex_reference = config['gex_reference'].get(s.name, config['gex_reference']['default'])
        #     s.vdj_reference = config['vdj_reference'].get(s.name, config['vdj_reference']['default'])
        #     s.feature_reference = config['feature_reference'].get(s.name, config['feature_reference']['default'])
        # # general config options
        # self.uiport = config.get('uiport', 72647)
        # self.cellranger = config.get('cellranger', 'cellranger')


class Run:
    """
    Object for aggregation of sequencing run information throughput the 10x processing
    """

    def __init__(
        self, name: str, config: dict,
    ):
        self.name = name
        self.config = config
        self.url = config.get("url", None)
        self.path = os.path.abspath(config["path"]) if "path" in config else None
        self.samplesheet = (
            os.path.abspath(config["samplesheet"]) if "samplesheet" in config else None
        )
        self.simple_csv = (
            os.path.abspath(config["simple_csv"]) if "simple_csv" in config else None
        )
        # self.is_compressed = config.get("is_compressed", True)
        # self.copy_to_project = config.get("copy_to_project", False)
        self.get_start = None
        self.get_finish = None
        self.mkfastq_start = None
        self.mkfastq_finish = None
        self._successful_get = False
        self._fastq_path = None
        self._libraries = None

    def __repr__(self):
        rstring = "RUN: {}".format(self.name)
        rlist = [rstring]
        rlist.append("-" * len(rstring))
        if self.url is not None:
            rlist.append("url: {}".format(self.url))
        if self.path is not None:
            rlist.append("path: {}".format(self.path))
        rlist.append("compressed: {}".format(self.is_compressed))
        if self.samplesheet is not None:
            rlist.append("samplesheet: {}".format(self.samplesheet))
        if self.simple_csv is not None:
            rlist.append("simple csv: {}".format(self.simple_csv))
        rlist.append("fastq path: {}".format(self.fastq_path))
        rlist.append("samples: {}".format(self.samples))
        return "\n".join(rlist)

    @property
    def sample_names(self) -> Sequence:
        if self.samples is not None:
            return [s.name for s in self.samples]
        return []

    @property
    def fastq_path(self) -> Union[str, pathlib.Path, None]:
        return self._fastq_path

    @fastq_path.setter
    def fastq_path(self, path: Union[str, pathlib.Path]):
        self._fastq_path = path

    @property
    def libraries(self) -> Sequence:
        if self._libraries is None:
            self._libraries = self._parse_libraries()
        return natsorted(self._libraries)

    @libraries.setter
    def libraries(self, libraries):
        self._libraries = libraries

    @property
    def successful_get(self) -> bool:
        return self._successful_get

    @successful_get.setter
    def successful_get(self, successful_get):
        self._successful_get = successful_get

    @property
    def successful_mkfastq(self) -> bool:
        if self.successful_mkfastq_libraries:
            return True
        return False

    @property
    def mkfastq_cli_options(self) -> str:
        return self.config.get_mkfastq_cli_options(self.name)

    @property
    def successful_mkfastq_libraries(self) -> Sequence:
        if self.fastq_path is None:
            return []
        lib_names = []
        for item in os.listdir(self.fastq_path):
            if item.startswith("Undetermined"):
                continue
            item_path = os.path.join(self.fastq_path, item)
            # some versions of CellRanger put fastqs in library-specific subfolders
            if os.path.isdir(item_path):
                if any([f.endswith(".fastq.gz") for f in os.listdir(item_path)]):
                    lib_names.append(item)
            # others just dump them all in the flowcell folder
            elif os.path.isfile(item_path):
                if item.endswith("fastq.gz"):
                    lib = "_".join(item.split("_")[:-4])
                    lib_names.append(lib)
        return natsorted(set(lib_names))

    @property
    def copy_to_project(self) -> bool:
        if "copy_to_project" in self.config:
            ctp = self.config["copy_to_project"]
            if isinstance(ctp, str):
                if ctp.strip().lower() == "true":
                    return True
                else:
                    return False
            else:
                return ctp
        else:
            return True

    @property
    def is_compressed(self) -> bool:
        if self.path is not None:
            if os.path.isdir(self.path):
                return False
        elif "is_compressed" in self.config:
            ic = self.config["is_compressed"]
            if isinstance(ic, str):
                if ic.lower() == "true":
                    return True
                else:
                    return False
            else:
                return ic
        else:
            return True

    def print_splash(self):
        l = len(self.name)
        logger.info("")
        # logger.info('-' * (l + 4))
        logger.info("  " + self.name)
        logger.info("-" * (l + 4))

    def print_get_completion(self):
        if self.successful_get:
            delta = self.get_finish - self.get_start
            logger.info(
                f"successfully retrieved run data in {humanize.precisedelta(delta)}"
            )
        else:
            logger.info("")
            logger.info("run data was not found in the expected location")
            logger.info(f"  --> {self.path}")
            logger.info("check the logs to see if any errors occured")
        logger.info("")

    def print_mkfastq_completion(self):
        if self.successful_mkfastq:
            delta = self.mkfastq_finish - self.mkfastq_start
            logger.info(
                f"mkfastq completed successfully in {humanize.precisedelta(delta)}"
            )
            logger.info("")
            logger.info("FASTQ files were created for the following libraries:")
            for l in self.successful_mkfastq_libraries:
                logger.info(f"  - {l}")
        else:
            logger.info("")
            logger.info(
                f"mkfastq may have failed, because no FASTQ output files were found at the expected location"
            )
            logger.info(f"  --> {self.fastq_path}")
            logger.info("check the logs to see if any errors occured")
        logger.info("")

    def get(
        self,
        raw_dir: Union[str, pathlib.Path],
        log_dir: Union[str, pathlib.Path, None] = None,
        debug: bool = False,
    ):
        """
        docstring for get()
        """
        self.get_start = datetime.now()
        destination = os.path.join(os.path.abspath(raw_dir), self.name)
        if all([self.path is not None, self.copy_to_project, not self.is_compressed]):
            self.path = self._copy(destination, log_dir=log_dir, debug=debug)
        if self.url is not None:
            self.path = self._download(
                self.url, destination, log_dir=log_dir, debug=debug
            )
        if self.is_compressed:
            self.path = self._decompress(
                self.path, destination, log_dir=log_dir, debug=debug
            )
        self.successful_get = self._verify_get_success()
        self.get_finish = datetime.now()

    def mkfastq(
        self,
        fastq_dir: Union[str, pathlib.Path],
        cellranger: str = "cellranger",
        uiport: Optional[int] = None,
        log_dir: Optional[str] = None,
        cli_options: Optional[str] = None,
        debug: bool = False,
    ) -> str:
        """
        docstring for mkfastq()
        """
        self.mkfastq_start = datetime.now()
        logger.info("running cellranger mkfastq....")
        mkfastq_cmd = f"cd '{fastq_dir}' && {cellranger} mkfastq"
        mkfastq_cmd += f" --id={self.name}"
        mkfastq_cmd += f" --run='{self.path}'"
        if self.samplesheet is not None:
            self._copy_samplesheet(fastq_dir)
            mkfastq_cmd += f" --samplesheet='{self.samplesheet}'"
        else:
            self._copy_simple_csv(fastq_dir)
            mkfastq_cmd += f" --csv='{self.simple_csv}'"
        if uiport is not None:
            mkfastq_cmd += f" --uiport={uiport}"
        if cli_options is not None:
            mkfastq_cmd += f" {cli_options}"
        p = sp.Popen(mkfastq_cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True, text=True)
        time.sleep(5)
        uifile = os.path.join(fastq_dir, f"{self.name}/_uiport")
        with open(uifile) as f:
            uistring = f.read().strip()
        external_ip = (
            urllib.request.urlopen("https://api.ipify.org").read().decode("utf8")
        )
        uistring = f"http://{external_ip}:{uistring.split(':')[-1]}"
        logger.info(f"  --> cellranger UI: {uistring}")
        o, e = p.communicate()
        if debug:
            logger.info("\nMKFASTQ")
            logger.info(mkfastq_cmd)
            logger.info(o)
            logger.info(e)
            logger.info("\n")
        if log_dir is not None:
            log_subdir = os.path.join(log_dir, "cellranger_mkfastq")
            make_dir(log_subdir)
            write_log(self.name, log_subdir, stdout=o, stderr=e)

        ## TODO:NEED TO DOUBLE-CHECK WHAT THE FASTQ PATH ACTUALLY IS
        ## is it just --output-dir? or do they go into an --id subfolder?

        ## Turns out, NO. It's not just /outs/fastq_path. We actually need to
        ## get the flowcell directory, since otherwise cellranger just finds
        ## the Undetermined FASTQs, which are in the /outs/fastq_path directory
        ##
        ## see here: https://github.com/10XGenomics/supernova/blob/master/tenkit/lib/python/tenkit/illumina_instrument.py#L12-L45
        ## for some regex ideas of how to spot the flowcell ID.
        fastq_path = os.path.join(fastq_dir, f"{self.name}/outs/fastq_path")
        flowcell_pattern = re.compile(
            "[[CHA][A-Z,0-9]{8}$|[ABDG][A-Z,0-9]{4}$]"
        )  # first part of the pattern matches all flowcells except MiSeq, second part matches MiSeq
        for root, subdirs, files in os.walk(fastq_path):
            for subdir in subdirs:
                if flowcell_pattern.match(subdir) is not None:
                    self.fastq_path = os.path.join(root, subdir)
                    break
            if self.fastq_path is not None:
                break
        self.mkfastq_finish = datetime.now()
        return self.fastq_path

    def _copy(self, destination, log_dir=None, debug=False):
        shutil.copytree(self.path, destination)
        return destination

    def _copy_samplesheet(
        self, d: Union[str, pathlib.Path],
    ):
        """
        Copies the run's samplesheet to a different directory.

        Parameters
        ----------
        d
            Directory into which the samlesheet will be copied.
        """
        dest = os.path.join(d, f"{self.name}_samplesheet.csv")
        shutil.copy(self.samplesheet, dest)

    def _copy_simple_csv(
        self, d: Union[str, pathlib.Path],
    ):
        """
        Copies the run's simple CSV to a different directory.

        Parameters
        ----------
        d
            Directory into which the simple CSV will be copied.
        """
        dest = os.path.join(d, f"{self.name}_simple.csv")
        shutil.copy(self.simple_csv, dest)

    def _download(
        self,
        url: str,
        destination: Union[str, pathlib.Path],
        log_dir: Union[str, pathlib.Path, None] = None,
        debug: bool = False,
    ) -> str:
        """
        docstring for _download()
        """
        logger.info("downloading run data....")
        destination = os.path.abspath(destination)
        make_dir(destination)
        wget_cmd = "wget -P '{}' {}".format(destination, url)
        p = sp.Popen(wget_cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True, text=True)
        o, e = p.communicate()
        if debug:
            logger.info("\nDOWNLOAD")
            logger.info(wget_cmd)
            logger.info(o)
            logger.info(e)
            logger.info("\n")
        if log_dir is not None:
            log_subdir = os.path.join(log_dir, "download")
            make_dir(log_subdir)
            write_log(self.name, log_subdir, stdout=o, stderr=e)
        fname = os.path.basename(url)
        return os.path.join(destination, fname)

    def _decompress(
        self,
        source: Union[str, pathlib.Path],
        destination: Union[str, pathlib.Path],
        log_dir: Union[str, pathlib.Path, None] = None,
        debug: bool = False,
    ) -> str:
        """
        docstring for _decompress()
        """
        source = os.path.abspath(source)
        destination = os.path.abspath(destination)
        if os.path.isdir(source):
            logger.info(
                "the supplied run data path is a directory, not a compressed file. "
            )
            if self.copy_to_project:
                logger.info("copying to the project directory without decompressing...")
                shutil.copytree(source, destination)
        else:
            logger.info("decompressing run data....")
            make_dir(destination)
            if source.endswith((".tar.gz", ".tgz")):
                cmd = f"tar xzvf '{source}' -C '{destination}'"
            elif source.endswith(".tar"):
                cmd = f"tar xvf '{source}' -C '{destination}'"
            elif source.endswith(".zip"):
                cmd = f"unzip {source} -d {destination}"
            else:
                err = f"\nERROR: input file {source} has an unsupported compression type\n"
                err += "only files with .tar, .tar.gz, .tgz or .zip extensions are supported\n"
                print(err)
                sys.exit()
            p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True, text=True)
            o, e = p.communicate()
            if debug:
                logger.info("\nDECOMPRESS")
                logger.info(cmd)
                logger.info(o)
                logger.info(e)
                logger.info("\n")
            if log_dir is not None:
                log_subdir = os.path.join(log_dir, "decompress")
                make_dir(log_subdir)
                write_log(self.name, log_subdir, stdout=o, stderr=e)
        run_dir = destination
        for root, subdirs, files in os.walk(destination):
            if "RTAComplete.txt" in files:
                run_dir = os.path.join(destination, root)
                break
        return run_dir

    def _verify_get_success(self):
        if os.path.isdir(self.path):
            if "RTAComplete.txt" in os.listdir(self.path):
                return True
        return False

    def _parse_libraries(self):
        """
        docstring for _parse_libraries()
        """
        if self.samplesheet is not None:
            return self._parse_samplesheet()
        if self.simple_csv is not None:
            return self._parse_simple_csv()

    def _parse_samplesheet(self):
        """
        docstring for _parse_samplesheet()
        """
        ss = SampleSheet(self.samplesheet)
        libraries = [s.Sample_Name for s in ss.samples]
        return libraries

    def _parse_simple_csv(self):
        """
        docstring for _parse_simple_csv()
        """
        libraries = []
        with open(self.simple_csv) as csvfile:
            reader = csv.DictReader(csvfile)
            for r in reader:
                rlower = {k.lower(): v for k, v in r.items()}
                if sample := rlower["sample"].strip():
                    libraries.append(sample)
        return libraries


class Sample:
    """
    Object for aggregating information about a single sample
    """

    def __init__(
        self,
        name: str,
        library_dict: Dict,
        gex_reference: Union[str, pathlib.Path, None] = None,
        vdj_reference: Union[str, pathlib.Path, None] = None,
        feature_reference: Union[str, pathlib.Path, None] = None,
    ):
        self.name = name
        self.gex_reference = gex_reference
        self.vdj_reference = vdj_reference
        self.feature_reference = feature_reference
        self._library_dict = library_dict
        self._libraries = None
        self._libraries_by_type = None

    def __lt__(self, other):
        return all([self.name < other.name])

    def __hash__(self):
        return hash(self.name)

    @property
    def libraries(self) -> Sequence:
        if self._libraries is None:
            self._libraries = []
            for lib_type, name in self._library_dict.items():
                self._libraries.append(Library(name, lib_type))
        return self._libraries

    @property
    def libraries_by_type(self) -> Dict:
        if self._libraries_by_type is None:
            self._libraries_by_type = {}
            for l in self.libraries:
                if l.type not in self._libraries_by_type:
                    self._libraries_by_type[l.type] = []
                self._libraries_by_type[l.type].append(l)
        return self._libraries_by_type

    def print_splash(self) -> None:
        l = len(self.name)
        logger.info("")
        logger.info("  " + self.name)
        logger.info("-" * (l + 4))
        logger.info("libraries:")
        for l in self.libraries:
            logger.info(f"  - {l.name}")
        logger.info("references:")
        if self.gex_reference is not None:
            logger.info(f"  - gex: {self.gex_reference}")
        if self.vdj_reference is not None:
            logger.info(f"  - vdj: {self.vdj_reference}")
        if self.feature_reference is not None:
            logger.info(f"  - features: {self.feature_reference}")
        logger.info("")

    def make_config_csv(self, csv_dir: Union[str, pathlib.Path]) -> str:
        """
        Makes a config CSV for cellranger multi. CSV will be named
        ``{sample.name}_config.csv`` and deposited into `csv_dir`.
        """
        if not os.path.isdir(csv_dir):
            make_dir(csv_dir)
        csv_path = os.path.join(csv_dir, f"{self.name}_config.csv")
        with open(csv_path, "w") as f:
            f.write(self._build_config_csv())
        return csv_path

    def _build_config_csv(self) -> str:
        """
        Builds a config CSV string, which can be written to file
        and used with cellranger multi.
        """
        config = ""
        if self.gex_reference is not None:
            config += "[gene-expression]\n"
            config += f"reference,{self.gex_reference}\n\n"
        if self.vdj_reference is not None:
            config += "[vdj]\n"
            config += f"reference,{self.vdj_reference}\n\n"
        if self.feature_reference is not None:
            config += "[feature]\n"
            config += f"reference,{self.feature_reference}\n\n"
        config += "[libraries]\n"
        config += "fastq_id,fastqs,feature_types\n"
        for library in self.libraries:
            for fastq in library.fastq_paths:
                config += f"{library.name},{fastq},{library.type}\n"
        return config


class Library:
    """
    Object for aggregating information about a single library
    """

    def __init__(self, name: str, library_type: str):
        self.name = name
        self.type = library_type
        self._fastq_paths = None

    @property
    def fastq_paths(self) -> Sequence:
        if self._fastq_paths is None:
            self._fastq_paths = []
        return self._fastq_paths

    def add_fastq_path(self, fastq_path: Union[str, pathlib.Path]):
        self.fastq_paths.append(os.path.abspath(fastq_path))


# ==================
#    OPERATIONS
# ==================


def cellranger_multi(
    sample: Sample,
    output_dir: Union[str, pathlib.Path],
    cellranger: Optional[str] = "cellranger",
    uiport: Optional[int] = None,
    log_dir: Union[str, pathlib.Path, None] = None,
    cli_options: Optional[str] = None,
    debug: bool = False,
):
    """
    docstring for cellranger_multi()
    """
    start = datetime.now()
    logger.info(f"making config CSV...")
    # config_csv = os.path.join(output_dir, f"{sample.name}_config.csv")
    config_csv = sample.make_config_csv(output_dir)
    multi_cmd = f"cd '{output_dir}'"
    multi_cmd += f" && {cellranger} multi --id {sample.name} --csv {config_csv}"
    if uiport is not None:
        multi_cmd += f" --uiport {uiport}"
    if cli_options is not None:
        multi_cmd += f" {cli_options}"
    logger.info(f"running cellranger multi..")
    p = sp.Popen(multi_cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True, text=True)
    time.sleep(3)
    uifile = os.path.join(output_dir, f"{sample.name}/_uiport")
    with open(uifile) as f:
        uistring = f.read().strip()
    external_ip = urllib.request.urlopen("https://api.ipify.org").read().decode("utf8")
    uistring = f"http://{external_ip}:{uistring.split(':')[-1]}"
    logger.info(f"  --> cellranger UI: {uistring}")
    o, e = p.communicate()
    if debug:
        logger.info("\nCELLRANGER MULTI")
        logger.info(multi_cmd)
        logger.info(o)
        logger.info(e)
        logger.info("\n")
    if log_dir is not None:
        log_subdir = os.path.join(log_dir, "cellranger_multi")
        make_dir(log_subdir)
        write_log(sample.name, log_subdir, stdout=o, stderr=e)
    # check for successful completion
    sample_output_dir = os.path.join(output_dir, sample.name)
    if "outs" not in os.listdir(sample_output_dir):
        logger.info("")
        logger.info(
            f'cellranger multi may have failed, because the "outs" directory was not found at the expected location'
        )
        logger.info(f"  --> {sample_output_dir}")
        logger.info("check the logs to see if any errors occured")
    else:
        delta = datetime.now() - start
        logger.info(f"cellranger multi completed in {humanize.precisedelta(delta)}")
    logger.info("")


# op_lookup = {'gex': 'Gene Expression',
#              'features': 'Antibody Capture',
#              'crispr': 'CRISPR Guide Capture',
#              'bcr': 'VDJ-B',
#              'tcr': 'VDJ-T',
#              'tcr-gd': 'VDJ-T-GD'}


# def cellranger_vdj(sample, vdj_dir, cellranger='cellranger', uiport=None, log_dir=None, debug=False):
#     '''
#     docstring
#     '''
#     vdj_dir = os.path.abspath(vdj_dir)
#     vdj_cmd = "cd '{}'".format(vdj_dir)
#     vdj_cmd += " && {} vdj --id {} --sample {} --reference '{}'".format(cellranger,
#                                                                         sample.name,
#                                                                         sample.id,
#                                                                         sample.reference)
#     for fastq in sample.fastqs:
#         vdj_cmd += " --fastq '{}'".format(fastq)
#     if uiport is not None:
#         vdj_cmd += ' --uiport {}'.format(uiport)
#     p = sp.Popen(vdj_cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
#     time.sleep(3)
#     uifile = os.path.join(vdj_dir, '{}/_uiport'.format(self.name))
#     with open(uifile) as f:
#         uistring = f.read().strip()
#     external_ip = urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
#     uistring = 'http://' + external_ip + ':' + uistring.split(':')[-1]
#     logger.info('CellRanger UI is at {}'.format(uistring))
#     o, e = p.communicate()
#     if debug:
#         logger.info('\nCELLRANGER VDJ')
#         logger.info(o)
#         logger.info(e)
#     if log_dir is not None:
#         log_subdir = os.path.join(log_dir, 'vdj')
#         make_dir(log_subdir)
#         write_log(sample.name, log_subdir, stdout=o, stderr=e)
#     return os.path.join(vdj_dir, sample.name)


# def cellranger_count(group, samples, feature_ref, count_dir,
#                      cellranger='cellranger', uiport=None, log_dir=None, debug=False):
#     '''
#     docstring
#     '''
#     count_dir = os.path.abspath(count_dir)
#     lib_csv = _make_feature_library_csv(samples, group, count_dir)
#     count_cmd = "cd '{}'".format(count_dir)
#     count_cmd += " && {} count --id {} --libraries {} --feature_ref {} --transcriptome '{}'".format(cellranger,
#                                                                                                     lib_csv,
#                                                                                                     feature_ref,
#                                                                                                     sample.id,
#                                                                                                     sample.transcriptome)
#     for fastq in sample.fastqs:
#         count_cmd += " --fastqs '{}'".format(fastq)
#     if uiport is not None:
#         count_cmd += " --uiport '{}'".format(uiport)
#     p = sp.Popen(count_cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
#     time.sleep(3)
#     uifile = os.path.join(count_dir, '{}/_uiport'.format(self.name))
#     with open(uifile) as f:
#         uistring = f.read().strip()
#     external_ip = urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
#     uistring = 'http://' + external_ip + ':' + uistring.split(':')[-1]
#     logger.info('CellRanger UI is at {}'.format(uistring))
#     o, e = p.communicate()
#     if debug:
#         logger.info('\nCELLRANGER COUNT')
#         logger.info(o)
#         logger.info(e)
#     if log_dir is not None:
#         log_subdir = os.path.join(log_dir, 'count')
#         make_dir(log_subdir)
#         write_log(sample.name, log_subdir, stdout=o, stderr=e)
#     return os.path.join(count_dir, sample.name)


# def cellranger_feature_barcoding(sample, feature_dir, cellranger='cellranger', uiport=None, log_dir=None, debug=False):
#     feature_dir = os.path.abspath(feature_dir)
#     lib_csv = _make_feature_library_csv(sample, feature_dir)
#     feature_cmd = "cd '{}'".format(feature_dir)
#     feature_cmd += " && {} count --id {} --libraries '{}' --feature-ref '{}' --sample {}'.format(cellranger,
#                                                                                                  sample.name,
#                                                                                                  lib_csv,
#                                                                                                  sample.feature_reference,
#                                                                                                  sample.name)
#     p = sp.Popen(feature_cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
#     time.sleep(3)
#     uifile = os.path.join(feature_dir, '{}/_uiport'.format(self.name))
#     with open(uifile) as f:
#         uistring = f.read().strip()
#     external_ip = urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
#     uistring = 'http://' + external_ip + ':' + uistring.split(':')[-1]
#     logger.info('CellRanger UI is at {}'.format(uistring))
#     o, e = p.communicate()
#     if debug:
#         logger.info('\nCELLRANGER FEATURES')
#         logger.info(o)
#         logger.info(e)
#     if log_dir is not None:
#         log_subdir = os.path.join(log_dir, 'features')
#         make_dir(log_subdir)
#         write_log(sample.name, log_subdir, stdout=o, stderr=e)
#     return os.path.join(feature_dir, sample.name)


# def _make_feature_library_csv(samples, feature_dir):
#     lib_str = 'fastqs,sample,library_type\n'
#     for sample in samples:
#         for fastq in sample.fastqs:
#             lib_str += '{},{},{}'.format(fastq, sample.name, sample.op_type)
#     lib_path = os.path.join(feature_dir, '{}_feature-library.csv'.format(sample.name))
#     with open(lib_path, 'w') as f:
#         f.write(lib_str)
#     return lib_path


# def cellranger_aggr(samples, group, aggr_dir, normalize='mapped', cellranger='cellranger', uiport=None, log_dir=None, debug=False):
#     aggr_dir = os.path.abspath(aggr_dir)
#     aggr_csv = _make_aggr_csv(samples, aggr_dir)
#     aggr_cmd = "cd '{}'".format(aggr_dir)
#     aggr_cmd += " && {} count --id {} --csv '{}' --normalize {}".format(cellranger,
#                                                                         group,
#                                                                         aggr_csv,
#                                                                         normalize)
#     ## Eventually want to replace grabbing stdout/stderr with p.communicate(), so we can grab the standard output
#     ## in real time, parse out the url for the UI and print to screen so the user can follow along with the UI
#     p = sp.Popen(aggr_cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
#     o, e = p.communicate()
#     if debug:
#         logger.info('\nCELLRANGER AGGR')
#         logger.info(o)
#         logger.info(e)
#     if log_dir is not None:
#         log_subdir = os.path.join(log_dir, 'aggr')
#         make_dir(log_subdir)
#         write_log(group, log_subdir, stdout=o, stderr=e)
#     return os.path.join(aggr_dir, group)


# def _make_aggr_csv(samples, aggr_dir):
#     aggr_dir = os.path.join(aggr_dir)
#     aggr_csv = os.path.join(aggr_dir, 'aggr.csv')
#     lines = ['library_id,molecule_h5', ]
#     for sample in samples:
#         h5_path = os.path.join(sample.count_path, 'outs/molecule_info.h5')
#         lines.append('{},{}'.format(sample.id, h5_path))
#     with open(aggr_csv, 'w') as f:
#         f.write('\n'.join(lines))
#     return aggr_csv


def build_directory_structure(project_dir: Union[str, pathlib.Path], cfg: Config):
    """
    docstring for build_directory_structure()
    """
    dirs = {}
    make_dir(project_dir)
    shutil.copy(cfg.config_file, os.path.join(project_dir, "config.yaml"))
    dirs["run"] = os.path.join(project_dir, "run_data")
    dirs["mkfastq"] = os.path.join(project_dir, "cellranger/mkfastq")
    dirs["multi"] = os.path.join(project_dir, "cellranger/multi")
    dirs["log"] = os.path.join(project_dir, "logs")
    for path in dirs.values():
        make_dir(path)
    return dirs


def write_log(
    prefix: str,
    dir: Union[str, pathlib.Path],
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
):
    """
    docstring for write_log()
    """
    if stdout is not None:
        stdout_file = os.path.join(dir, "{}.stdout".format(prefix))
        with open(stdout_file, "w") as f:
            f.write(stdout)
    if stderr is not None:
        stderr_file = os.path.join(dir, "{}.stderr".format(prefix))
        with open(stderr_file, "w") as f:
            f.write(stderr)


def print_plan(cfg: Config):
    """
    prints the plan (runs, samples, references, etc)
    """
    print_logo()
    logger.info("======================")
    logger.info("    RUN PARAMETERS")
    logger.info("======================")
    logger.info("")
    # CellRanger version
    version_cmd = f"{cfg.cellranger} --version"
    p = sp.Popen(version_cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    o, e = p.communicate()
    cellranger_version = (
        o.decode("utf-8").replace("cellranger", "").replace("-", "").strip()
    )
    logger.info(f"CELLRANGER VERSION: {cellranger_version}")
    logger.info(f"SCAB VERSION: {__version__}")
    logger.info("")
    if cfg.gex_reference:
        gex_plural = "S" if len(cfg.gex_reference) > 1 else ""
        logger.info(f"GEX REFERENCE PATH{gex_plural}:")
        for n, r in cfg.gex_reference.items():
            logger.info(f"  - {n}: {r}")
    if cfg.vdj_reference:
        vdj_plural = "S" if len(cfg.vdj_reference) > 1 else ""
        logger.info(f"VDJ REFERENCE PATH{vdj_plural}:")
        for n, r in cfg.vdj_reference.items():
            logger.info(f"  - {n}: {r}")
    if cfg.feature_reference:
        feature_plural = "S" if len(cfg.feature_reference) > 1 else ""
        logger.info(f"FEATURE REFERENCE PATH{feature_plural}:")
        for n, r in cfg.feature_reference.items():
            logger.info(f"  - {n}: {r}")
    logger.info("")
    logger.info("RUNS:")
    for run in cfg.runs:
        logger.info(f"  {run.name}")
        if run.url is not None:
            logger.info(f"  - url: {run.url}")
        if run.path is not None:
            logger.info(f"  - path: {run.path}")
        if run.simple_csv is not None:
            logger.info(f"  - simple csv: {run.simple_csv}")
        if run.samplesheet is not None:
            logger.info(f"  - samplesheet: {run.samplesheet}")
    logger.info("")
    logger.info("SAMPLES:")
    for sample in cfg.samples:
        logger.info(f"  {sample.name}")
        for lib_type, libs in natsorted(sample.libraries_by_type.items()):
            logger.info(f"    {lib_type}: {', '.join([l.name for l in libs])}")


def print_logo():
    #     logo = '''
    #     __          __       __                 ____
    #    / /_  ____ _/ /______/ /_     ________  / / /________ _____  ____ ____  _____
    #   / __ \/ __ `/ __/ ___/ __ \   / ___/ _ \/ / / ___/ __ `/ __ \/ __ `/ _ \/ ___/
    #  / /_/ / /_/ / /_/ /__/ / / /  / /__/  __/ / / /  / /_/ / / / / /_/ /  __/ /
    # /_.___/\__,_/\__/\___/_/ /_/   \___/\___/_/_/_/   \__,_/_/ /_/\__, /\___/_/
    #                                                              /____/
    #  '''

    logo = """
   _____            __    ____                             
  / ___/_________ _/ /_  / __ \____ _____  ____ ____  _____
  \__ \/ ___/ __ `/ __ \/ /_/ / __ `/ __ \/ __ `/ _ \/ ___/
 ___/ / /__/ /_/ / /_/ / _, _/ /_/ / / / / /_/ /  __/ /    
/____/\___/\__,_/_.___/_/ |_|\__,_/_/ /_/\__, /\___/_/     
                                        /____/             
    """

    logger.info("")
    for l in logo.split("\n"):
        logger.info(l)


def print_runs_header():
    logger.info("")
    logger.info("")
    logger.info("")
    logger.info("=======================")
    logger.info("    SEQUENCING RUNS")
    logger.info("=======================")


def print_samples_header():
    # logger.info('')
    logger.info("")
    logger.info("")
    logger.info("===============")
    logger.info("    SAMPLES")
    logger.info("===============")


# def print_op_splash(op, samples):
#     # TODO
#     pass


# def print_aggr_splash(aggr):
#     # TODO
#     pass


def main(args: Args):
    # parse the config file
    cfg = Config(args.config_file)

    # build directory structure
    dirs = build_directory_structure(args.project_dir, cfg)

    # setup logging and print plan
    run_log = os.path.join(dirs["log"], "batch_cellranger.log")
    log.setup_logging(run_log, print_log_location=False, debug=args.debug)
    global logger
    logger = log.get_logger("batch_cellranger")
    print_plan(cfg)

    # sequencing runs
    print_runs_header()
    for run in cfg.runs:
        run.print_splash()
        # get data
        run.get(dirs["run"], log_dir=dirs["log"], debug=args.debug)
        run.print_get_completion()
        if not run.successful_get:
            continue
        # mkfastq
        run.mkfastq(
            dirs["mkfastq"],
            cellranger=cfg.cellranger,
            log_dir=dirs["log"],
            cli_options=cfg.get_mkfastq_cli_options(run.name),
            debug=args.debug,
        )
        run.print_mkfastq_completion()
        for sample in cfg.samples:
            for library in sample.libraries:
                # if library.name in run.successful_mkfastq_libraries:
                if library.name in run.libraries:
                    library.add_fastq_path(run.fastq_path)

    # cellranger multi
    print_samples_header()
    for sample in cfg.samples:
        if not sample.libraries:
            continue
        sample.print_splash()
        # sample.make_config_csv(config_csv)
        # config_csv = os.path.join(dirs['multi'], f"{sample.name}_config.csv")
        cellranger_multi(
            sample,
            dirs["multi"],
            cellranger=cfg.cellranger,
            uiport=cfg.uiport,
            log_dir=dirs["log"],
            cli_options=cfg.get_multi_cli_options(sample.name),
            debug=args.debug,
        )
    logger.info("")
    logger.info("")

    # compress
    # TODO

    # upload to S3
    # TODO

    # # operations (except aggr)
    # opmap = {'vdj': cellranger_vdj,
    #          'count': cellranger_count,
    #          'features': cellranger_feature_barcoding}

    # for op in ['vdj', 'count', 'features']:
    #     print_op_splash(op)
    #     opfunction = opmap[op]
    #     for sample in cfg.samples:
    #         if op not in sample.ops:
    #             continue
    #         opfunction(sample,
    #                    dirs[op],
    #                    cellranger=cfg.cellranger,
    #                    uiport=cfg.uiport,
    #                    log_dir=dirs['log'],
    #                    debug=args.debug)

    # vdj
    # print_op_splash('vdj', cfg.samples)
    # for sample in cfg.samples:
    #     if 'vdj' not in sample.ops:
    #         continue
    #     path = cellranger_vdj(sample,
    #                           dirs['vdj'],
    #                           cellranger=cfg.cellranger,
    #                           uiport=cfg.uiport,
    #                           log_dir=dirs['log'],
    #                           debug=args.debug)
    #     sample.vdj_path = path

    # # count
    # print_op_splash('count', cfg.samples)
    # for group, sample_dict in cfg.ops['count']:
    #     samples = [s for s in cfg.samples if s.name in sample_dict]
    #     for s in samples:
    #         s.op_type = sample_dict[s.name]
    #     path = cellranger_count(samples,
    #                             dirs['count'],
    #                             cellranger=cfg.cellranger,
    #                             uiport=cfg.uiport,
    #                             log_dir=dirs['log'],
    #                             debug=args.debug)

    # for sample in cfg.samples:
    #     if 'count' not in sample.ops:
    #         continue
    #     path = cellranger_count(sample,
    #                             dirs['count'],
    #                             cellranger=cfg.cellranger,
    #                             uiport=cfg.uiport,
    #                             log_dir=dirs['log'],
    #                             debug=args.debug)
    #     sample.count_path = path

    # # features
    # print_op_splash('features', cfg.samples)
    # for sample in cfg.samples:
    #     if 'features' not in sample.ops:
    #         continue
    #     path = cellranger_feature_barcoding(sample,
    #                                         dirs['features'],
    #                                         cellranger=cfg.cellranger,
    #                                         uiport=cfg.uiport,
    #                                         log_dir=dirs['log'],
    #                                         debug=args.debug)
    #     sample.feature_path = path

    # # aggr
    # print_aggr_splash(cfg.ops['aggr'])
    # for group, sample_names in cfg.ops['aggr'].items():
    #     samples = [s for s in cfg.samples if s.name in sample_names]
    #     path = cellranger_aggr(samples,
    #                            group,
    #                            dirs['aggr'],
    #                            normalize='mapped',
    #                            cellranger=cfg.cellranger,
    #                            uiport=cfg.uiport,
    #                            log_dir=dirs['log'],
    #                            debug=args.debug)
    #     for s in samples:
    #         s.aggr_path = path

    # compress


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
