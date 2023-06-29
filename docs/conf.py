# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess

from rocm_docs import ROCmDocs


external_projects_current_project = "rccl"

name = "RCCL"
get_major = r'sed -n -e "s/^NCCL_MAJOR.*\([0-9]\+\).*/\1/p" ../makefiles/version.mk'
get_minor = r'sed -n -e "s/^NCCL_MINOR.*\([0-9]\{2,\}\).*/\1/p" ../makefiles/version.mk'
get_patch = r'sed -n -e "s/^NCCL_PATCH.*\([0-9]\+\).*/\1/p" ../makefiles/version.mk'
major = subprocess.getoutput(get_major)
minor = subprocess.getoutput(get_minor)
patch = subprocess.getoutput(get_patch)

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(f"{name} {major}.{minor}.{patch} Documentation")
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/docBin/xml")
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
