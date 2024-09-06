import sys
import re
from typing import List, Optional

import os
import shutil
from pathlib import Path
import litgen
import litgen.options

from setuptools import setup
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext

def build(cpp_files, binding_cpp_file, output_dir) -> None:
    # name
    name = "py_nn"

    # remove prev build
    tmp_folder = os.path.join(output_dir, "temp")
    if os.path.isdir(tmp_folder):
        shutil.rmtree(tmp_folder)

    # Change the argv to execute the setup.py
    sys.argv = [
        sys.executable, 
        "build_ext", 
        f"--build-lib={output_dir}",       # Where the final .so file will be placed
        f"--build-temp={output_dir}/temp"  # Where the temporary object files will be placed
    ]

    # If include dir is None just set as empty list
    include_dirs = []

    # Make the module
    ext_modules = [
        Pybind11Extension(
            name,
            [binding_cpp_file, *cpp_files],
            include_dirs=[pybind11.get_include(), *include_dirs],
            extra_compile_args = None,
            extra_link_args = None)
    ]

    # Finally exectute the setup
    setup(
        name=name,
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
    )

if __name__ == "__main__":
    workspace_dir = Path("")
    output_binding_file = workspace_dir / "src" / "py_nn" / "binding.cpp"
    output_stub_file = workspace_dir / "src" / "py_nn" / "py_nn.pyi"
    hpp_files = list(workspace_dir.rglob("*.hpp"))
    cpp_files = list(workspace_dir.rglob("*.cpp"))

    # TEMP TODO DELETE THIS
    def mfilt(f: str) -> bool:
        for filt in ["Values", "NodesTypes", "Tensor"]:
            if filt.lower() in str(f).lower():
                return True
        return False
    hpp_files = [f for f in hpp_files if mfilt(f)]
    cpp_files = [f for f in cpp_files if mfilt(f)]
    build(cpp_files, str(output_binding_file), "src/py_nn")