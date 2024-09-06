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

# Define the workspace directory (change this to your actual workspace path)

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
    
    lg_config = litgen.options.LitgenOptions()
    generator = litgen.LitgenGenerator(lg_config)

    for hppfile in hpp_files:
        generator.process_cpp_file(str(hppfile))

    generator.write_generated_code(str(output_binding_file), str(output_stub_file))

    # Read the generated binding file
    with open(str(output_binding_file), 'r') as file:
        binding_code = file.read()

    # Replace unique_ptr with shared_ptr for the Value class
    binding_code = re.sub(r'py::class_<Value>(\s*\([^)]*\))', r'py::class_<Value, std::shared_ptr<Value>>\1', binding_code)

    # Write the modified code back to the file
    with open(str(output_binding_file), 'w') as file:
        file.write(binding_code)



# # Generate the binding.cpp file
# binding_code = lg.write_generated_code()
# output_binding_file.write_text(binding_code)

# print(f"Bindings have been written to {output_binding_file}")

# # Optionally, you can format the generated binding.cpp file using clang-format
# subprocess.run(["clang-format", "-i", str(output_binding_file)])
