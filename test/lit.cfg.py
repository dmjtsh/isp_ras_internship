import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

config.name = "LIFE_RANGE"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.life_range_obj_root, "test")
config.life_range_tools_dir = os.path.join(config.life_range_obj_root, "bin")
config.life_range_libs_dir = os.path.join(config.life_range_obj_root, "lib")

config.substitutions.append(("%life_range_libs", config.life_range_libs_dir))

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.life_range_tools_dir, config.llvm_tools_dir]
tools = [
    "lr-mlir-opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

llvm_config.with_environment(
    "PYTHONPATH",
    [
        os.path.join(config.mlir_obj_dir, "python_packages", "life_range"),
    ],
    append_path=True,
)
