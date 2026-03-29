#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SOL ExecBench profile runner — self-contained script invoked under NCU.

Reuses the same staging files (definition.json, workload.jsonl, solution.json,
config.json) produced by ProblemPackager.  Generates the same inputs with the
same seed and calls the target kernel exactly once so NCU can capture a profile.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

STAGING_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(STAGING_DIR))

from sol_execbench.core.bench.config import BenchmarkConfig  # noqa: E402
from sol_execbench.core.bench.correctness import set_seed  # noqa: E402
from sol_execbench.core.bench.io import (  # noqa: E402
    allocate_outputs,
    gen_inputs,
    load_safetensors,
    normalize_outputs,
)
from sol_execbench.core import (  # noqa: E402
    Definition,
    Solution,
    SupportedLanguages,
    Workload,
)
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a single kernel under NCU")
    parser.add_argument(
        "--target",
        choices=["user", "ref"],
        required=True,
        help="Which kernel to profile",
    )
    parser.add_argument(
        "--workload-uuid",
        required=True,
        help="UUID of the workload to profile",
    )
    args = parser.parse_args()

    # -- Load config & set seed --
    _config_path = STAGING_DIR / "config.json"
    bench_config = (
        BenchmarkConfig(**json.loads(_config_path.read_text()))
        if _config_path.exists()
        else BenchmarkConfig()
    )
    set_seed(bench_config.seed)

    # -- Load definition & workloads --
    definition = Definition(**json.loads((STAGING_DIR / "definition.json").read_text()))
    workloads_raw = []
    wkl_path = STAGING_DIR / "workload.jsonl"
    if wkl_path.exists():
        for line in wkl_path.read_text().splitlines():
            line = line.strip()
            if line:
                workloads_raw.append(json.loads(line))

    workloads = [Workload(**w) for w in workloads_raw]

    # Find matching workload
    target_workload = None
    for wl in workloads:
        if str(wl.uuid) == args.workload_uuid:
            target_workload = wl
            break
    if target_workload is None:
        print(
            f"Workload UUID {args.workload_uuid} not found",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- Generate inputs --
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    resolved_axes = definition.get_resolved_axes_values(target_workload.axes)

    safe_tensors: dict = {}
    if any(v.type == "safetensors" for v in target_workload.inputs.values()):
        safetensors_roots = [STAGING_DIR]
        benchmark_dir = os.environ.get("FLASHINFER_TRACE_DIR", None)
        if benchmark_dir:
            safetensors_roots.append(Path(benchmark_dir))
        safe_tensors = load_safetensors(definition, target_workload, safetensors_roots)

    solution = Solution(**json.loads((STAGING_DIR / "solution.json").read_text()))
    custom_inputs_fn = None

    inputs = gen_inputs(
        definition,
        target_workload,
        device=device,
        safe_tensors=safe_tensors or None,
        custom_inputs_fn=custom_inputs_fn,
    )

    # -- Run target kernel --
    if args.target == "ref":
        # Import reference code
        ref_file = STAGING_DIR / "_reference.py"
        ref_spec = importlib.util.spec_from_file_location("_reference", ref_file)
        ref_module = importlib.util.module_from_spec(ref_spec)
        ref_spec.loader.exec_module(ref_module)
        ref_fn = vars(ref_module).get("run")
        if ref_fn is None:
            print("Reference code does not define 'run'", file=sys.stderr)
            sys.exit(1)
        ref_fn(*inputs)
    else:
        # Import user solution
        entry_point = solution.spec.entry_point
        if "::" in entry_point:
            entry_module_or_file, entry_func_name = entry_point.rsplit("::", 1)
        else:
            entry_module_or_file, entry_func_name = entry_point, "run"

        _CPP_LANGUAGES = {
            SupportedLanguages.CUDA_CPP,
            SupportedLanguages.CUTLASS,
            SupportedLanguages.CUDNN,
            SupportedLanguages.CUBLAS,
        }
        if any(lang in _CPP_LANGUAGES for lang in solution.spec.languages):
            so_path = STAGING_DIR / "benchmark_kernel.so"
            spec_obj = importlib.util.spec_from_file_location(
                "benchmark_kernel", so_path
            )
            user_mod = importlib.util.module_from_spec(spec_obj)
            spec_obj.loader.exec_module(user_mod)
            user_fn = getattr(user_mod, entry_func_name)
        else:
            mod_name = (
                entry_module_or_file.removesuffix(".py")
                .replace("/", ".")
                .replace(os.sep, ".")
            )
            user_mod = importlib.import_module(mod_name)
            user_fn = getattr(user_mod, entry_func_name)

        output_names = list(definition.outputs.keys())
        output_dtypes_torch = {
            k: dtype_str_to_torch_dtype(v.dtype) for k, v in definition.outputs.items()
        }

        if solution.spec.destination_passing_style:
            outputs = allocate_outputs(definition, resolved_axes, device)
            user_fn(*inputs, *outputs)
        else:
            result = user_fn(*inputs)
            normalize_outputs(
                result,
                device=torch.device(device),
                output_names=output_names,
                output_dtypes=output_dtypes_torch,
            )

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
