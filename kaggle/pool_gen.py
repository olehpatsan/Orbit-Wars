"""
Worker module for parallel map pool generation.

Launched via ProcessPoolExecutor with the 'spawn' context so environment variables
are set before JAX imports. Forcing CPU in worker processes prevents multiple
workers from competing for the GPU, which causes OOM errors or deadlocks.
The os.environ assignments must appear before any import that transitively imports JAX.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import logging
os.environ["LITELLM_LOG"] = "ERROR"
logging.disable(logging.WARNING)

from orbit_rollout import make_init_states


def gen_chunk(seeds):
    """Generates a batch of initial game states from a list of seeds."""
    return make_init_states(seeds)
