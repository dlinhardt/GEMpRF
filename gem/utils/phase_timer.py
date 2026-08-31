# -*- coding: utf-8 -*-

"""Per-phase wall-clock accounting for the inside of one analysis.

The run report says how long a file took; this says where that time went -- grid search, derivative
gather, refinement, R2, and so on -- so that an optimisation can be pointed at the phase that
actually costs something.
"""

import time
from contextlib import contextmanager

import cupy as cp

from gem.utils.logger import Logger


class PhaseTimer:
    """Accumulate wall time per named phase, in the order the phases are first seen.

    Disabled by default, and a genuine no-op when disabled. That matters: CUDA calls are
    asynchronous, so reading the clock around a CuPy call measures kernel *launch* time and nothing
    else. Making the numbers mean anything requires a device synchronise at every phase boundary,
    which serialises work the pipeline would otherwise overlap. So this is gated on
    ``write_debug_info`` -- the same flag ``Logger.enable_timing`` reads -- and a normal analysis
    pays neither the sync nor the bookkeeping.
    """

    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.totals = {}

    @staticmethod
    def _synchronise_all_devices():
        # Phases can leave work in flight on any device the model signals were split across, so
        # syncing only the current one would bill that work to whichever phase happens to touch it.
        for device_id in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(device_id):
                cp.cuda.runtime.deviceSynchronize()

    @contextmanager
    def phase(self, name):
        if not self.enabled:
            yield
            return

        self._synchronise_all_devices()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._synchronise_all_devices()
            self.totals[name] = self.totals.get(name, 0.0) + (time.perf_counter() - start)

    def add(self, name, seconds):
        """Record a phase measured outside a ``phase()`` block (the h5 write, which has no GPU work)."""
        if self.enabled:
            self.totals[name] = self.totals.get(name, 0.0) + float(seconds)

    def render(self):
        """One line, phases ordered by cost, with each one's share of the measured total."""
        if not self.totals:
            return ""
        total = sum(self.totals.values())
        ranked = sorted(self.totals.items(), key=lambda kv: kv[1], reverse=True)
        parts = [f"{name} {seconds:.1f}s ({100 * seconds / total:.0f}%)" for name, seconds in ranked]
        return f"fit phases [{total:.1f}s measured]: " + ", ".join(parts)

    def report(self):
        """Print the breakdown; no-op when timing is off."""
        line = self.render()
        if line:
            Logger.print_timing_message(line)
