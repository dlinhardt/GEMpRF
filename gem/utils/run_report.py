# -*- coding: utf-8 -*-

"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024-2025, Medical University of Vienna",
"@Desc    :   Collects what was completed, skipped and what failed during a run,
              prints a summary and writes a plain-text report to the results dir.",
"""

import os
import datetime
import traceback

from gem.utils.logger import Logger


class RunReport:
    def __init__(self, run_type, gemprf_version, config_filepath, result_dir):
        self.run_type = run_type
        self.gemprf_version = gemprf_version
        self.config_filepath = config_filepath
        self.result_dir = result_dir

        self.start_time = datetime.datetime.now()
        self.completed = []   # list of (input_desc, duration_sec)
        self.skipped = []     # list of result filepaths
        self.failed = []      # list of (input_desc, error_type, message, traceback_str)
        self.grid_fallbacks = []  # list of (input_desc, stats_dict) for refined -> grid reverts

    # ------------------------------------------------------------------ counts
    @property
    def num_completed(self):
        return len(self.completed)

    @property
    def num_skipped(self):
        return len(self.skipped)

    @property
    def num_failed(self):
        return len(self.failed)

    @property
    def num_total(self):
        return self.num_completed + self.num_skipped + self.num_failed

    # ----------------------------------------------------------------- collect
    def add_completed(self, input_desc, duration_sec):
        self.completed.append((input_desc, duration_sec))

    def add_skipped(self, result_filepath):
        self.skipped.append(result_filepath)

    def add_failed(self, input_desc, exc):
        # traceback.format_exc() only has something to report inside an "except" block
        tb = traceback.format_exc()
        if not tb or tb.startswith("NoneType"):
            tb = ""
        self.failed.append((input_desc, type(exc).__name__, str(exc), tb))

    def add_grid_fallback(self, input_desc, stats):
        # stats: dict with keys total, on_grid, worse_error, x_too_far, y_too_far,
        #        sigma_too_far, nan_refined, zero_signal
        self.grid_fallbacks.append((input_desc, dict(stats)))

    # ------------------------------------------------------------------ render
    def render(self):
        duration = datetime.datetime.now() - self.start_time
        lines = [
            "GEMpRF run report",
            "=================",
            f"Version:   {self.gemprf_version}",
            f"Config:    {self.config_filepath}",
            f"Started:   {self.start_time:%Y-%m-%d %H:%M:%S}",
            f"Duration:  {duration}",
            f"Run type:  {self.run_type}",
            "",
            "Summary",
            "-------",
            f"Total analyses : {self.num_total}",
            f"  completed    : {self.num_completed}",
            f"  skipped      : {self.num_skipped} (result already exists)",
            f"  failed       : {self.num_failed}",
        ]

        if self.skipped:
            lines += ["", f"Skipped ({self.num_skipped})", "-" * len(f"Skipped ({self.num_skipped})")]
            lines += [f"  {path}" for path in self.skipped]

        if self.failed:
            lines += ["", f"Failed ({self.num_failed})", "-" * len(f"Failed ({self.num_failed})")]
            for input_desc, error_type, message, tb in self.failed:
                lines.append(f"  {input_desc}")
                lines.append(f"    {error_type}: {message}")
                if tb:
                    lines += [f"    {tb_line}" for tb_line in tb.rstrip().splitlines()]
                lines.append("")

        if self.grid_fallbacks:
            title = "On-grid fallbacks (refined fit reverted to grid point)"
            lines += ["", title, "-" * len(title)]
            for input_desc, s in self.grid_fallbacks:
                lines.append(f"  {input_desc}")
                lines.append(f"      vertices                 : {s.get('total', 0)}")
                lines.append(f"      reverted to grid (any)   : {s.get('on_grid', 0)}")
                lines.append(f"        worse error            : {s.get('worse_error', 0)}")
                lines.append(f"        x > 2 grid steps       : {s.get('x_too_far', 0)}")
                lines.append(f"        y > 2 grid steps       : {s.get('y_too_far', 0)}")
                lines.append(f"        sigma > 2 grid steps   : {s.get('sigma_too_far', 0)}")
                lines.append(f"        nan refined params     : {s.get('nan_refined', 0)}")
                lines.append(f"      zero model signal (R2=-2): {s.get('zero_signal', 0)}")
                lines.append("")

        return "\n".join(lines) + "\n"

    # ---------------------------------------------------------------- finalize
    def finalize(self):
        """Print the summary to stdout, write the report file and print its path."""
        summary = (f"Analyses: {self.num_completed} completed, "
                   f"{self.num_skipped} skipped, {self.num_failed} failed.")
        if self.num_failed:
            Logger.print_red_message(summary, print_file_name=False)
        else:
            Logger.print_green_message(summary, print_file_name=False)

        report_filepath = self.write()
        if report_filepath is not None:
            Logger.print_blue_message(f"Run report written to: {report_filepath}", print_file_name=False)
        return report_filepath

    def write(self):
        """Write the report to the results directory. Never raises."""
        try:
            os.makedirs(self.result_dir, exist_ok=True)
            filename = f"gem_run_report_{self.start_time:%Y%m%d-%H%M%S}.txt"
            report_filepath = os.path.join(self.result_dir, filename)
            with open(report_filepath, "w", encoding="utf-8") as f:
                f.write(self.render())
            return report_filepath
        except Exception as exc:
            Logger.print_red_message(f"Could not write the run report: {exc}", print_file_name=False)
            return None
