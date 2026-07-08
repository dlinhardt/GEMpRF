# -*- coding: utf-8 -*-

"""
"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024-2025, Medical University of Vienna",
"@Desc    :   Exceptions raised for a single failing analysis. These are caught by
              the per-analysis loops so that the remaining analyses still run.",
"""


class GemAnalysisError(Exception):
    """Base class for errors that only invalidate a single analysis."""
    pass


class TimepointMismatchError(GemAnalysisError):
    """Number of timepoints in the measured data does not match the stimulus."""
    pass


class InputFileMissingError(GemAnalysisError):
    """An input data file referenced by the configuration does not exist."""
    pass
