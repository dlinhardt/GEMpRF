"""BIDS input discovery test for an individual run.

Checks that `GemBidsHandler.get_input_filepaths` resolves the expected bold file from the bundled
`testdata/bids_test` tree, given a BIDS filter. The current handler allows a single task per
individual run, so this queries one task (`dummybar`) and expects the one matching file
(sub-dummy / ses-001 / run-01 / hemi-L). Empirically confirmed against the test data.

Config is built from `analysis_configs/analysis_config.xml` (the template matching the current
parser and version) and loaded via `ConfigurationWrapper` directly, so the test needs no GPU/CuPy.
"""
import os
import shutil
import unittest
import xml.etree.ElementTree as ET


class GemIndividualRunBidsHandling(unittest.TestCase):

    def setUp(self):
        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        gem_dir = os.path.dirname(self.current_script_dir)

        self.bids_base = os.path.join(self.current_script_dir, "testdata", "bids_test", "BIDS")
        self.stimuli_dir = os.path.join(
            self.bids_base, "derivatives", "prfprepare", "analysis-01", "sub-dummy", "stimuli")

        self.temp_dir = os.path.join(self.current_script_dir, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        template = os.path.join(gem_dir, "configs", "analysis_configs", "analysis_config.xml")
        self.config_path = os.path.join(self.temp_dir, "test_bids_handler_config.xml")
        shutil.copyfile(template, self.config_path)
        self._write_bids_config()

    def _write_bids_config(self):
        """Patch the template to point the BIDS filter at the bundled test tree (single task)."""
        tree = ET.parse(self.config_path)
        root = tree.getroot()

        bids = root.find(".//input_datasrc/BIDS")
        bids.set("enable", "True")
        bids.set("run_type", "individual")
        bids.find("basepath").text = self.bids_base
        bids.find("append_to_basepath").text = "derivatives, prfprepare"
        bids.find("analysis").text = "01"
        bids.find("sub").text = "dummy"
        bids.find("hemi").text = "L"
        bids.find("space").text = "all"
        bids.find("input_file_extension").text = ".nii.gz"
        bids.find("individual/task").text = "dummybar"
        bids.find("individual/ses").text = "001"
        bids.find("individual/run").text = "01"

        # the handler resolves each task's stimulus from cfg.stimulus/directory
        root.find(".//stimulus/directory").text = self.stimuli_dir

        tree.write(self.config_path)

    def test_bids_handler_finds_expected_input_file(self):
        from gem.configs.config_manager import ConfigurationWrapper
        from gem.data.bids_handler import GemBidsHandler

        ConfigurationWrapper.load_configuration(config_filepath=self.config_path)

        measured_data_info_list = GemBidsHandler.get_input_filepaths(
            ConfigurationWrapper.bids, stimuli_dir_path=ConfigurationWrapper.stimulus["directory"])
        found = [info[0] for info in measured_data_info_list]

        expected = os.path.join(
            self.bids_base, "derivatives", "prfprepare", "analysis-01", "sub-dummy",
            "ses-001", "func", "sub-dummy_ses-001_task-dummybar_run-01_hemi-L_bold.nii.gz")

        self.assertEqual(found, [expected],
                         f"BIDS handler returned {found}, expected exactly [{expected}]")


if __name__ == "__main__":
    unittest.main()
