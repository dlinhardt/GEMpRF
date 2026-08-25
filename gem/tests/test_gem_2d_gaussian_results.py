"""End-to-end result tests on the simulated 3n2 pRF (true parameters x=3, y=2, sigma=1).

These run the full pipeline through the real entry point ``gem.run(config_filepath)`` on a small
fixed-paths (non-BIDS) config, then check the written estimates. Two guards:

  1. results match the pinned benchmark within 0.01 -- a regression/invariance guard. On this clean
     simulated pRF the refined-fit -> grid fallback should trigger on ~0 vertices, so the estimates
     should still match the pre-change benchmark. If this fails, the fallback fired on the test data
     (or a default changed): investigate before re-baselining the benchmark.
  2. the recovered pRF centre/size averages to the simulated (3, 2, 1) -- robust to small numerical
     changes and independent of the benchmark.

Requires CuPy + a GPU and the test data under ``testdata/simulated/3n2``. The config is built from
``analysis_configs/analysis_config.xml`` (the template matching the current parser and the
``version`` the loader enforces), patched to the current schema.
"""
import os
import shutil
import unittest
import xml.etree.ElementTree as ET


class GemAnalysisTests(unittest.TestCase):

    def setUp(self):
        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        gem_dir = os.path.dirname(self.current_script_dir)
        self.data_dir = os.path.join(self.current_script_dir, "testdata", "simulated", "3n2")

        self.temp_dir = os.path.join(self.current_script_dir, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        # Base the test config on the current-schema template (carries the version the loader checks).
        template = os.path.join(gem_dir, "configs", "analysis_configs", "analysis_config.xml")
        self.config_path = os.path.join(self.temp_dir, "test_config.xml")
        shutil.copyfile(template, self.config_path)
        self._write_fixed_paths_config()

    def _write_fixed_paths_config(self):
        """Patch the template into a single-file, fixed-paths (non-BIDS) 51x51x8 run in JSON.

        Done with ElementTree in one pass (rather than the find-first XML helpers) so the single
        measured-data filepath and the grid/sigma attributes can be set unambiguously.
        """
        bold = os.path.join(self.data_dir, "sub-001_ses-3n2_task-prf_acq-normal_run-01_bold.nii.gz")
        apertures = os.path.join(self.data_dir, "sub-001_ses-3n2_task-prf_apertures.nii.gz")

        tree = ET.parse(self.config_path)
        root = tree.getroot()

        # refine fitting stays on (default) -- that is the path the fallback lives in.

        # use the fixed-paths branch, not BIDS
        root.find(".//input_datasrc/BIDS").set("enable", "False")

        # stimulus (the model uses fixed_paths/stimulus_filepath; <stimulus> supplies field/size)
        root.find(".//stimulus/visual_field").text = "10"
        root.find(".//stimulus/width").text = "101"
        root.find(".//stimulus/height").text = "101"
        root.find(".//fixed_paths/stimulus_filepath").text = apertures

        # exactly one measured-data filepath (parsed as a scalar -> a single analysis)
        mdf = root.find(".//fixed_paths/measured_data_filepath")
        for fp in mdf.findall("filepath"):
            mdf.remove(fp)
        ET.SubElement(mdf, "filepath").text = bold

        # results: plain JSON in the temp dir, filename == <input>_estimates.json
        results = root.find(".//fixed_paths/results")
        results.set("output_format", "json")
        results.find("basepath").text = self.temp_dir
        results.find("custom_filename_postfix").text = ""
        results.find("prepend_date").text = "False"

        # model + measured-data batching
        root.find(".//pRF_model/model").text = "2d_gaussian"
        root.find(".//measured_data/batches").text = "4"

        # search grid: 51 x 51 x 8, radius 13.5 (matches the pinned benchmark)
        grid = root.find(".//default_spatial_grid")
        grid.set("visual_field_radius", "13.5")
        grid.set("num_horizontal_prfs", "51")
        grid.set("num_vertical_prfs", "51")
        sigmas = root.find(".//default_sigmas")
        sigmas.set("num_sigmas", "8")
        sigmas.set("min_sigma", "0.5")
        sigmas.set("max_sigma", "5")

        tree.write(self.config_path)

    def _result_json(self):
        return os.path.join(self.temp_dir, "sub-001_ses-3n2_task-prf_acq-normal_run-01_estimates.json")

    def test_results_match_benchmark(self):
        """Full run reproduces the pinned benchmark within 0.01 (regression / invariance guard)."""
        import gem
        from gem.utils.gem_load_estimations import EstimationsUtils

        gem.run(config_filepath=self.config_path)

        benchmark = os.path.join(
            self.data_dir,
            "2024-06-18_sub-001_ses-3n2_task-prf_acq-normal_run-01_estimates_[gem-benchmark_51x51x8].json")
        new_results = EstimationsUtils.get_estimation_data(filepath=self._result_json())
        benchmark_results = EstimationsUtils.get_estimation_data(filepath=benchmark)

        max_difference = EstimationsUtils.compare_estimation_results(
            new_results=new_results, benchmark_results=benchmark_results)

        self.assertLessEqual(
            max_difference, 0.01,
            f"Results diverged from the benchmark (max difference {max_difference}). If the refined-fit "
            f"grid fallback intentionally changed these vertices, re-baseline the benchmark; otherwise "
            f"this is a regression.")

    def test_recovers_simulated_prf_3n2(self):
        """The recovered pRF averages to the simulated centre/size (x=3, y=2, sigma=1)."""
        import gem
        from gem.utils.gem_load_estimations import EstimationsUtils

        gem.run(config_filepath=self.config_path)

        results = EstimationsUtils.get_estimation_data(filepath=self._result_json())
        mean_x, mean_y, mean_sigma = EstimationsUtils.get_avg_2d_gaussian_estimated_values(json_data=results)

        self.assertTrue(2.9 < mean_x < 3.1, f"mean Centerx0 = {mean_x} not in 2.9-3.1")
        self.assertTrue(1.9 < mean_y < 2.1, f"mean Centery0 = {mean_y} not in 1.9-2.1")
        self.assertTrue(0.9 < mean_sigma < 1.1, f"mean sigmaMajor = {mean_sigma} not in 0.9-1.1")


if __name__ == "__main__":
    unittest.main()
