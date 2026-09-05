import io
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from infercap.cli import main


class CLITests(unittest.TestCase):
    def test_no_command_shows_help_without_running_workload(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main([]), 0)
        self.assertIn("check", output.getvalue())
        self.assertIn("benchmark", output.getvalue())

    def test_top_level_help_does_not_import_workload_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; "
             "sys.modules.update({name: None for name in "
             "('aiohttp', 'matplotlib', 'vllm', 'torch')}); "
             "from infercap.cli import main; raise SystemExit(main(['--help']))"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("check", result.stdout)
        self.assertIn("benchmark", result.stdout)

    def test_check_forwards_options_and_exit_status(self):
        argv = ["check", "Qwen", "--offline", "--json"]
        for status in (0, 1):
            with self.subTest(status=status), patch(
                "infercap.preflight.cli.main", return_value=status
            ) as check:
                self.assertEqual(main(argv), status)
                check.assert_called_once_with(argv[1:], prog="infercap check")
        self.assertEqual(argv, ["check", "Qwen", "--offline", "--json"])

    def test_benchmark_forwards_options_and_normalizes_success(self):
        with patch("infercap.benchmark.cli.main", return_value=None) as benchmark:
            self.assertEqual(main(["benchmark", "--concurrency", "1,4"]), 0)
            benchmark.assert_called_once_with(
                ["--concurrency", "1,4"], prog="infercap benchmark"
            )

    def test_subcommand_help_uses_its_own_options_and_name(self):
        for command, option in (("check", "--offline"), ("benchmark", "--concurrency")):
            with self.subTest(command=command):
                output = io.StringIO()
                with redirect_stdout(output), self.assertRaises(SystemExit) as exit:
                    main([command, "--help"])
                self.assertEqual(exit.exception.code, 0)
                self.assertIn(f"usage: infercap {command}", output.getvalue())
                self.assertIn(option, output.getvalue())

    def test_invalid_commands_and_options_fail_before_work(self):
        cases = (
            ["unknown"], ["--model", "example"],
            ["check", "--unknown"], ["benchmark", "--unknown"],
            ["check", "--tensor-parallel-size", "0"],
            ["benchmark", "--concurrency", "0"],
        )
        for argv in cases:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exit:
                    main(argv)
                self.assertEqual(exit.exception.code, 2)
