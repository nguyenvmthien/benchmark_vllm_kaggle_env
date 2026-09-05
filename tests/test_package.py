import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path


class InstalledPackageTests(unittest.TestCase):
    def test_module_and_console_commands_work_outside_checkout(self):
        executable = "infercap.exe" if sys.platform == "win32" else "infercap"
        console = Path(sysconfig.get_path("scripts")) / executable
        commands = ([sys.executable, "-m", "infercap"], [str(console)])
        with tempfile.TemporaryDirectory() as directory:
            for command in commands:
                for subcommand in ([], ["check"], ["benchmark"]):
                    with self.subTest(command=command, subcommand=subcommand):
                        result = subprocess.run(
                            [*command, *subcommand, "--help"],
                            cwd=directory, capture_output=True, text=True,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("usage: infercap", result.stdout)
                        if subcommand:
                            self.assertIn(f"infercap {subcommand[0]}", result.stdout)

    def test_invalid_subcommand_returns_usage_error(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-m", "infercap", "invalid"],
                cwd=directory, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)
