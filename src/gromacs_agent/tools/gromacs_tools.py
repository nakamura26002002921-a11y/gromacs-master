# src/gromacs_agent/tools/gromacs_tools.py
import subprocess
from typing import Tuple
import structlog

logger = structlog.get_logger()

class GromacsTools:
    @staticmethod
    def run_gmx_command(cmd: str, args: list) -> Tuple[int, str, str]:
        """
        Execute a GROMACS command.
        Returns: (return_code, stdout, stderr)
        """
        full_cmd = ["gmx", cmd] + args
        logger.info("Executing GROMACS command", cmd=full_cmd)
        
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=3600 # 1 hour timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except FileNotFoundError:
            return -1, "", "GROMACS not found in PATH"
