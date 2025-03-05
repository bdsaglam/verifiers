import os
from typing import Optional

from e2b_code_interpreter import Sandbox

from verifiers.codex.models import CodeExecutor


class E2BPythonExecutor(CodeExecutor):
    """
    Python code executor using E2B sandbox.

    This executor runs Python code in a secure E2B sandbox and returns the output.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the E2B Python executor."""
        # Get API key from environment variable
        self.api_key = api_key or os.environ.get("E2B_API_KEY")
        if not self.api_key:
            raise ValueError("E2B_API_KEY environment variable is not set")

    def execute(self, code: str, timeout: int = 10, **kwargs) -> str:
        """
        Execute the given Python code in a secure E2B sandbox and return the output.

        Args:
            code (str): The Python code to execute.
            timeout (int): The timeout in seconds for the code execution. Default is 10 seconds.
        Returns:
            str: The output of the executed code.
        """

        with Sandbox(api_key=self.api_key) as sandbox:
            execution = sandbox.run_code(code, timeout=timeout)
            if execution.error:
                return str(execution.error)
            else:
                return execution.text

    def destroy(self, **kwargs) -> None:
        """
        Clean up resources used by the executor.
        This closes the sandbox if it's open.
        """
        pass
