import subprocess

from verifiers.codex.models import CodeExecutor


class LocalPythonExecutor(CodeExecutor):
    """
    Executes Python code locally in a subprocess for isolation.
    WARNING: This is dangerous as it executes arbitrary Python code directly on the host machine.
    Even with subprocess isolation, malicious code could potentially harm the system.
    Use only in controlled environments with trusted code.
    """

    def execute(self, code: str, timeout: int = 10, **kwargs) -> str:
        """
        Execute the given code and return the output.

        Args:
            code: The Python code to execute.
            timeout: Maximum execution time in seconds.
            **kwargs: Additional arguments (unused).

        Returns:
            The output of the code execution as a string.
        """
        try:
            # Run the code block in subprocess with timeout
            result = subprocess.run(
                ["python", "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True,
            )
            if result.stderr:
                return f"Error: {result.stderr.strip()}"
            return result.stdout.strip() if result.stdout else ""
        except subprocess.TimeoutExpired:
            return f"Error: Code execution timed out after {timeout} seconds"
        except Exception as e:
            return f"Error: Code execution failed: {str(e)}"

    def destroy(self, **kwargs) -> None:
        """
        Clean up resources used by the executor.
        """
        pass
