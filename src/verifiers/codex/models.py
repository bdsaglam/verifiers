from typing import Protocol


class CodeExecutor(Protocol):
    """
    Protocol defining the interface for code executors.
    """

    def execute(self, code: str, timeout: int = 10, **kwargs) -> str:
        """
        Execute the given code and return the output.
        """
        ...

    def destroy(self, **kwargs) -> None:
        """
        Clean up resources used by the executor.
        This should be called when the executor is no longer needed.
        """
        ...
