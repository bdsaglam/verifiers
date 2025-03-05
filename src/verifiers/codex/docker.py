import json
import time
from pathlib import Path

import docker
import requests
from websocket import create_connection

from verifiers.codex.models import CodeExecutor


class DockerExecutor(CodeExecutor):
    """
    Executes Python code using Jupyter Kernel Gateway in a Docker container.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8888,
    ):
        """
        Initialize the Docker-based Jupyter Kernel Gateway executor.
        """
        self.host = host
        self.port = port
        self.container = None
        self.kernel_id = None
        self.ws = None
        self.base_url = None

        # Initialize Docker
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            raise RuntimeError("Could not connect to Docker daemon: make sure Docker is running.") from e

        # Build and start container
        try:
            dockerfile_path = Path(__file__).parent / "Dockerfile"
            if not dockerfile_path.exists():
                with open(dockerfile_path, "w") as f:
                    f.write(
                        """FROM python:3.12-slim

RUN pip install jupyter_kernel_gateway requests numpy pandas
RUN pip install jupyter_client notebook

EXPOSE {port}
CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip='0.0.0.0'", "--KernelGatewayApp.port={port}", "--KernelGatewayApp.allow_origin='*'"]
""".format(port=self.port)
                    )
            _, build_logs = self.client.images.build(
                path=str(dockerfile_path.parent), dockerfile=str(dockerfile_path), tag="jupyter-kernel"
            )

            self.container = self.client.containers.run(
                "jupyter-kernel", ports={f"{self.port}/tcp": (host, port)}, detach=True
            )

            retries = 0
            while self.container.status != "running" and retries < 5:
                time.sleep(1)
                self.container.reload()
                retries += 1

            self.base_url = f"http://{host}:{port}"

            # Create new kernel via HTTP
            r = requests.post(f"{self.base_url}/api/kernels")
            if r.status_code != 201:
                error_details = {
                    "status_code": r.status_code,
                    "headers": dict(r.headers),
                    "url": r.url,
                    "body": r.text,
                }
                raise RuntimeError(f"Failed to create kernel: {error_details}")

            self.kernel_id = r.json()["id"]
            ws_url = f"ws://{host}:{port}/api/kernels/{self.kernel_id}/channels"
            self.ws = create_connection(ws_url)

        except Exception as e:
            self.destroy()
            raise RuntimeError(f"Failed to initialize Jupyter kernel: {e}") from e

    def _send_execute_request(self, code: str) -> str:
        """Send code execution request to kernel."""
        import uuid

        msg_id = str(uuid.uuid4())
        execute_request = {
            "header": {
                "msg_id": msg_id,
                "username": "anonymous",
                "session": str(uuid.uuid4()),
                "msg_type": "execute_request",
                "version": "5.0",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
            },
        }

        self.ws.send(json.dumps(execute_request))
        return msg_id

    def execute(self, code: str, timeout: int = 10, **kwargs) -> str:
        """
        Execute the given code and return the output.
        """
        try:
            msg_id = self._send_execute_request(code)
            outputs = []
            start_time = time.time()

            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Code execution timed out after {timeout} seconds")

                msg = json.loads(self.ws.recv())
                msg_type = msg.get("msg_type", "")
                parent_msg_id = msg.get("parent_header", {}).get("msg_id")

                if parent_msg_id != msg_id:
                    continue

                if msg_type == "stream":
                    outputs.append(msg["content"]["text"])
                elif msg_type == "error":
                    traceback = msg["content"].get("traceback", [])
                    raise RuntimeError("\n".join(traceback))
                elif msg_type == "status" and msg["content"]["execution_state"] == "idle":
                    break

            return "".join(outputs)

        except Exception as e:
            raise RuntimeError(f"Code execution failed: {e}")

    def destroy(self, **kwargs) -> None:
        """
        Clean up resources used by the executor.
        """
        try:
            if self.ws:
                self.ws.close()
            if self.container:
                self.container.stop()
                self.container.remove()
        except Exception as e:
            raise RuntimeError(f"Error during cleanup: {e}")
