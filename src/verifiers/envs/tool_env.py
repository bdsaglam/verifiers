import inspect
import json
from textwrap import dedent
from typing import Any, Callable, Dict, List

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.datasets.utils import prepare_dataset_for_env
from verifiers.envs.multistep_env import CompletionOutput, MultiStepEnv, State
from verifiers.parsers import XMLParser
from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE
from verifiers.rubrics import Rubric
from verifiers.rubrics.format import make_format_reward_func, make_xml_reward_func
from verifiers.rubrics.tool import make_tool_use_reward_func


class ToolEnv(MultiStepEnv):
    def __init__(
        self,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        tools: List[Callable] = [],
        system_prompt: str = DEFAULT_TOOL_PROMPT_TEMPLATE,
        assistant_parser: XMLParser = XMLParser(fields=["think", ("tool", "answer")]),
        env_parser: XMLParser = XMLParser(fields=["result"]),
        rubric: Rubric | None = None,
        few_shot: List[Dict[str, str]] = [],
        few_shot_prob: float = 0.5,
        additional_sampling_args={},
        mask_env_response: bool = True,
        **kwargs,
    ):
        # Add stop tokens from the tokenizer
        self.special_stop_tokens = [
            "</tool>",
            "</answer>",
        ]
        additional_stop_tokens = additional_sampling_args.pop("stop", [])
        stop_tokens = list(
            {
                tokenizer.eos_token,
                tokenizer.pad_token,
                *self.special_stop_tokens,
                *additional_stop_tokens,
            }
        )
        sampling_args = {
            "stop": stop_tokens,
            "include_stop_str_in_output": True,
            **additional_sampling_args,
        }
        super().__init__(
            mask_env_response=mask_env_response,
            sampling_args=sampling_args,
            **kwargs,
        )

        self.system_prompt = system_prompt
        self.few_shot = few_shot

        # Infer schemas from tool functions
        self.tool_schemas = [infer_schema_from_function(tool) for tool in tools]
        self.tools = {tool.__name__: tool for tool in tools}

        # Format the system prompt with tool descriptions
        tool_descriptions = format_tool_descriptions(self.tool_schemas)
        formatted_prompt = system_prompt.format(tool_descriptions=tool_descriptions)

        self.dataset = prepare_dataset_for_env(
            dataset=train_dataset,
            system_prompt=formatted_prompt,
            few_shot=few_shot,
            few_shot_prob=few_shot_prob,
        )
        self.eval_dataset = (
            prepare_dataset_for_env(
                dataset=eval_dataset,
                system_prompt=formatted_prompt,
                few_shot=few_shot,
                few_shot_prob=few_shot_prob,
            )
            if eval_dataset
            else None
        )
        self.assistant_parser = assistant_parser
        self.env_parser = env_parser
        self.rubric = rubric or Rubric(
            reward_funcs=[
                make_xml_reward_func(assistant_parser),
                make_format_reward_func(assistant_parser),
                make_tool_use_reward_func(assistant_parser=self.assistant_parser, env_parser=self.env_parser),
            ],
        )

    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset

    def get_eval_dataset(self, n: int = -1, **kwargs: Any) -> Dataset:
        if n > 0:
            return self.eval_dataset.shuffle().select(range(n))  # type: ignore
        return self.eval_dataset

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def is_completed(self, state: State, completion_output: CompletionOutput, **kwargs: Any) -> bool:
        messages = state["messages"]

        # Check if the completion output stopped because of a tool call
        if completion_output.stop_reason not in self.special_stop_tokens:
            return True

        try:
            parsed = self.assistant_parser.parse(messages[-1]["content"])
            # Check if we got a valid answer field (not just None from failed parsing)
            return getattr(parsed, "answer", None) is not None
        except Exception:
            return False

    def call_tool(self, tool_json: str, run_context: Dict[str, Any]) -> str:
        """Call a tool based on JSON command."""
        try:
            command = json.loads(tool_json)
            if not isinstance(command, dict):
                return "Error: Tool command must be a JSON object"

            tool_name = command.get("name")
            if not tool_name:
                return "Error: Tool command must specify 'name'"

            if tool_name not in self.tools:
                return f"Error: Unknown tool '{tool_name}'"

            # Call the tool function with arguments
            tool_func = self.tools[tool_name]
            tool_args = command.get("args", {})
            if isinstance(tool_args, dict):
                result = tool_func(**tool_args, run_context=run_context)
            else:
                result = tool_func(tool_args, run_context=run_context)

            return str(result)
        except json.JSONDecodeError:
            return "Error: Invalid JSON inside <tool> tags"
        except Exception as e:
            return f"Error: {str(e)}"

    def env_response(self, state: State, **kwargs: Any) -> Dict[str, str]:
        run_context = dict(input=state["input"])
        messages = state["messages"]
        try:
            parsed = self.assistant_parser.parse(messages[-1]["content"])
            # Check if we got a valid tool field (not just None from failed parsing)
            if tool := getattr(parsed, "tool", None):
                result = self.call_tool(tool, run_context=run_context)
                if len(result.strip()) > 0:
                    return {
                        "role": "tool",
                        "content": self.env_parser.format(result=result),
                    }
                else:
                    return {
                        "role": "tool",
                        "content": "Error: Tool execution returned empty output.",
                    }
        except Exception:
            pass
        return {
            "role": "user",
            "content": "Error: Tool command not found or invalid XML format. Please ensure correct formatting.",
        }


def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Infers a tool schema from a function's signature and docstring."""
    sig = inspect.signature(func)
    doc = dedent(inspect.getdoc(func) or "")

    # Parse docstring sections
    doc_parts = doc.split("\n\n")
    description = doc_parts[0].strip()

    # Extract examples if present
    examples = []
    for part in doc_parts:
        part = part.strip()
        if part.startswith("Examples:"):
            examples = [line.strip() for line in part.split("\n")[1:] if line.strip()]

    # Build args schema
    args = {}
    for name, param in sig.parameters.items():
        if name == "kwargs" or name == "run_context":
            continue
        param_doc = ""
        for part in doc_parts:
            if part.strip().startswith("Args:"):
                for line in part.split("\n")[1:]:
                    if line.strip().startswith(f"{name}:"):
                        param_doc = line.strip()[len(name) + 1 :].strip()

        args[name] = {
            "type": str(param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "any"),
            "description": param_doc,
        }
        if param.default != inspect.Parameter.empty:
            args[name]["default"] = param.default

    return {
        "name": func.__name__,
        "description": description,
        "args": args,
        "returns": str(sig.return_annotation.__name__ if sig.return_annotation != inspect.Parameter.empty else "any"),
        "examples": examples,
    }


def format_tool_descriptions(schemas: List[Dict[str, Any]]) -> str:
    """Formats tool schemas into a user-friendly description string."""
    descriptions = []
    for schema in schemas:
        desc = [f"{schema['name']}: {schema['description']}"]

        desc.append("\nArguments:")
        for arg_name, arg_info in schema["args"].items():
            default = f" (default: {arg_info['default']})" if "default" in arg_info else ""
            desc.append(f"  - {arg_name}: {arg_info['description']}{default}")

        if schema["examples"]:
            desc.append("\nExamples:")
            for example in schema["examples"]:
                desc.append(f"  {example}")

        descriptions.append("\n".join(desc))

    return "\n\n".join(descriptions)
