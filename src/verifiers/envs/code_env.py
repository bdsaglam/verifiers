import logging
from typing import Any, Dict, List

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.codex.models import CodeExecutor
from verifiers.datasets.utils import prepare_dataset_for_env
from verifiers.envs.multistep_env import CompletionOutput, MultiStepEnv, State
from verifiers.parsers import XMLParser
from verifiers.prompts import CODE_FEW_SHOT, CODE_PROMPT
from verifiers.rubrics import CodeRubric

log = logging.getLogger(__name__)


class CodeEnv(MultiStepEnv):
    def __init__(
        self,
        code_executor: CodeExecutor,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        system_prompt: str = CODE_PROMPT,
        few_shot: List[Dict[str, str]] = CODE_FEW_SHOT[0],
        additional_sampling_args={},
        mask_env_response: bool = True,
        max_steps: int = 5,
        **kwargs,
    ):
        # Add stop tokens from the tokenizer
        self.special_stop_tokens = [
            "</code>",
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

        self.code_executor = code_executor

        self.system_prompt = system_prompt
        self.few_shot = few_shot

        self.dataset = prepare_dataset_for_env(
            dataset=train_dataset,
            system_prompt=system_prompt,
            few_shot=few_shot,
            fewshot_prob=1.0,
        )
        self.eval_dataset = (
            prepare_dataset_for_env(
                dataset=eval_dataset,
                system_prompt=system_prompt,
                few_shot=few_shot,
                fewshot_prob=1.0,
            )
            if eval_dataset
            else None
        )
        self.max_steps = max_steps
        self.llm_parser = XMLParser(fields=["reasoning", ("code", "answer")])
        self.env_parser = XMLParser(fields=["output"])
        self.rubric = CodeRubric(parser=self.llm_parser, env_parser=self.env_parser)

    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset

    def get_eval_dataset(self, n: int = -1, **kwargs: Any) -> Dataset | None:
        if n > 0 and self.eval_dataset is not None:
            return self.eval_dataset.shuffle().select(range(n))  # type: ignore
        return self.eval_dataset

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def _get_step_count(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of code executions in the message history, excluding few-shot examples."""
        step_count = 0

        # Skip messages that are part of few-shot examples
        # We need to determine where the actual conversation starts
        # System message + few-shot examples + user query = start of actual conversation
        conversation_start = 1  # Start after system message
        if self.few_shot:
            # Account for all few-shot messages
            conversation_start += len(self.few_shot)

        # Only count code executions from the actual conversation
        for message in messages[conversation_start:]:
            if message.get("role") == "assistant":
                try:
                    parsed = self.llm_parser.parse(message["content"])
                    if hasattr(parsed, "code") and parsed.code is not None:
                        step_count += 1
                except Exception:
                    pass
        return step_count

    def is_completed(self, state: State, completion_output: CompletionOutput, **kwargs: Any) -> bool:
        # Check if we've hit max steps by counting code executions in the message history
        messages = state["messages"]
        step_count = self._get_step_count(messages)
        if step_count >= self.max_steps:
            return True

        # Check if the completion output stopped because of a code execution
        if completion_output.stop_reason not in self.special_stop_tokens:
            return True

        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid answer field (not just None from failed parsing)
            return hasattr(parsed, "answer") and parsed.answer is not None
        except Exception:
            return False

    def run_code(self, code: str) -> str:
        try:
            return self.code_executor.execute(code, timeout=10)
        except Exception as e:
            log.warning(f"Code execution failed: {e}")
            return "Error: Code execution failed"

    def env_response(self, state: State, **kwargs: Any) -> Dict[str, str]:
        messages = state["messages"]
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid code field (not just None from failed parsing)
            if code := getattr(parsed, "code", None):
                output = self.run_code(code)
                if output.strip():
                    return {
                        "role": "tool",
                        "content": self.env_parser.format(output=output),
                    }
                else:
                    log.warning("Code execution returned empty output.")
                    return {
                        "role": "tool",
                        "content": "Error: Code execution returned empty output.",
                    }
        except Exception:
            pass
        return {
            "role": "user",
            "content": "Error: Code not found or invalid XML format. Please ensure correct formatting.",
        }
