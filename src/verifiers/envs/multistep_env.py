from abc import abstractmethod
from typing import Any, Dict, List, Sequence, TypedDict, Union

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.environment import Environment
from verifiers.models import Input, Message

from ..imports import LLM, CompletionOutput, SamplingParams  # type: ignore


class State(TypedDict):
    completed: bool
    input: Input
    messages: List[Message]
    n_prompt_messages: int
    prompt_ids: List[int]
    completion_ids: List[int]
    completion_mask: List[int]


class MultiStepEnv(Environment):
    def __init__(
        self,
        sampling_args: Dict[str, Any] = {},
        mask_env_response: bool = True,
        max_steps: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1,
        }
        self.sampling_args.update(sampling_args)
        self.env_mask = 0 if mask_env_response else 1
        self.max_steps = max_steps

    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    @abstractmethod
    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        pass

    @abstractmethod
    def is_completed(self, state: State, completion_output: CompletionOutput, **kwargs: Any) -> bool:
        pass

    def eval(self, model: Union[str, LLM], batch_size: int = 10, **kwargs: Any):
        if self.eval_dataset is None:
            self.eval_dataset = self.get_eval_dataset()

        rewards = []
        return self.eval_dataset, rewards

    @abstractmethod
    def env_response(
        self,
        state: State,
        **kwargs: Any,
    ) -> Dict[str, str]:
        pass

    def generate(
        self,
        inputs: List[Input],
        llm: LLM,
        sampling_params: SamplingParams,
        **kwargs: Any,
    ) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Message]]]:
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        # initialize state variables
        all_completed = False
        states: List[State] = [
            {
                "completed": False,
                "input": input,
                "messages": input["prompt"],
                "n_prompt_messages": len(input["prompt"]),
                "prompt_ids": [],
                "completion_ids": [],
                "completion_mask": [],
            }
            for input in inputs
        ]

        # main loop
        while not all_completed:
            states = self._step(states, llm, custom_sp)
            all_completed = all(state["completed"] for state in states)

        completion_messages = [s["messages"][s["n_prompt_messages"] :] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        output = {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask,
        }
        return output

    def _step(
        self,
        states: List[State],
        llm: LLM,
        sampling_params: SamplingParams,
    ) -> List[State]:
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        messages_to_step = [states[i]["messages"] for i in live_indices]
        llm_responses = llm.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=False)  # type: ignore

        for i, j in enumerate(live_indices):
            if len(states[j]["prompt_ids"]) == 0:
                states[j]["prompt_ids"] = llm_responses[i].prompt_token_ids
            states[j]["messages"].append({"role": "assistant", "content": llm_responses[i].outputs[0].text})

            # get token lengths of env response and new completion
            total_prev_len = len(states[j]["prompt_ids"]) + len(states[j]["completion_ids"])
            env_response_len = len(list(llm_responses[i].prompt_token_ids)) - total_prev_len  # type: ignore
            new_completion_len = len(llm_responses[i].outputs[0].token_ids)

            # update completion masks
            states[j]["completion_mask"].extend([self.env_mask] * env_response_len)
            states[j]["completion_mask"].extend([1] * new_completion_len)

            # update completion ids
            states[j]["completion_ids"] = list(llm_responses[i].prompt_token_ids)  # type: ignore
            states[j]["completion_ids"].extend(list(llm_responses[i].outputs[0].token_ids))
            states[j]["completion_ids"] = states[j]["completion_ids"][len(states[j]["prompt_ids"]) :]

            if (
                self._is_reached_max_steps(states[j])
                or self.is_completed(states[j], completion_output=llm_responses[i].outputs[0])
                or len(states[j]["completion_ids"]) > sampling_params.max_tokens
            ):  # type: ignore
                states[j]["completed"] = True
                states[j]["completion_ids"] = states[j]["completion_ids"][: sampling_params.max_tokens]
                states[j]["completion_mask"] = states[j]["completion_mask"][: sampling_params.max_tokens]
            else:
                states[j]["messages"].append(self.env_response(states[j]))

            assert len(states[j]["completion_mask"]) == len(states[j]["completion_ids"])

        return states

    def _is_reached_max_steps(self, state: State) -> bool:
        messages = state["messages"]
        n_prompt_messages = state["n_prompt_messages"]
        step_count = (len(messages) - n_prompt_messages) // 2
        return step_count >= self.max_steps
