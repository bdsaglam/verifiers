from abc import abstractmethod
from typing import Any, Dict, List, Sequence, TypedDict, Union

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.environment import Environment

from ..imports import LLM, CompletionOutput, SamplingParams  # type: ignore


class State(TypedDict):
    messages: List[Dict[str, str]]
    n_prompt_messages: int
    input: Dict[str, Any]
    prompt_ids: List[int]
    completed: bool
    completion_ids: List[int]


class MultiStepEnv(Environment):
    def __init__(
        self,
        sampling_args: Dict[str, Any] = {},
        mask_env_response: bool = True,
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

    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    @abstractmethod
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        pass

    @abstractmethod
    def is_completed(self, state: State, completion_output: CompletionOutput, **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def env_response(
        self,
        state: State,
        **kwargs: Any,
    ) -> Dict[str, str]:
        pass

    def step(
        self,
        states: List[Dict[str, Any]],
        llm: LLM,
        sampling_params: SamplingParams,
    ) -> List[Dict[str, Any]]:
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
                self.is_completed(states[j], completion_output=llm_responses[i].outputs[0])
                or len(states[j]["completion_ids"]) > sampling_params.max_tokens
            ):  # type: ignore
                states[j]["completed"] = True
                states[j]["completion_ids"] = states[j]["completion_ids"][: sampling_params.max_tokens]
                states[j]["completion_mask"] = states[j]["completion_mask"][: sampling_params.max_tokens]
            else:
                states[j]["messages"].append(self.env_response(states[j]))

            assert len(states[j]["completion_mask"]) == len(states[j]["completion_ids"])

        return states

    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm: LLM,
        sampling_params: SamplingParams,
        inputs: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        # initialize state variables
        all_completed = False
        states = [
            {
                "messages": m,
                "n_prompt_messages": len(m),
                "input": input,
                "prompt_ids": [],
                "completed": False,
                "completion_ids": [],
                "completion_mask": [],
            }
            for m, input in zip(prompts, inputs)
        ]

        # main loop
        while not all_completed:
            states = self.step(states, llm, custom_sp)
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

    def eval(self, model: Union[str, LLM], batch_size: int = 10, **kwargs: Any):
        if self.eval_dataset is None:
            self.eval_dataset = self.get_eval_dataset()

        rewards = []
        return self.eval_dataset, rewards
