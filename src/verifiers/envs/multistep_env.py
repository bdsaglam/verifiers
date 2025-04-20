import logging
from abc import abstractmethod
from typing import Any, Dict, List, Sequence, TypedDict, Union

from datasets import Dataset
from joblib import Parallel, delayed
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.environment import Environment
from verifiers.imports import LLM, CompletionOutput, RequestOutput, SamplingParams
from verifiers.models import Input, Message

logger = logging.getLogger(__name__)


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
        n_jobs: int = 1,  # 1 means no parallelization, -1 means use all processors
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
        self.n_jobs = n_jobs

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
                "metrics": {},
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
        metrics = [s["metrics"] for s in states]

        output = {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask,
            "metrics": metrics,
        }
        return output

    def _step(
        self,
        states: List[State],
        llm: LLM,
        sampling_params: SamplingParams,
    ) -> List[State]:
        live_indices = [i for i, state in enumerate(states) if not state["completed"]]

        # Generate LLM responses for incomplete states
        messages_to_step = [states[idx]["messages"] for idx in live_indices]
        llm_responses = self.act(llm, messages_to_step, sampling_params)

        # Define the processing function with all the state processing logic
        def process_one_state(idx, state, llm_response):
            if len(state["prompt_ids"]) == 0:
                state["prompt_ids"] = llm_response.prompt_token_ids
            state["messages"].append({"role": "assistant", "content": llm_response.outputs[0].text})

            # Update completion mask for env response
            total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
            env_response_len = len(list(llm_response.prompt_token_ids)) - total_prev_len
            state["completion_mask"].extend([self.env_mask] * env_response_len)

            # Update completion mask for new completion
            new_completion_len = len(llm_response.outputs[0].token_ids)
            state["completion_mask"].extend([1] * new_completion_len)

            # update completion ids
            all_ids = list(llm_response.prompt_token_ids) + list(llm_response.outputs[0].token_ids)
            state["completion_ids"] = all_ids[len(state["prompt_ids"]) :]

            is_completed = (
                self._is_reached_max_steps(state)
                or self.is_completed(state, completion_output=llm_response.outputs[0])
                or len(state["completion_ids"]) > sampling_params.max_tokens
            )  # type: ignore

            if is_completed:
                state["completed"] = True
                state["completion_ids"] = state["completion_ids"][: sampling_params.max_tokens]
                state["completion_mask"] = state["completion_mask"][: sampling_params.max_tokens]
            else:
                state["messages"].append(self.env_response(state))

            if len(state["completion_mask"]) != len(state["completion_ids"]):
                # === Logging before assertion ===
                logger.info(f"--- Debugging State {idx} ---")
                logger.info(f"Initial prompt IDs length: {len(state['prompt_ids'])}")
                # Note: completion_ids length *after* update is relevant for the assert
                # Need to calculate the 'before update' length for total_prev_len context if needed elsewhere,
                # but the critical comparison is between the final lengths.
                logger.info(f"Calculated total_prev_len: {total_prev_len}")
                logger.info(f"LLM prompt_token_ids length: {len(list(llm_response.prompt_token_ids))}")  # type: ignore
                logger.info(f"Calculated env_response_len: {env_response_len}")
                logger.info(f"New completion length: {new_completion_len}")
                logger.info(f"Final completion_mask length: {len(state['completion_mask'])}")
                logger.info(f"Final completion_ids length: {len(state['completion_ids'])}")

                if len(state["completion_mask"]) > len(state["completion_ids"]):
                    state["completion_mask"] = state["completion_mask"][: len(state["completion_ids"])]
                else:
                    state["completion_mask"].extend(
                        [1] * (len(state["completion_ids"]) - len(state["completion_mask"]))
                    )

                # raise AssertionError(f"Completion mask and completion ids lengths do not match for state {idx}")

            return idx, state

        if self.n_jobs == 1:
            # Process sequentially
            results = [
                process_one_state(idx, states[idx], llm_resp) for idx, llm_resp in zip(live_indices, llm_responses)
            ]
        else:
            # Process in parallel using joblib
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(process_one_state)(idx, states[idx], llm_resp)
                for idx, llm_resp in zip(live_indices, llm_responses)
            )

        # Update states with results
        for idx, updated_state in results:
            states[idx] = updated_state

        return states

    def _is_reached_max_steps(self, state: State) -> bool:
        messages = state["messages"]
        n_prompt_messages = state["n_prompt_messages"]
        step_count = (len(messages) - n_prompt_messages) // 2
        return step_count >= self.max_steps

    def act(
        self,
        llm: LLM,
        messages_list: List[List[Message]],
        sampling_params: SamplingParams,
    ) -> List[RequestOutput]:
        return llm.chat(messages_list, sampling_params=sampling_params, use_tqdm=False)
