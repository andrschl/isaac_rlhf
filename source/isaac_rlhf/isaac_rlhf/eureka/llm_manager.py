# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0
from isaac_rlhf.eureka.prompts import PREFERENCE_SYSTEM_PROMPT, PREFERENCE_USER_PROMPT
import os
import re

import traceback
import openai
import logging
from isaac_rlhf.utils.rlhf_utils import load_tensorboard_logs
import numpy as np
import ast


class LLMManager:
    def __init__(
        self,
        gpt_model: str = "deepseek/deepseek-r1-0528:free",
        system_prompt: str = PREFERENCE_SYSTEM_PROMPT,
    ):
        """Initialize the LLMManager

        Args:
            gpt_model: The model to use for the LLM API
            num_suggestions: The number of independent suggestions to generate
            temperature: The temperature to use for the LLM API
            system_prompt: The system prompt to provide to the LLM API
        """

        self._gpt_model = gpt_model
        self._temperature = 1
        # self._prompts = [{"role": "system", "content": system_prompt}]
        self._system_prompt = {"role": "system", "content": system_prompt}
        self._prompts = [self._system_prompt]
        api_key = os.getenv("OPENROUTER_API_KEY")
        print(f"Using OpenRouter API key: {api_key}")
        self._client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self._total_tokens = 0
        self._total_query_tokens = 0
        self._total_response_tokens = 0
        self._input_token_price = 0.27
        self._output_token_price = 1.10
        logging.info(f"LLMManager initialized with model: {gpt_model}")
        self._success_metric_to_win = 1.0  # default success metric to win

    # appends a string to an already existing system prompt
    def append_to_system_prompt(self, additional_prompt: str):
        assert self._prompts[0]["role"] == "system"
        self._prompts[0]["content"] += additional_prompt

    # granular control over self._prompts
    # in the main codeblock, make sure
    # appends a new system prompt to an empty list
    def append_system_prompt(self, prompt: str):
        assert not self._prompts
        self._prompts.append({"role": "system", "content": prompt})

    # appends a new user prompt
    def append_user_prompt(self, prompt: str):
        self._prompts.append({"role": "user", "content": prompt})

    # appends a new assistant prompt
    def append_assistant_prompt(self, prompt: str):
        self._prompts.append({"role": "assistant", "content": prompt})

    def clear_prompts(self):
        """Clear the prompts list"""
        self._prompts = []

    # call llm only with system + a single user prompt
    # still saves conversation history in self._prompts
    def call_llm_single(self, user_prompt, max_retries=3):
        logging.info(f"Calling LLM single")
        attempt = 0
        while attempt < max_retries:
            try:
                responses = self._client.chat.completions.create(
                    model=self._gpt_model,
                    messages=[
                        self._system_prompt,
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self._temperature,
                )

                raw_output = responses.choices[0].message.content

                # this catches raw_output that is empty string or None
                if not raw_output:
                    logging.error(
                        f"LLM call failed with empty response, finish reason: {responses.choices[0].finish_reason}"
                    )
                    raise RuntimeError("LLM call failed with empty response")

                preference_string = self.extract_preference_from_response(raw_output)
                if preference_string:
                    self._prompts.append({"role": "user", "content": user_prompt})
                    self._prompts.append({"role": "assistant", "content": raw_output})
                    # preference_string is either "policy_0" or "policy_1"
                    return preference_string, raw_output
                else:
                    print(
                        f"response was okay but failed to extract preference at {attempt + 1}. Retrying..."
                    )
                    logging.error(
                        f"LLM response was non empty but failed to extract preference at attempt {attempt + 1}.\n raw_output: \n{raw_output}\n Retrying..."
                    )
                    attempt += 1
            except Exception as e:
                print(f"[ERROR] LLM call failed on attempt {attempt + 1}: {e}")
                traceback.print_exc()
                logging.error(
                    f"LLM call failed on attempt {attempt + 1}: {traceback.format_exc()}"
                )
                attempt += 1
        logging.error(
            "LLM failed to provide valid weight strings after multiple attempts."
        )
        raise RuntimeError(
            "LLM failed to provide valid weight strings after multiple attempts."
        )

    # no history, doesn't change self._prompts, just calling with a user prompt
    def call_llm_for_summary(self, user_prompt, max_retries=3) -> str:
        logging.info(f"Calling LLM for summary")
        attempt = 0
        while attempt < max_retries:
            try:
                responses = self._client.chat.completions.create(
                    model=self._gpt_model,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=self._temperature,
                    n=self._num_suggestions,
                )

                raw_output = responses.choices[0].message.content

                # this catches raw_output that is empty string or None
                if not raw_output:
                    logging.error(
                        f"LLM call failed with empty response, finish reason: {responses.choices[0].finish_reason}"
                    )
                    raise RuntimeError("LLM call failed with empty response")
                # Success!
                self._total_tokens += responses.usage.total_tokens
                self._total_query_tokens += responses.usage.prompt_tokens
                self._total_response_tokens += responses.usage.completion_tokens
                return raw_output

            except Exception as e:
                print(f"[ERROR] LLM call failed on attempt {attempt + 1}: {e}")
                traceback.print_exc()
                logging.error(
                    f"LLM call failed on attempt {attempt + 1}: {traceback.format_exc()}"
                )
                attempt += 1
        logging.error(
            "LLM failed to provide valid weight strings after multiple attempts."
        )
        raise RuntimeError(
            "LLM failed to provide valid weight strings after multiple attempts."
        )

    def get_or_generate_summary(
        self,
        summary_type: str,
        identifier: str,
        prompt: str,
        code_string: str,
        eureka_root_dir: str,
        use_cache: bool = True,
    ):
        """
        Get or generate a summary for context code, PPO code, or success metric code.
        summaries are saved in EUREKA_ROOT_DIR/summary.

        Args:
            summary_type (str): One of "context_code", "ppo_code", or "success_metric_code".
            identifier (str): Either rl_task_type (for context/success metric) or rl_library (for PPO).
            prompt (str): Instructional prompt to prepend to code_string.
            code_string (str): The code to summarize.
            eureka_root_dir (str): Path to the Eureka root directory.
            override (bool): If True, regenerate and overwrite the summary even if it exists.

        Returns:
            str: The summary string.
        """
        os.makedirs(os.path.join(eureka_root_dir, "summary"), exist_ok=True)
        filename = f"{summary_type}_summary_{identifier}.txt"
        summary_path = os.path.join(eureka_root_dir, "summary", filename)

        if use_cache and os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                cached = f.read().strip()
                if cached:
                    logging.info(f"Using cached summary from {summary_path}")
                    return cached

        # Otherwise, generate new summary via LLM
        logging.info(f"Generating new {filename}")
        llm_input = f"{prompt}\n{code_string}"
        summary = self.call_llm_for_summary(llm_input)
        with open(summary_path, "w") as f:
            f.write(summary)
            logging.info(f"Saved new {filename}")
        return summary

    def extract_preference_from_response(self, response: str):
        match = re.search(r"preference:\s*(policy_[01])", response)
        preferred_policy = match.group(1)
        return preferred_policy

    def extract_relevant_metrics(self, log_dir) -> list[tuple[str, list]]:
        data = load_tensorboard_logs(log_dir)

        ordered_patterns = [
            r".*Eureka/success_metric$",
            r".*Eureka/.+",
            r".*Episode_Reward/.+",
        ]
        filtered_data = []
        seen = set()

        for pattern in ordered_patterns:
            for key in sorted(data.keys()):
                if key not in seen and re.match(pattern, key):
                    filtered_data.append((key, data[key]))
                    seen.add(key)

        return filtered_data

    def get_final_success_metric(self, log_dir: str) -> float:
        data = load_tensorboard_logs(log_dir)
        for metric_name, metric_data in data.items():
            if "success_metric" in metric_name:
                # Return the last value of the success metric
                return metric_data[-1]
        raise KeyError(
            "Could not find a key containing 'success_metric' in TensorBoard logs."
        )

    def get_text_summary_of_training(
        self, log_dir: str, reward_weight_string, feedback_subsampling: int = 10
    ) -> tuple[str, float, float]:
        # filtered data from tensorboard logs
        # data is a list of tuples (key, value), in order
        data = self.extract_relevant_metrics(log_dir=log_dir)

        success_metric_max = None
        reward_weight_dict = ast.literal_eval(reward_weight_string)

        def find_actual_iterations(data, suffix):
            """Find the correct key in `data` that ends with the given suffix."""
            for key, value in data:
                if key.endswith(suffix):
                    return len(value)
            raise KeyError(
                f"Could not find a key ending with '{suffix}' in TensorBoard logs."
            )

        actual_training_iterations = find_actual_iterations(
            data, "Eureka/success_metric"
        )
        adaptive_feedback_subsampling = (
            actual_training_iterations // feedback_subsampling
        )
        total_feed_back_string = ""
        # just give everything, hope that llm can figure it out
        for metric_name, metric_data in data:
            # find weight for this metric and divide, so we somehow have unweighted feature values
            matched_weight = None
            for term_name, weight in reward_weight_dict.items():
                if term_name in metric_name:
                    matched_weight = weight
                    break
            # Default weight = 1.0 if no match found
            if matched_weight is None:
                matched_weight = 1.0
            if matched_weight != 0:
                metric_data = [value / matched_weight for value in metric_data]

            if len(metric_data) > 2:
                metric_data = metric_data[2:]
            metric_min = min(metric_data)
            metric_max = max(metric_data)
            metric_mean = sum(metric_data) / len(metric_data)
            # Best metric is the one closest to the target
            if "Eureka/success_metric" in metric_name:
                metric_best = metric_data[
                    np.abs(np.array(metric_data) - self._success_metric_to_win).argmin()
                ]
                success_metric_max = metric_best
            data_string = [
                f"{data:.2f}" for data in metric_data[::adaptive_feedback_subsampling]
            ]
            feedback_string = (
                f"{metric_name}: {data_string}, Min: {metric_min:.2f}, Max: {metric_max:.2f}, Mean:"
                f" {metric_mean:.2f} \n"
            )
            total_feed_back_string += feedback_string

        total_feed_back_string += f"\nThe desired Eureka/success_metric to win is: 1\n"
        total_feed_back_string += f"Each metric was sampled at every {adaptive_feedback_subsampling} learning iterations.\n"
        return total_feed_back_string, success_metric_max, 0

    def query_preference(self, summary0, summary1) -> bool:
        user_prompt = PREFERENCE_USER_PROMPT.format(
            metrics_policy_0=summary0,
            metrics_policy_1=summary1,
        )
        answer, raw_output = self.call_llm_single(user_prompt)
        if answer == "policy_0":
            return 1
        elif answer == "policy_1":
            return 0
        else:
            raise RuntimeError(
                f"LLM returned unexpected preference: {answer}. Expected 'policy_0' or 'policy_1'."
            )


if __name__ == "__main__":
    # Example usage
    llm_manager = LLMManager()
    response, raw_output = llm_manager.call_llm_single(PREFERENCE_USER_PROMPT)
    print(f"raw_output: {raw_output}")
