import time
import logging

from typing import Any, Dict, Optional, List
from vllm import LLM
from vllm import SamplingParams


class LLMProcessor:
    def __init__(
        self,
        model_path: str,
        mode: str = "chat",
        tokenizer: str = None,
        quantization: Optional[str] = None,
        max_model_len: int = 512,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "auto",
        sampling_params_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LLMProcessor

        Args:
            model_path (str): Hugging Face model path.
            mode (str): Processing mode ('chat' or 'generation').
            quantization (Optional[str]): Quantization method (e.g., 'awq', 'gptq', 'bitsandbytes', None).
            tensor_parallel_size (int): Tensor parallel processing size.
            gpu_memory_utilization (float): GPU memory utilization limit.
            dtype (str): Model data type ('auto', 'bfloat16', 'float16').
            sampling_params_dict (Optional[Dict[str, Any]]): vLLM SamplingParams settings to use in Generation mode.
        """

        if mode.lower() not in ["chat", "generation"]:
            raise ValueError("Mode must be either 'chat' or 'generation'.")

        self.model_path = model_path
        # self.tokenizer_path = self._find_path(tokenizer) if tokenizer else None
        self.mode = mode
        self.quantization = quantization
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.sampling_params_dict = sampling_params_dict
        self.gpu_memory_utilization = gpu_memory_utilization

        logging.info(f"Initializing vLLM server for model: {model_path}...")
        start_time = time.time()
        try:
            self.instance = LLM(
                model=self.model_path,
                tokenizer=tokenizer,
                max_model_len=self.max_model_len,
                tensor_parallel_size=self.tensor_parallel_size,
                quantization=self.quantization,
                dtype=self.dtype,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
            end_time = time.time()
            logging.info(
                f"Model loaded successfully in {end_time - start_time:.2f} seconds."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_path} with error: {e}")

    def interactive(self, use_chat_template: bool = True):
        """
        Start an interactive chat session with the LLM.

        Args:
            use_chat_template (bool): Whether to use chat template for conversation history.
        """
        if self.mode != "chat":
            raise ValueError("Interactive mode is only available in 'chat' mode.")

        print("\n--- Interactive Chat Session ---")
        print("Type 'quit' or 'exit' to end the session.")

        conversation_history = []

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit"]:
                    print("Exiting chat session.")
                    break
                if not user_input.strip():
                    continue

                if use_chat_template and hasattr(
                    self.instance.get_tokenizer(),
                    "apply_chat_template",
                ):
                    tokenizer = self.instance.get_tokenizer()
                    conversation_history.append({"role": "user", "content": user_input})
                    prompt_text = tokenizer.apply_chat_template(
                        conversation_history,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    prompt_text = user_input

                print("LLM: Thinking...", end="\r")
                start_time = time.time()
                outputs = self.instance.generate(
                    [prompt_text],
                    self.sampling_params_dict,
                    use_tqdm=False,
                )
                end_time = time.time()
                print("LLM:          ", end="\r")

                generated_text = outputs[0].outputs[0].text.strip()
                print(f"LLM: {generated_text}")
                print(f"(Generated in {end_time - start_time:.2f}s)")

                if use_chat_template and hasattr(
                    self.instance.get_tokenizer(), "apply_chat_template"
                ):
                    conversation_history.append(
                        {"role": "assistant", "content": generated_text}
                    )

            except KeyboardInterrupt:
                print("\nExiting chat session.")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")

    def generate(self, batch_prompts: List[str]) -> List[List[str]]:
        """
        Process a batch of texts and return the outputs from the LLM.

        Args:
            batch_texts (List[str]): List of text strings to process (mini-batch).

        Returns:
            List[List[str]]: List of generated results for each text, where each inner list contains n outputs.
        """
        if not batch_prompts:
            return []

        start_time = time.time()

        if self.mode == "generation":
            outputs = self.instance.generate(
                batch_prompts,
                self.sampling_params_dict,
                use_tqdm=False,
            )

            # Extract all outputs for each prompt
            generated_texts = []
            for output in outputs:
                prompt_outputs = [
                    output.outputs[i].text.strip() for i in range(len(output.outputs))
                ]
                generated_texts.append(prompt_outputs)

            logging.info(f"Processed in {time.time() - start_time:.2f} seconds.")
            return generated_texts

        else:
            raise RuntimeError("Invalid mode encountered during processing.")
