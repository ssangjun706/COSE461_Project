import time
import torch
import logging
import src.utils as utils

from torch import dtype
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


class LLMLoader:
    def __init__(
        self,
        model_name: str,
        num_sequences: int = 1,
        max_tokens: int = 512,
        torch_dtype: dtype | str | Any = "auto",
        quantization_config: BitsAndBytesConfig | Any = None,
    ):
        """
        LLMLoader 클래스의 초기화 메서드입니다.
        Args:
            model_name (str): 로드할 모델의 이름입니다.
            num_sequences (int, optional): 생성할 시퀀스의 개수입니다. 기본값은 1입니다.
            max_tokens (int, optional): 생성할 최대 토큰 수입니다. 기본값은 512입니다.
            torch_dtype (dtype | str | Any, optional): 모델의 데이터 타입을 지정합니다. 기본값은 "auto"입니다.
            quantization_config (Any | None, optional): 양자화 설정을 위한 구성 객체입니다. 기본값은 None입니다.
        Attributes:
            model (AutoModelForCausalLM): 사전 학습된 언어 모델 객체입니다.
            tokenizer (AutoTokenizer): 모델과 함께 사용할 토크나이저 객체입니다.
            device (torch.device): 모델이 로드된 디바이스 정보입니다.
            sampling_params (dict): 텍스트 생성 시 사용할 샘플링 매개변수입니다.
        Notes:
            - 모델과 토크나이저는 `transformers` 라이브러리의 `from_pretrained` 메서드를 사용하여 로드됩니다.
            - 모델 로드 시 `torch.compile`을 사용하여 컴파일됩니다.
            - 샘플링 매개변수는 기본값으로 설정되어 있으며, 필요에 따라 수정할 수 있습니다.
        """

        self.model = None
        self.tokenizer = None

        try:
            _model_path = utils.find_model_path(model_name)
        except:
            available_models = utils.list_models()
            logging.error(
                "Model {} not found in cache.\n Available models are: \n- {}".format(
                    model_name, "\n- ".join(available_models)
                )
            )
            exit(1)

        self.tokenizer = AutoTokenizer.from_pretrained(
            _model_path,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        logging.info(f"Loading model: {model_name}...")
        _start = time.time()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        self.model = torch.compile(self.model)
        logging.info(f"Model loaded successfully in {(time.time() - _start):.2f}s")

        self.device = self.model.device
        self.sampling_params = {
            "num_return_sequences": num_sequences,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0,
            "max_new_tokens": max_tokens,
            "do_sample": True,
        }

    def generate(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        generate 함수는 주어진 입력 텐서를 기반으로 문장을 생성하는 기능을 제공합니다.
        Args:
            inputs (torch.Tensor): 입력 데이터로 텐서 형식의 데이터를 받을 수 있습니다.
        Returns:
            torch.Tensor: 생성된 텍스트의 토큰화된 출력 텐서를 반환합니다.
        동작:
            1. 입력 데이터를 토크나이저를 사용하여 텐서 형식으로 변환하고, 패딩을 추가합니다.
            2. 변환된 입력 데이터를 모델에 전달하여 텍스트를 생성합니다.
            3. 이 함수의 출력은 decode 함수를 이용하여 문장으로 변환 가능합니다.
        """

        _start = time.time()
        _inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        self.input_length = _inputs["input_ids"].shape[1]
        _outputs = self.model.generate(**_inputs, **self.sampling_params)
        logging.info(f"Generation completed in {(time.time() - _start):.2f}s")
        return _outputs

    def decode(self, outputs: torch.Tensor):
        """
        주어진 텐서를 디코딩하여 텍스트로 변환합니다.
        Args:
            outputs (torch.Tensor): 디코딩할 텐서.
                텐서는 모델의 출력으로, 입력 길이 이후의 토큰들을 포함합니다.
        Returns:
            List[str]: 디코딩된 텍스트의 리스트.
                특수 토큰은 제거됩니다.
        """

        decoded_outputs = self.tokenizer.batch_decode(
            outputs[:, self.input_length :],
            skip_special_tokens=True,
        )
        return decoded_outputs

    def collate_fn(
        self,
        inputs: str,
        label: str,
        label_values: str | list[str] | Any,
    ) -> list[str]:
        """
        collate_fn 함수는 입력 데이터를 기반으로 프롬프트를 생성하는 기능을 제공합니다.
        이 함수는 LLM과의 상호작용을 위한 프롬프트를 구성하는 데 사용됩니다.
        Args:
            inputs (str): X 데이터로 사용할 입력 정보의 리스트입니다.
            label (str): 예측해야 할 Y-label의 이름입니다.
            label_values (str | list[str] | Any): Y-label이 가질 수 있는 가능한 값들의 리스트 또는 문자열입니다.
        Returns:
            list[str]: 각 입력 데이터에 대해 생성된 프롬프트의 리스트입니다.

        주요 기능:
        1. 입력 데이터를 기반으로 LLM이 예측 작업을 수행할 수 있도록 프롬프트를 생성합니다.
        2. 생성된 프롬프트는 X 데이터, Y-label, 그리고 출력 형식 제약 조건을 포함합니다.
        """
        label_values = "or ".join(label_values)

        def _build_meta_prompt(X_data: Any) -> str:
            return f"""You are an AI assistant specialized in crafting **diverse and creative prompts** for other LLM models. Your current task is to generate a **potential prediction prompt** that will be given to another LLM (a predictor).

The predictor LLM's ultimate goal is to analyze input features (X data) and predict a target variable (Y-label), outputting **only** one of two specific values: {label_values}.

**Your main objective now is to generate *one example* of a prediction prompt, focusing on exploring different ways to structure the request.** 

**Crucially, *how* you structure this prompt is up to you.** You could:
*   Simply list the data and ask for the prediction.
*   Include explanations of what the feature labels mean.
*   Phrase it as a question about the person's profile.
*   **Or try other structures entirely!**

The key is **variety and experimentation** in the prompt format itself, while still providing the necessary information and output constraints *to the predictor*.

**Important!** The prompt should 1) Contain all (or part) of the X data (should not be modified), 2) Mention the Y-label to predict ('{label}'), and 3) Somehow convey the requirement for the predictor's final output format ({label_values} only).

Input Information for you to use:

X data: {X_data}
Y-label to predict: '{label}'
Required final output from the predictor: {label_values} (strictly one of these)

Now, generate **one unique prediction prompt** based on these instructions. Focus on creating a potentially effective but possibly unconventional structure.
"""

        return list(map(_build_meta_prompt, inputs))
