from __future__ import annotations

import torch
from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CausalLMExtractor:
    def __init__(
        self,
        model_name: str,
        *,
        hf_token: str | None = None,
        enable_thinking: bool = True,
    ) -> None:
        self.device = default_device()
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "token": hf_token,
            "torch_dtype": "auto",
            "trust_remote_code": True,
        }
        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).eval()
        if self.device.type != "cuda":
            self.model.to(self.device)

    def _format_prompt(self, prompt: str, *, generation: bool) -> str:
        if self.tokenizer.chat_template is None:
            return f"User: {prompt}" + ("\nAssistant:" if generation else "")

        messages = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": generation}
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=self.enable_thinking,
                **kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _tokenize(self, prompt: str, *, generation: bool):
        text = self._format_prompt(prompt, generation=generation)
        return self.tokenizer([text], return_tensors="pt").to(self.device)

    def _last_hidden_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                output_hidden_states=True,
                use_cache=False,
            )
        return outputs.hidden_states[-1]

    def embed_prompt_vector(self, prompt: str) -> tuple[torch.Tensor, str]:
        inputs = self._tokenize(prompt, generation=False)
        hidden = self._last_hidden_state(inputs["input_ids"])
        emb = hidden.mean(dim=1)
        text = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)[0]
        return emb.cpu(), text

    def generate_token_features(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float | None,
        top_p: float | None,
    ) -> tuple[torch.Tensor, str]:
        inputs = self._tokenize(prompt, generation=True)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            generated = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        hidden = self._last_hidden_state(generated.sequences)
        emb = hidden[:, prompt_len:, :].squeeze(0)
        text = self.tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)[0]
        return emb.cpu(), text


def load_vision_model(model_name: str, hf_token: str | None = None):
    device = default_device()
    model = AutoModel.from_pretrained(model_name, token=hf_token, trust_remote_code=True).eval().to(device)
    processor = AutoImageProcessor.from_pretrained(model_name, token=hf_token)
    return model, processor, device
