"""Qwen3-VL generation used for target captions and vision-side captions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def _sanitize(text: str) -> str:
    return str(text).replace("<image>", "[IMAGE]").replace("<IMAGE>", "[IMAGE]")


class Qwen3VLEngine:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        max_num_seqs: int = 8,
        dtype: str = "bfloat16",
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "Caption generation needs vLLM. Install it in this environment, then rerun the script."
            ) from exc

        self.model_path = str(model_path)
        kwargs: Dict[str, Any] = {
            "model": self.model_path,
            "trust_remote_code": True,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "tensor_parallel_size": 1,
            "limit_mm_per_prompt": {"image": 1, "video": 0},
        }
        try:
            self.model = LLM(**kwargs, enable_prefix_caching=True)
        except TypeError:
            self.model = LLM(**kwargs)
        self.tokenizer = self.model.get_tokenizer()

    @staticmethod
    def load_image(path: Union[str, Path], max_side: int = 1280):
        from PIL import Image

        img = Image.open(path).convert("RGB")
        width, height = img.size
        long_side = max(width, height)
        if long_side > max_side:
            scale = max_side / float(long_side)
            img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BICUBIC)
        return img

    def build_prompt(self, prompt: str, num_images: int = 0) -> str:
        text = _sanitize(prompt)
        if num_images > 0:
            content: List[Dict[str, Any]] = [{"type": "image"} for _ in range(num_images)]
            content.append({"type": "text", "text": text})
        else:
            content = [{"type": "text", "text": text}]
        messages = [{"role": "user", "content": content}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        image_paths: Optional[Sequence[Optional[Union[str, Path]]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> List[str]:
        from vllm import SamplingParams

        if image_paths is None:
            image_paths = [None] * len(prompts)
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts must have the same length")

        requests = []
        for prompt, img_path in zip(prompts, image_paths):
            req: Dict[str, Any] = {"prompt": self.build_prompt(prompt, num_images=1 if img_path else 0)}
            if img_path:
                req["multi_modal_data"] = {"image": self.load_image(img_path)}
            requests.append(req)
        sampling = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens)
        outputs = self.model.generate(requests, sampling)
        return [item.outputs[0].text for item in outputs]
