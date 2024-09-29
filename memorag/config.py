import torch
from typing import Optional
from dataclasses import dataclass
from transformers.utils import logging

logger = logging.get_logger(__name__)

CONFIG_PATTERMS = {
    "lite": {},
    "full": {},
}


@dataclass
class MemoRAGConfig:
    use_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gen_model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    ret_model_name_or_path: str = "BAAI/bge-m3"
    customized_gen_model = None
    ret_hit: int = 3
    retrieval_chunk_size: int = 512
    cache_dir: Optional[str] = None
    access_token: Optional[str] = None
    load_in_4bit: bool = False
    enable_flash_attn: bool = True
    beacon_ratio: int = 4

    corups_contents: Optional[list[str]] = None

    def __post_init__(self):
        # We validate the config here
        is_on_cpu = self.use_device == "cpu"
        self.corups_contents = self.corups_contents or []
        if is_on_cpu:
            # TODO load in 4bit seems only works for GPU
            self.load_in_4bit = False
            self.enable_flash_attn = False
            logger.warning(
                "Using CPU, load_in_4bit and enable_flash_attn are set to False."
            )

    @classmethod
    def from_pattern(cls: "MemoRAGConfig", pattern_string, **kwargs) -> "MemoRAGConfig":
        assert (
            pattern_string in CONFIG_PATTERMS
        ), f"Invalid pattern string: {pattern_string}"

        return cls(**{**CONFIG_PATTERMS[pattern_string], **kwargs})

    def add_corups_files(self, files: list[str]):
        for file in files:
            with open(file, "r") as f:
                self.corups_contents.append(f.read())
        logger.info(f"Added {len(files)} corups files to config.")
