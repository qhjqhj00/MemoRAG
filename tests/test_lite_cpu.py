from memorag import MemoRAGLite
from memorag.config import MemoRAGConfig


def test_cpu_lite_init():
    config = MemoRAGConfig()
    assert config.load_in_4bit == False
    assert config.enable_flash_attn == False
    model = MemoRAGLite.from_config(config)
