import torch
from transformers.utils import logging

logger = logging.get_logger(__name__)

from typing import Dict, Union, List, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DynamicCache,
)
from transformers.tokenization_utils_base import BatchEncoding
from itertools import chain
from semantic_text_splitter import TextSplitter
from .retrieval import DenseRetriever, FaissIndex
from typing import Dict, List, Union
from .prompt import en_prompts, zh_prompts
import os
import json
import tiktoken
import copy

try:
    from minference import MInference
except ModuleNotFoundError:
    logger.warning(
        "MInference is not installed. You can install it by running `pip install minference`."
    )


def merge_inputs(inputs1: BatchEncoding, inputs2: BatchEncoding) -> BatchEncoding:

    merged_input_ids = torch.cat([inputs1["input_ids"], inputs2["input_ids"]], dim=1)
    merged_attention_mask = torch.cat(
        [inputs1["attention_mask"], inputs2["attention_mask"]], dim=1
    )

    merged_inputs = BatchEncoding(
        {"input_ids": merged_input_ids, "attention_mask": merged_attention_mask}
    )
    return merged_inputs


class Model:
    def __init__(
        self,
        model_name_or_path: str,
        cache_dir: str = "",
        access_token: str = "",
        beacon_ratio: int = None,
        load_in_4bit: bool = False,
        enable_flash_attn: bool = True,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if enable_flash_attn:
            if model_name_or_path.find("mistral") != -1:
                attn_implementation = "sdpa"
            else:
                attn_implementation = "flash_attention_2"
        else:
            attn_implementation = None

        if model_name_or_path.find("memorag") == -1:
            load_in_4bit = True

        self.model_kwargs = {
            "cache_dir": cache_dir,
            "token": access_token,
            "device_map": {"": device},
            "attn_implementation": attn_implementation,
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        self.model_name_or_path = model_name_or_path

        if load_in_4bit:
            quant_config = BitsAndBytesConfig(load_in_4bit=load_in_4bit)
            self.model_kwargs["quantization_config"] = quant_config

        if beacon_ratio and model_name_or_path.find("memorag") != -1:
            self.model_kwargs["beacon_ratio"] = [beacon_ratio]

        tokenizer_kwargs = {
            "cache_dir": cache_dir,
            "token": access_token,
            "padding_side": "left",
            "trust_remote_code": True,
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, **tokenizer_kwargs
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **self.model_kwargs
        ).eval()

        logger.info(f"Model loaded from {model_name_or_path}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def ids2text(self, inputs, **generation_kwargs) -> str:
        outputs = self.model.generate(
            **inputs, **generation_kwargs, pad_token_id=self.tokenizer.eos_token_id
        )

        decoded_output = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return decoded_output

    def template2ids(self, templates: List, remove_symbol=None):
        if isinstance(templates, str):
            templates = [templates]

        batch_prompts = []
        for template in templates:
            to_encode = self.tokenizer.apply_chat_template(
                template, tokenize=False, add_generation_prompt=True
            )
            if remove_symbol:
                to_encode = to_encode.replace(remove_symbol, "")
            batch_prompts.append(to_encode)

        inputs = self.tokenizer(
            batch_prompts, add_special_tokens=False, return_tensors="pt", padding=True
        ).to(self.model.device)

        return inputs

    def minference_patch(
        self, model_type: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ):
        minference_patch = MInference("minference", model_type)
        self.model = minference_patch(self.model)

    def reload_model(self):
        # TODO
        del self.model
        torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, **self.model_kwargs
        ).eval()

    def generate(
        self,
        prompts: Union[str, List[str]],
        batch_size: int = 1,
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
    ) -> Union[str, List[str]]:

        if isinstance(prompts, str):
            prompts = [prompts]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }

        all_outputs = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = []
            for prompt in prompts[i : i + batch_size]:
                batch_prompts.append([{"role": "user", "content": prompt}])
            inputs = self.template2ids(batch_prompts)
            outputs = self.ids2text(inputs, **generation_kwargs)
            all_outputs.extend(outputs)
        return all_outputs


class Memory(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = None
        if self.model_name_or_path.find("memorag") != -1:
            self.memo_type = "beacon"
        else:
            self.memo_type = "longllm"

        if self.model_name_or_path.lower().find("chinese") != -1:
            self.prompts = zh_prompts
        else:
            self.prompts = en_prompts

    def memorize(self, context, max_length=None, reload_model: bool = True):

        context_inputs = self.template2ids(
            [
                [
                    {
                        "role": "user",
                        "content": self.prompts["context"].format(context=context),
                    },
                    {
                        "role": "assistant",
                        "content": "I have read the article. Please provide your question.",
                    },
                ]
            ]
        )
        if self.memo_type == "beacon":
            self.reset()
            with torch.no_grad():
                self.model(**context_inputs)
            self.memory = self.model.memory.export()
        elif self.memo_type == "longllm":
            self.minference_patch()
            self.memory = DynamicCache()
            with torch.no_grad():
                model_outputs = self.model(
                    **context_inputs, past_key_values=self.memory
                )
            self.memory = model_outputs.past_key_values
            self.context_inputs = context_inputs
            if reload_model:
                self.reload_model()

    def reset(self) -> None:
        self.memory = None
        self.model.memory.reset()

    def answer(self, query, max_new_tokens=128) -> str:
        return self.generate(self.prompts["qa"], query, max_new_tokens=max_new_tokens)[
            0
        ]

    def recall(self, query, max_new_tokens=128) -> str:
        return self.generate(
            self.prompts["span"], query, max_new_tokens=max_new_tokens
        )[0]

    def rewrite(self, query, max_new_tokens=128) -> str:
        return self.generate(self.prompts["sur"], query, max_new_tokens=max_new_tokens)[
            0
        ]

    def summarize(self, max_new_tokens: int = 512) -> str:
        return self.generate(self.prompts["sum"], max_new_tokens=max_new_tokens)[0]

    def generate(
        self,
        instruct: Union[str, List[str]],
        query: str = "",
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
        do_sample: bool = False,
        with_cache: bool = True,
    ) -> List[str]:
        if not self.memory:
            raise ValueError(
                "Memory is not initialized. Please ensure that memory has been formed before using generate."
            )

        if isinstance(instruct, str):
            instruct = [instruct]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
        }
        if self.memo_type == "longllm" and with_cache:
            generation_kwargs["past_key_values"] = copy.deepcopy(self.memory)

        outputs = []

        for i, inst in enumerate(instruct):
            if self.memo_type == "beacon":
                self.model.memory.reset(**self.memory)
            if query:
                sample_inputs = self.template2ids(
                    [[{"role": "user", "content": inst.format(question=query)}]]
                )
            else:
                sample_inputs = self.template2ids([[{"role": "user", "content": inst}]])
            if self.memo_type == "longllm" and with_cache:
                sample_inputs = merge_inputs(self.context_inputs, sample_inputs)
            response = self.ids2text(sample_inputs, **generation_kwargs)
            outputs.extend(response)
            if self.memo_type == "longllm" and with_cache:
                del generation_kwargs["past_key_values"]
                torch.cuda.empty_cache()
        return outputs

    def save(self, path):
        if self.memo_type == "beacon":
            torch.save(self.memory, path)
        elif self.memo_type == "longllm":
            torch.save(
                {"memory": self.memory, "context_inputs": self.context_inputs}, path
            )
        else:
            raise NotImplementedError

    def load(self, path):
        if self.memo_type == "beacon":
            self.memory = torch.load(path)
        elif self.memo_type == "longllm":
            _cache = torch.load(path)
            self.memory = _cache["memory"]
            self.context_inputs = _cache["context_inputs"]


class MemoRAG:
    def __init__(
        self,
        mem_model_name_or_path: str,
        ret_model_name_or_path: str,
        gen_model_name_or_path: str = None,
        customized_gen_model=None,
        ret_hit: int = 3,
        retrieval_chunk_size: int = 512,
        cache_dir: Optional[str] = None,
        access_token: Optional[str] = None,
        beacon_ratio: int = 4,
        load_in_4bit: bool = False,
        enable_flash_attn: bool = True,
    ):

        if mem_model_name_or_path.lower().find("chinese") != -1:
            self.prompts = zh_prompts
            retrieval_chunk_size = 2048
        else:
            self.prompts = en_prompts

        self.mem_model = Memory(
            mem_model_name_or_path,
            cache_dir=cache_dir,
            beacon_ratio=beacon_ratio,
            load_in_4bit=load_in_4bit,
            enable_flash_attn=enable_flash_attn,
        )

        if gen_model_name_or_path:
            self.gen_model = Model(
                gen_model_name_or_path,
                cache_dir=cache_dir,
                access_token=access_token,
                load_in_4bit=load_in_4bit,
                enable_flash_attn=enable_flash_attn,
            )
        elif customized_gen_model:  # for API-based models
            self.gen_model = customized_gen_model
        else:
            self.gen_model = self.mem_model

        self.retriever = DenseRetriever(
            ret_model_name_or_path,
            hits=ret_hit,
            cache_dir=cache_dir,
            load_in_4bit=load_in_4bit,
        )

        self.text_splitter = TextSplitter.from_tiktoken_model(
            "gpt-3.5-turbo", retrieval_chunk_size
        )

    def memorize(self, context: str, save_dir: str = None, print_stats: bool = False):
        self.retriever.remove_all()

        self.mem_model.memorize(context)
        self.retrieval_corpus = self.text_splitter.chunks(context)
        self.retriever.add(self.retrieval_corpus)

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.mem_model.save(os.path.join(save_dir, "memory.bin"))
            self.retriever._index.save(os.path.join(save_dir, "index.bin"))
            with open(os.path.join(save_dir, "chunks.json"), "w") as f:
                json.dump(self.retrieval_corpus, f, ensure_ascii=False, indent=2)
            if print_stats:
                self._print_stats(save_dir, context)

    def _print_stats(self, save_dir: str, context: str = None):
        memory_path = os.path.join(save_dir, "memory.bin")
        memory_size_gb = os.path.getsize(memory_path) / (1024**3)
        print(f"Memory file size: {memory_size_gb:.2f} GB")

        encoding = tiktoken.get_encoding("cl100k_base")
        if context:
            encoded_context = encoding.encode(context)
            print(f"Encoded context length: {len(encoded_context)} tokens")
        print(f"Number of chunks in retrieval corpus: {len(self.retrieval_corpus)}")

    def load(self, save_dir: str, print_stats: bool = False):
        self.mem_model.load(os.path.join(save_dir, "memory.bin"))
        _index = FaissIndex(self.retriever.device)
        _index.load(os.path.join(save_dir, "index.bin"))
        self.retriever._index = _index
        self.retrieval_corpus = json.load(open(os.path.join(save_dir, "chunks.json")))
        if print_stats:
            self._print_stats(save_dir)

    def __call__(
        self,
        query: str = None,
        context: str = None,
        task_type: str = "memorag",
        prompt_template: str = None,
        max_new_tokens: int = 256,
        reset_each_call: bool = False,
        use_memory_answer: bool = False,
    ):
        assert self.gen_model is not None

        if reset_each_call:
            self.mem_model.reset()
            self.retriever.remove_all()

        if not self.mem_model.memory:
            if not context:
                raise ValueError("Please provide your input context...")
            self.memorize(context)

        if task_type == "qa":
            return self._handle_qa(query, max_new_tokens)
        elif task_type == "memorag":
            return self._handle_rag(
                query, prompt_template, max_new_tokens, use_memory_answer
            )
        elif task_type == "summarize":
            return self._handle_summarization(prompt_template, max_new_tokens)
        else:
            raise NotImplementedError(f"Task type '{task_type}' is not supported.")

    def _handle_qa(self, query: str, max_new_tokens: int = 128):
        return self.mem_model.answer(query, max_new_tokens)

    def _handle_rag(
        self,
        query: str,
        prompt_template: str,
        max_new_tokens: int,
        use_memory_answer: bool,
    ):
        text_spans = self.mem_model.recall(query)
        surrogate_queries = self.mem_model.rewrite(query)
        retrieval_query, potential_answer = self._prepare_retrieval_query(
            query, text_spans, surrogate_queries, use_memory_answer
        )

        retrieval_results = self._retrieve(retrieval_query)

        if potential_answer:
            retrieval_results.append(f"The answer might be {potential_answer}.")

        knowledge = "\n\n".join(retrieval_results)

        return self._generate_response(
            "qa_gen", query, knowledge, prompt_template, max_new_tokens
        )

    def _handle_summarization(self, prompt_template: str, max_new_tokens: int):
        key_points = self.mem_model.summarize()
        retrieval_query = [
            query for query in key_points.split("\n") if len(query.split()) > 3
        ]

        retrieval_results = self._retrieve(retrieval_query)
        knowledge = "\n\n".join(retrieval_results)

        return self._generate_response(
            "sum_gen", None, knowledge, prompt_template, max_new_tokens
        )

    def _prepare_retrieval_query(
        self, query, text_spans, surrogate_queries, use_memory_answer
    ):
        retrieval_query = text_spans.split("\n") + surrogate_queries.split("\n")
        retrieval_query = [q for q in retrieval_query if len(q.split()) > 3]
        potential_answer = None
        if use_memory_answer:
            potential_answer = self.mem_model.answer(query)
            retrieval_query.append(potential_answer)
        retrieval_query.append(query)
        return retrieval_query, potential_answer

    def _retrieve(self, retrieval_query):
        topk_scores, topk_indices = self.retriever.search(queries=retrieval_query)
        topk_indices = list(
            chain(*[topk_index.tolist() for topk_index in topk_indices])
        )
        topk_indices = sorted(set([x for x in topk_indices if x > -1]))
        return [self.retrieval_corpus[i].strip() for i in topk_indices]

    def _generate_response(
        self,
        task_key: str,
        query: str,
        knowledge: str,
        prompt_template: str,
        max_new_tokens: int,
    ):
        if prompt_template:
            prompt = (
                prompt_template.format(input=query, context=knowledge)
                if query
                else prompt_template.format(context=knowledge)
            )
        else:
            prompt = (
                self.prompts[task_key].format(input=query, context=knowledge)
                if query
                else self.prompts[task_key].format(context=knowledge)
            )

        if (
            self.gen_model.__class__.__name__ == "Memory"
            and self.mem_model.memo_type == "beacon"
        ):
            # self.gen_model._enable_beacon = False
            output = self.gen_model.generate(prompt, max_new_tokens=max_new_tokens)[0]
            # self.gen_model._enable_beacon = True
        else:
            output = self.gen_model.generate(
                prompt, max_new_tokens=max_new_tokens, with_cache=False
            )[0]
        torch.cuda.empty_cache()
        return output
