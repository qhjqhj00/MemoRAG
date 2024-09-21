import torch
from transformers.utils import logging
from typing import Dict, Union, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from transformers.tokenization_utils_base import BatchEncoding
from itertools import chain
from semantic_text_splitter import TextSplitter
from typing import Dict, List, Union
import os 
import time
import json
import tiktoken
import copy
from minference import MInference
from langdetect import detect
from .memorag import Model, merge_inputs
from .retrieval import DenseRetriever, FaissIndex
from .prompt import en_prompts, zh_prompts
import pynvml

def get_first_gpu_memory():
    pynvml.nvmlInit()

    device_count = pynvml.nvmlDeviceGetCount()

    if device_count > 0:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        free_memory = mem_info.free / 1024 ** 2  

        return free_memory
    else:
        return None

    pynvml.nvmlShutdown()

class MemoRAGLite:
    def __init__(
        self,
        gen_model_name_or_path: str="Qwen/Qwen2.5-1.5B-Instruct",
        ret_model_name_or_path: str="BAAI/bge-m3",
        customized_gen_model=None,
        ret_hit: int = 3,
        retrieval_chunk_size: int = 512,
        cache_dir: Optional[str] = None,
        access_token: Optional[str] = None,
        load_in_4bit: bool = False,
        enable_flash_attn: bool = True):
        
        if gen_model_name_or_path.find("Qwen2.5-1.5B-Instruct") == -1:
            self.adapt_bs = False
        else:
            self.adapt_bs = True

        if gen_model_name_or_path:
            self.gen_model = Model(
                gen_model_name_or_path, cache_dir=cache_dir, access_token=access_token, load_in_4bit=load_in_4bit, enable_flash_attn=enable_flash_attn)
        elif customized_gen_model:  # for API-based models
            self.gen_model = customized_gen_model
        else:
            raise NotImplementedError

        self.ret_model_name_or_path = ret_model_name_or_path
        self.retrieval_chunk_size = retrieval_chunk_size
        self.ret_hit = ret_hit
        self.cache_dir = cache_dir
        self.load_in_4bit = load_in_4bit

        self.prefix = "<|im_start|>user\n{input}"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n"

        self.gists = None
        self.memory = None
        self.prompts = None
        self.context_inputs = None
        self.retriever = None
        self.retrieval_corpus = None

    def __call__(
        self, 
        query: str = None, 
        context: str = None, 
        task_type: str = "memorag", 
        prompt_template: str = None,
        max_new_tokens: int = 256,
        reset_each_call: bool = False,
        use_memory_answer: bool = False
    ):
        assert self.gen_model is not None
        
        if reset_each_call:
            self.reset()

        if not self.memory:
            if not context:
                raise ValueError("Please provide your input context...")

            self.memorize(context)

        if task_type == 'qa':
            return self.answer(query, max_new_tokens)
        elif task_type == 'memorag':
            return self._handle_rag(query, max_new_tokens, use_memory_answer)
        else:
            raise NotImplementedError(f"Task type '{task_type}' is not supported.")

    def _handle_rag(self, query: str, max_new_tokens: int=128, use_memory_answer: bool=True):
        text_spans = self.recall(query)
        surrogate_queries = self.rewrite(query)
        retrieval_query, potential_answer = self._prepare_retrieval_query(
            query, text_spans, surrogate_queries, use_memory_answer)

        retrieval_results = self._retrieve(retrieval_query)
        if potential_answer:
            retrieval_results.append(f"The answer might be {potential_answer}.")

        knowledge = "\n\n".join(retrieval_results)
        _prompt = self.prompts["qa_gen"].format(context=knowledge, input=query)
        return self.gen_model.generate(_prompt, max_new_tokens=max_new_tokens, repetition_penalty=1.2)[0]


    def _prepare_retrieval_query(self, query, text_spans, surrogate_queries, use_memory_answer):
        retrieval_query = text_spans.split("\n") + surrogate_queries.split("\n")
        if self.language == "zh-cn":
            retrieval_query = [q for q in retrieval_query if len(q) > 3] # TODO
        else:
            retrieval_query = [q for q in retrieval_query if len(q.split()) > 3]

        potential_answer = None
        if use_memory_answer:
            potential_answer = self.answer(query)
            retrieval_query.append(potential_answer)
        retrieval_query.append(query)
        return retrieval_query, potential_answer

    def _retrieve(self, retrieval_query):
        topk_scores, topk_indices = self.retriever.search(queries=retrieval_query)
        topk_indices = list(chain(*[topk_index.tolist() for topk_index in topk_indices]))
        topk_indices = sorted(set([x for x in topk_indices if x > -1]))
        return [self.retrieval_corpus[i].strip() for i in topk_indices]

    def reset(self):
        torch.cuda.empty_cache()
        if self.retriever:
            self.retriever.remove_all()
        self.gists = None
        self.memory = None
        self.prompts = None
        self.context_inputs = None
        self.language = None

    def adapt_batch_size(self):
        free_memory = get_first_gpu_memory()

        if free_memory < 23000:
            print(f"The minimum recommended GPU memory for MemoRAG is 24GiB, but only {round(free_memory / 1024, 1)} GiB is available.")

        if self.adapt_bs:
            memory_thresholds = {
                "en": [(70000, 16), (60000, 10), (38000, 8), (20000, 4), (14000, 2)], 
                "zh-cn": [(70000, 16), (60000, 10), (38000, 8), (20000, 4), (14000, 2)]  
            }
            thresholds = memory_thresholds.get(self.language, memory_thresholds["en"])

            for threshold, bs in thresholds:
                if free_memory > threshold:
                    batch_size = bs
                    break
        return batch_size

    def memorize(
        self, 
        context: str, 
        save_dir: str = None, 
        print_stats: bool = True, 
        batch_size: int = 1,
        gist_chunk_size: int = 4096):
    
        self.reset()

        # Detect language
        text_sample = context[:1024]
        self.language = detect(text_sample)
        if print_stats:
            print(f"Detected language: {self.language}")

        batch_size = self.adapt_batch_size()

        # Encode context
        encoding = tiktoken.get_encoding("cl100k_base")
        encoded_context = encoding.encode(context)
        if print_stats:
            print(f"Context length: {len(encoded_context)} tokens")

        # Set appropriate prompts based on detected language
        self.prompts = zh_prompts if self.language == "zh-cn" else en_prompts

        # Split context into gists
        text_splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", gist_chunk_size)
        gist_chunks = text_splitter.chunks(context)
        gist_chunks = [self.prompts["gist"].format(context=chunk) for chunk in gist_chunks]

        # Generate gists
        if print_stats:
            print(f"Forming memory of the context...")

        self.gists = []
        for i in range(0, len(gist_chunks), batch_size):
            if print_stats and i > 1:
                progress = round(i / len(gist_chunks) * 100, 2)
                print(f"Progress: {progress}% of the context memorized...")

            gists_batch = self.gen_model.generate(
                gist_chunks[i:i+batch_size], 
                batch_size=batch_size, 
                max_new_tokens=512, 
                repetition_penalty=1.2)
            torch.cuda.empty_cache()
            
            self.gists.extend(gists_batch)

        # Join generated gists and clear cache
        gists_concatenated = "\n".join(self.gists)
        torch.cuda.empty_cache()

        # Prepare the context for memory formation
        context_to_encode = self.prefix.format(
            input=self.prompts["context"].format(context=gists_concatenated))

        context_inputs = self.gen_model.tokenizer(
            [context_to_encode], 
            add_special_tokens=False, 
            return_tensors="pt", 
            padding=True
        ).to(self.gen_model.model.device)

        self.context_inputs = context_inputs

        # Initialize memory and process the context through the model
        self.memory = DynamicCache()
        with torch.no_grad():
            model_outputs = self.gen_model.model(**context_inputs, past_key_values=self.memory)
        self.memory = model_outputs.past_key_values
        torch.cuda.empty_cache()

        if print_stats:
            print("Context memorization completed successfully.")

        # Set up dense retrieval index
        self.text_splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", self.retrieval_chunk_size)
        self.retriever = DenseRetriever(
            self.ret_model_name_or_path,
            hits=self.ret_hit,
            cache_dir=self.cache_dir,
            load_in_4bit=self.load_in_4bit
        )

        # Add retrieval corpus and build the index
        self.retrieval_corpus = self.text_splitter.chunks(context)
        with torch.no_grad():
            self.retriever.add(self.retrieval_corpus)
        torch.cuda.empty_cache()

        if print_stats:
            print("Dense retrieval index has been built.")

        # Save memory and index if save_dir is specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {
                    "memory": self.memory,
                    "context_inputs": self.context_inputs,
                    "prompts": self.prompts,
                    "language": self.language
                }, 
                os.path.join(save_dir, "memory.bin")
            )
            self.retriever._index.save(os.path.join(save_dir, "index.bin"))
            with open(os.path.join(save_dir, "chunks.json"), "w") as f:
                json.dump(self.retrieval_corpus, f, ensure_ascii=False, indent=2)

            if print_stats:
                self._print_stats(save_dir, context)

    def _print_stats(self, save_dir: str, context: str = None):
        memory_path = os.path.join(save_dir, "memory.bin")
        memory_size_gb = os.path.getsize(memory_path) / (1024 ** 3)
        print(f"Memory file size: {memory_size_gb:.2f} GB")
        print(f"Number of chunks in retrieval corpus: {len(self.retrieval_corpus)}")

    def generate_w_memory(
        self, 
        instruct: Union[str, List[str]], 
        query: str = "",  
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
        do_sample: bool = False,
        with_cache: bool = True,
        repetition_penalty: float=1.2):
        
        if not self.memory:
            raise ValueError("Memory is not initialized. Please ensure that memory has been formed before using generate.")

        if isinstance(instruct, str):
            instruct = [instruct]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "past_key_values": copy.deepcopy(self.memory),
            "repetition_penalty": repetition_penalty
        }

        outputs = []
        for i, inst in enumerate(instruct):
            
            if query:
                inst = inst.format(question=query)
            sample_to_encode = f"{inst}{self.suffix}"
            sample_inputs = self.gen_model.tokenizer(
                                [sample_to_encode], 
                                add_special_tokens=False, 
                                return_tensors="pt", 
                                padding=True
                            ).to(self.gen_model.model.device)

            sample_inputs = merge_inputs(self.context_inputs, sample_inputs)
            response = self.gen_model.ids2text(sample_inputs, **generation_kwargs)
            outputs.extend(response)

        return outputs

    def load(self, path):
        _cache = torch.load(os.path.join(path, "memory.bin"))
        self.memory = _cache["memory"]
        self.context_inputs = _cache["context_inputs"]
        self.prompts = _cache["prompts"]
        self.language = _cache["language"]
        
        if not self.retriever:
            self.retriever = DenseRetriever(
                self.ret_model_name_or_path,
                hits=self.ret_hit,
                cache_dir=self.cache_dir,
                load_in_4bit=self.load_in_4bit
            )
        _index = FaissIndex(self.retriever.device)
        _index.load(os.path.join(path, "index.bin"))
        self.retriever._index = _index
        self.retrieval_corpus = json.load(open(os.path.join(path, "chunks.json")))


    def answer(
        self,
        query, max_new_tokens=128) -> str:
        return self.generate_w_memory(self.prompts["qa"], query, max_new_tokens=max_new_tokens)[0]

    def recall(
        self,
        query, max_new_tokens=128) -> str:
        return self.generate_w_memory(self.prompts["span"], query, max_new_tokens=max_new_tokens)[0]

    def rewrite(
        self,
        query, max_new_tokens=128) -> str:
        return self.generate_w_memory(self.prompts["sur"], query, max_new_tokens=max_new_tokens)[0]

    def summarize(
        self, max_new_tokens:int=512) -> str:
        return self.generate_w_memory(self.prompts["sum"], max_new_tokens=max_new_tokens)[0]
        