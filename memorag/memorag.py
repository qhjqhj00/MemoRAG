import torch
from transformers.utils import logging
from typing import Dict, Union, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import chain
from semantic_text_splitter import TextSplitter
from .retrieval import DenseRetriever, FaissIndex
from typing import Dict, List, Union
from .prompt import prompts
import os 
import json
import tiktoken

logger = logging.get_logger(__name__)          

class Model:
    def __init__(
        self, 
        model_name_or_path: str, 
        cache_dir: str="",
        access_token: str="",
        beacon_ratio: int=None,
    ):  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name_or_path.find("mistral") != -1:
            attn_implementation = "sdpa"
        else:
            attn_implementation = "flash_attention_2"

        model_kwargs = {
            "cache_dir": cache_dir,
            "token": access_token,
            "device_map": {"": device},
            "attn_implementation": attn_implementation,
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if beacon_ratio:
            model_kwargs["beacon_ratio"] = [beacon_ratio]

        tokenizer_kwargs = {
            "cache_dir": cache_dir,
            "token": access_token,
            "padding_side": "left",
            "trust_remote_code": True,
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            **tokenizer_kwargs
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            **model_kwargs
        ).eval()

        logger.info(f"Model loaded from {model_name_or_path}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def ids2text(
        self, 
        inputs, 
        **generation_kwargs
    ) -> str:
        outputs = self.model.generate(
            **inputs, 
            **generation_kwargs, 
            pad_token_id=self.tokenizer.eos_token_id
        )

        decoded_output = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )

        return decoded_output

    def template2ids(
        self, 
        templates: List, 
        remove_symbol=None
    ):
        if isinstance(templates, str):
            templates = [templates]
        
        batch_prompts = []
        for template in templates:
            to_encode = self.tokenizer.apply_chat_template(
                template, 
                tokenize=False, 
                add_generation_prompt=True
            )
            if remove_symbol:
                to_encode = to_encode.replace(remove_symbol, "")
            batch_prompts.append(to_encode)

        inputs = self.tokenizer(
            batch_prompts, 
            add_special_tokens=False, 
            return_tensors="pt", 
            padding=True
        ).to(self.model.device)

        return inputs

    def generate(
        self, 
        prompts: Union[str, List[str]], 
        batch_size: int = 1, 
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
        do_sample: bool = False,
    ) -> Union[str, List[str]]:

        if isinstance(prompts, str):
            prompts = [prompts]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p
        }

        all_outputs = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = []
            for prompt in prompts[i: i + batch_size]:
                batch_prompts.append([{"role": "user", "content": prompt}])
            inputs = self.template2ids(batch_prompts)
            outputs = self.ids2text(inputs, **generation_kwargs)
            all_outputs.extend(outputs)
        return all_outputs


class Memory(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = None

    def memorize(
        self, 
        context, 
        max_length=None
    ):
        self.reset() 
        context_inputs = self.template2ids([[
            {"role": "user", "content": prompts["context"].format(context=context)},
            {"role": "assistant", "content": "I have read the article. Please provide your question."}
        ]])

        self.model(**context_inputs)
        self.memory = self.model.memory.export()

    def reset(
        self
    ) -> None:
        self.memory = None
        self.model.memory.reset()

    def answer(
        self,
        query) -> str:
        return self.generate(prompts["qa"], query, 128)[0][0]

    def recall(
        self,
        query) -> str:
        return self.generate(prompts["span"], query, 256)[0][0]

    def rewrite(
        self,
        query) -> str:
        return self.generate(prompts["sur"], query, 256)[0][0]

    def summarize(
        self, max_new_tokens:int=512) -> str:
        return self.generate(prompts["sum"], max_new_tokens=max_new_tokens)[0][0]

    def generate(
        self, 
        instruct: Union[str, List[str]], 
        query: str = "",  
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
        do_sample: bool = False,
    ) -> List[str]:
        if not self.memory:
            raise ValueError("Memory is not initialized. Please ensure that memory has been formed before using generate.")

        if isinstance(instruct, str):
            instruct = [instruct]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p
        }
        outputs = []

        for i, inst in enumerate(instruct):
            self.model.memory.reset(**self.memory)
            if query:
                sample_inputs = self.template2ids([[{"role": "user", "content": inst.format(question=query)}]])
            else:
                sample_inputs = self.template2ids([[{"role": "user", "content": inst}]])
            response = self.ids2text(sample_inputs, **generation_kwargs)
            outputs.append(response)
        return outputs
    
    def save(self, path):
        torch.save(self.memory, path)

    def load(self, path):
        self.memory = torch.load(path)

class MemoRAG:
    def __init__(
        self, 
        mem_model_name_or_path: str, 
        ret_model_name_or_path: str,
        gen_model_name_or_path: str=None,
        ret_hit:int=3,
        retrieval_chunk_size:int=512,
        cache_dir:Optional[str]=None,
        access_token:Optional[str]=None,):

        self.mem_model = Memory(
            mem_model_name_or_path, cache_dir=cache_dir, beacon_ratio=4)

        self.gen_model = Model(
            gen_model_name_or_path, cache_dir=cache_dir, access_token=access_token)      

        self.retriever = DenseRetriever(
            ret_model_name_or_path, hits=ret_hit, cache_dir=cache_dir)

        self.text_splitter = TextSplitter.from_tiktoken_model(
            "gpt-3.5-turbo", retrieval_chunk_size)

    def memorize(self, context:str, save_dir:str=None, print_stats:bool=False):
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
                memory_path = os.path.join(save_dir, "memory.bin")
                memory_size_gb = os.path.getsize(memory_path) / (1024 ** 3)
                print(f"Memory file size: {memory_size_gb:.2f} GB")

                encoding = tiktoken.get_encoding("cl100k_base")
                encoded_context = encoding.encode(context)
                print(f"Encoded context length: {len(encoded_context)} tokens")

                # Print the length of the split retrieval corpus
                print(f"Number of chunks in retrieval corpus: {len(self.retrieval_corpus)}")

    def load(self, save_dir:str, print_stats:bool=False):
        self.mem_model.load(os.path.join(save_dir, "memory.bin"))
        _index = FaissIndex(self.retriever.device)
        _index.load(os.path.join(save_dir, "index.bin"))
        self.retriever._index = _index
        self.retrieval_corpus = json.load(open(os.path.join(save_dir, "chunks.json")))
        if print_stats:
                memory_path = os.path.join(save_dir, "memory.bin")
                memory_size_gb = os.path.getsize(memory_path) / (1024 ** 3)
                print(f"Loaded Memory Cache: {memory_size_gb:.2f} GB")

                encoding = tiktoken.get_encoding("cl100k_base")
                encoded_context = encoding.encode(context)
                print(f"Encoded context length: {len(encoded_context)} tokens")

                # Print the length of the split retrieval corpus
                print(f"Number of chunks in retrieval corpus: {len(self.retrieval_corpus)}")
            
    def __call__(
        self, 
        context:str, 
        query:str=None, 
        task_type:str="rag", 
        prompt_template:str=None,
        max_new_tokens:int=256,
        reset_each_call:bool=False,
        use_memory_answer:bool=False):
        if reset_each_call:
            self.mem_model.reset()
            self.retriever.remove_all()

        if not self.mem_model.memory:
            self.memorize(context)

        potention_answer = None

        if task_type == 'qa':
            return self.mem_model.answer(query)
        
        elif task_type == 'rag':
            text_spans = self.mem_model.recall(query)
            surrogate_queries = self.mem_model.rewrite(query)
            retrieval_query = text_spans.split("\n") + surrogate_queries.split("\n")
            retrieval_query = [query for query in retrieval_query if len(query.split()) > 3]
            if use_memory_answer:
                potention_answer = self.mem_model.answer(query)
                retrieval_query.append(potention_answer)

            retrieval_query.append(query)
                    
        elif task_type == 'summarize':
            key_points = self.mem_model.summarize()
            retrieval_query = key_points.split("\n")
            retrieval_query = [query for query in retrieval_query if len(query.split()) > 3]
        
        else:
            raise NotImplementedError

        topk_scores, topk_indices = self.retriever.search(
            queries=retrieval_query)
        topk_indices = list(chain(*[topk_index.tolist() for topk_index in topk_indices]))
        topk_indices = list(set(topk_indices))

        topk_indices = sorted([x for x in topk_indices if x > -1])
        retrieval_results = [self.retrieval_corpus[i].strip() for i in topk_indices]
        if potention_answer:
            retrieval_results.append(f"The answer might be {potential_answer}.")
            
        knowledge = "\n\n".join(retrieval_results)

        if task_type in ["rag"]:
            if prompt_template:
                prompt = prompt_template.format(input=query, context=knowledge)
            else:
                prompt = prompts["qa_gen"].format(input=query, context=knowledge)
            answer = self.gen_model.generate(prompt, max_new_tokens=max_new_tokens)[0]
        elif task_type in ['summarize']:
            if prompt_template:
                prompt = prompt_template.format(context=knowledge)
            else:
                prompt = prompts["sum_gen"].format(context=knowledge)
            answer = self.gen_model.generate(prompt, max_new_tokens=max_new_tokens)[0]
        else:
            raise NotImplementedError

        return answer

if __name__ == "__main__":


    # model = Memory(
    #     "/share/qhj/memorag-mistral-7b-inst", beacon_ratio=4
    # )
    pipe = MemoRAG(
        "/share/qhj/rags/data/memory_model/qwen2-sft-0830-beacon/checkpoint-7683",
        "BAAI/bge-m3",
        "mistralai/Mistral-7B-Instruct-v0.2",
        cache_dir="/share/shared_models/",
        access_token="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"
    )
    query = "how many times does the chamber be opened in Harry Potter?"
    # res = model.generate(query, batch_size=2)
    # print(res)
    test_txt = open("examples/harry_potter.txt").read()
    # res = pipe(test_txt, query)
    # print(res)
    pipe.memorize(test_txt, "examples/qwen/", True)
    pipe.load("examples/qwen/")
    res = pipe(test_txt, query, "qa", max_new_tokens=256)
    print(res)
    res = pipe(test_txt, query, "rag", max_new_tokens=256)
    print(res)
    res = pipe(test_txt, query, "summarize", max_new_tokens=512)
    print(res)
    
