import torch
import faiss
import numpy as np
from typing import List, Mapping, Optional, Union
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers.utils import logging
from semantic_text_splitter import TextSplitter

logger = logging.get_logger(__name__)

class FaissIndex:
    def __init__(self, device) -> None:
        if isinstance(device, torch.device):
            if device.index is None:
                device = "cpu"
            else:
                device = device.index
        self.device = device

    def build(self, doc_embeddings, index_factory, metric):
        if metric == "l2":
            metric = faiss.METRIC_L2
        elif metric in ["ip", "cos"]:
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise NotImplementedError(f"Metric {metric} not implemented!")
        
        index = faiss.index_factory(doc_embeddings.shape[1], index_factory, metric)
        
        if self.device != "cpu":
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            # logger.info("using fp16 on GPU...")
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device, index, co)

        index.train(doc_embeddings)
        index.add(doc_embeddings)
        self.index = index
        return index
    
    def add(self, doc_embeddings):
        self.index.add(doc_embeddings)

    def load(self, index_path):
        # logger.info(f"loading index from {index_path}...")
        index = faiss.read_index(index_path)
        if self.device != "cpu":
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device, index, co)
        self.index = index

    def save(self, index_path):
        logger.info(f"saving index at {index_path}...")
        if isinstance(self.index, faiss.GpuIndex):
            index = faiss.index_gpu_to_cpu(self.index)
        else:
            index = self.index
        faiss.write_index(index, index_path)

    def search(self, query, hits):
        return self.index.search(query, k=hits)


class DenseRetriever:
    def __init__(
        self, 
        encoder:str='BAAI/bge-large-en-v1.5', 
        pooling_method:List[str]=["cls"], 
        dense_metric:str="cos", 
        query_max_length:int=128, 
        key_max_length:int=512, 
        hits:int=10, 
        dtype:str="fp16", 
        cache_dir:Optional[str]=None, 
        query_instruct:str=None, 
        doc_instruct:str=None) -> None:
        self.name = encoder
        self.query_instruct = query_instruct
        self.doc_instruct = doc_instruct

        self.pooling_method = pooling_method
        self.dense_metric = dense_metric
        self.query_max_length = query_max_length
        self.key_max_length = key_max_length
        self.hits = hits
        logger.info(f"Loading tokenizer and model from {encoder}...")

        if dtype == "bf16":
            dtype = torch.bfloat16
        elif dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(encoder, cache_dir=cache_dir)
        self.encoder = AutoModel.from_pretrained(encoder, cache_dir=cache_dir, torch_dtype=dtype, device_map={'': "cuda"}).eval()

        self.ndim = self.encoder.config.hidden_size
        self._index = None
        self.docs = []

    @property
    def device(self):
        return self.encoder.device

    @property
    def num_keys(self):
        if self._index is not None:
            return self._index.index.ntotal
        else:
            return 0

    def _prepare(self, inputs: Union[str, List[str], Mapping], field="key"):
        """Convert inputs into tokenized input_ids"""
        if isinstance(inputs, str) or (isinstance(inputs, list) and isinstance(inputs[0], str)):
            if field == "key":
                inputs = self.tokenizer(
                    inputs, return_tensors="pt", padding=True, truncation=True, max_length=self.key_max_length)
                inputs = inputs.to(self.device)
            elif field == "query":
                inputs = self.tokenizer(
                    inputs, return_tensors="pt", padding=True, truncation=True, max_length=self.query_max_length)
                inputs = inputs.to(self.device)
            else:
                raise NotImplementedError
        elif isinstance(inputs, Mapping) and "input_ids" in inputs:
            if field == "key":
                for k, v in inputs.items():
                    inputs[k] = v[:, :self.key_max_length].to(self.device)
            elif field == "query":
                for k, v in inputs.items():
                    inputs[k] = v[:, :self.query_max_length].to(self.device)
            else:
                raise NotImplementedError
        else:
            raise ValueError(f"Expected inputs of type str, list[str], or dict, got {type(inputs)}!")
        return inputs

    def _pool(self, embeddings, attention_mask):
        if "mean" in self.pooling_method:
            embeddings = embeddings.masked_fill(
                ~attention_mask[..., None].bool(), 0.0)
            embedding = embeddings.sum(
                dim=1) / attention_mask.sum(dim=1, keepdim=True)
        elif "cls" in self.pooling_method:
            embedding = embeddings[:, 0]
        else:
            raise NotImplementedError(
                f"Pooling_method {self.pooling_method} not implemented!")
        return embedding

    @torch.no_grad()
    def encode(self, inputs: Union[str, List[str], Mapping], field:str="key"):
        """Encode inputs into embeddings

        Args:
            inputs: can be string, list of strings, or BatchEncoding results from tokenizer

        Returns:
            Tensor: [batch_size, d_embed]
        """
        inputs = self._prepare(inputs, field=field)
        encoder = self.encoder

        embeddings = encoder(**inputs).last_hidden_state    # B, L, D
        embedding = self._pool(embeddings, inputs["attention_mask"])
        if self.dense_metric == "cos":
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

    def remove_all(self):
        """Remove all keys from the index."""
        if self._index is not None:
            self._index.index.reset()
        self.docs = []

    @torch.no_grad()
    def add(self, docs: List[str], index_factory:str="Flat", batch_size=500):
        """Build faiss index.
        
        Args:
            shard_across_devices: split the corpus onto all devices and encode them
        """
        if len(docs) == 0:
            return

        metric = self.dense_metric
        doc_embeddings = np.zeros((len(docs), self.ndim), dtype=np.float32)

        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i: i + batch_size]
            embeddings = self.encode(batch_docs)    # batch_size, ndim
            doc_embeddings[i: i + batch_size] = embeddings.cpu().numpy()

        if self._index is None:
            index = FaissIndex(self.device)
            index.build(doc_embeddings, index_factory, metric)
            self._index = index
        else:
            self._index.add(doc_embeddings)

        self.docs.extend(docs)

    @torch.no_grad()
    def search(self, queries: Union[str, List[str]], hits:Optional[int]=None):
        if hits is None:
            hits = self.hits
    
        assert self._index is not None, "Make sure there is an indexed corpus!"

        embeddings = self.encode(queries, field="query").cpu().numpy().astype(np.float32, order="C")
        scores, indices = self._index.search(embeddings, hits)
        return scores, indices

