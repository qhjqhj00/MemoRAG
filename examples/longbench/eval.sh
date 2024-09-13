source /opt/conda/bin/activate /share/qhj/env/daily

torchrun --nproc_per_node 8 -m longbench.eval \
            --dataset_names qasper \
            --result_dir test \
            --gen_model_path mistralai/Mistral-7B-Instruct-v0.2 \
            --mem_model_path /share/qhj/memorag-qwen2-7b-inst \
            --ret_model_path BAAI/bge-m3 \
            --access_token [huggingface-token] \
            --eval_data data/longbench.json \
            --max_length 100000 \
