import sys
sys.path.append("./")

import subprocess
import os
from utils import file_control

doc_questions_path = "data/gpt_data/result/documents_questions_final04.jsonl"
ranking_filename = "multilingual-colbert-filter-nbit8.ranking.tsv"
output_path = "data/rerank_data/rerank_triple01.jsonl"

if __name__ == "__main__":
    
    # colbert 기반 reranker 학습용 데이터셋 구축
    print("=== Making Rerank Triple ===")
    ranking_path = file_control.find_ranking_path(ranking_filename)
    file_control.make_rerank_triples(doc_questions_path, ranking_path, output_path, range_top=6, range_bottom=56, neg_size=50)
    
    
    # FlagEmbedding 라이브러리 활용하여 Hard negtaive mind
    print("=== Start Hard negtaive mind using FlagEmbedding ===")
    
    # 필요한 인자들 정의
    model_name_or_path = "BAAI/bge-m3"
    input_file = os.path.abspath("data/rerank_data/rerank_triple01.jsonl")
    output_file = os.path.abspath("data/rerank_data/rerank_neg_triple01.jsonl")
    range_for_sampling = "5-15"
    negative_number = "4"

    # 실행할 명령어
    hn_mind_command = f"python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
    --model_name_or_path {model_name_or_path} \
    --input_file {input_file} \
    --output_file {output_file} \
    --range_for_sampling {range_for_sampling} \
    --negative_number {negative_number} \
    --use_gpu_for_searching"

    # subprocess 모듈을 사용하여 명령어 실행
    print(hn_mind_command.replace("--", "\n"))
    process = subprocess.Popen(hn_mind_command, shell=True)
    process.wait()  # 명령어 실행 완료까지 대기
    
    
    ## Finetune Reranker
    print("=== Start Finetune Reranker using FlagEmbedding ===")
    # 필요한 인자들 정의
    output_dir = os.path.abspath("models/ko_reranker")
    model_name_or_path = "Dongjin-kr/ko-reranker"
    train_data = os.path.abspath("data/rerank_data/rerank_neg_triple01.jsonl")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    finetune_command = f"torchrun --nproc_per_node 1 \
    -m FlagEmbedding.reranker.run \
    --output_dir {output_dir} \
    --model_name_or_path {model_name_or_path} \
    --train_data {train_data} \
    --learning_rate 5e-6 \
    --fp16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --dataloader_drop_last True \
    --train_group_size 4 \
    --max_len 512 \
    --weight_decay 0.01 \
    --logging_steps 30 \
    --save_steps 500 \
    --save_total_limit 3"

    # subprocess 모듈을 사용하여 bash 명령어 실행
    process = subprocess.Popen(finetune_command, shell=True)
    process.wait()  # 명령어 실행 완료까지 대기
    
    