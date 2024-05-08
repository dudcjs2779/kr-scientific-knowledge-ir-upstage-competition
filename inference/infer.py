import sys
sys.path.append("./")

import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import file_control
from utils import inference
from train import colbert_train

from transformers import AutoModelForSequenceClassification, AutoTokenizer

main_path = "sc_documents"
index_name = "multilingual-colbert-filter-nbit8"
eval_list = file_control.read_jsonl("data/eval_query_gpt4.jsonl")
docs = file_control.read_jsonl("data/documents.jsonl")


def evaluate_colbert(eval_list, docs, index_name, size, main_path):
    # colbert retrieval 결과 가져오기
    search_result = colbert_train.make_search_result_faiss(eval_list, docs, index_name, size, main_path)
    
    # Valid 셋으로 평가
    def keystoint(x):
        return {int(k): v for k, v in x.items()}

    with open("data/gt_eval_dict.json", 'r') as f:
        gt_eval_dict = json.load(f, object_hook=keystoint)
        
    mAP = inference.calc_map(gt_eval_dict, search_result[0:101])
    print("ColBERT vaild mAP Score: ", mAP)
    

def make_rerank_inputs(row):
    ranking_list =[]
    for reference in row['references']:
        pair = [row['standalone_query'], reference['content']]
        ranking_list.append(pair)
        
    return ranking_list

def make_rerank_list(row):
    rerank_list =[]
    for i in range(len(row['references'])):
        data = {"eval_id": row['eval_id'], "standalone_query": row['standalone_query'], "re_topk": row['topk'][i], "answer": "", "content": row['references'][i]['content']}
        rerank_list.append(data)
        
    return rerank_list

def make_batch_list(lst, batch_size):
    chunks = []
    for i in range(0, len(lst), batch_size):
        chunks.append(lst[i:i + batch_size])
    return chunks

def make_references(row):
    data_list = []
    for i in range(len(row['content'])):
        data = {"score": row['score'][i], "content": row['content'][i]}
        data_list.append(data)
        
    return data_list

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()


def inference_reranker(model, tokenizer, device, rerank_batchs):
    model.to(device)
    model.eval()
    with torch.no_grad():
        list_result = []
        for batch in tqdm(rerank_batchs, desc="Reranking"):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = inputs.to(device)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = exp_normalize(scores.detach().cpu().numpy())
            # # print (f'first: {scores[0]}, second: {scores[1]}')
            list_result.append(scores)
            
    list_result = [item for result in list_result for item in result]
    return list_result


def make_submission(search_df, scores):
    rerank_list = search_df.apply(make_rerank_list, axis=1)
    rerank_list = [item for rerank in rerank_list for item in rerank]
    
    ranking_df = pd.DataFrame(rerank_list)
    ranking_df['score'] = scores
    ranking_df = ranking_df.sort_values(by=['eval_id', 'score'], ascending=[True, False]).reset_index(drop=True)
    
    agg_func = {
        'standalone_query': 'first',
        're_topk': list,
        'answer': 'first',
        'content': list,
        'score': list,
    }
    ranking_df = ranking_df.groupby('eval_id').agg(agg_func).reset_index()
    ranking_df['re_references'] = ranking_df.apply(make_references, axis=1)
    ranking_df = ranking_df.drop(columns=['content', 'score'])
    
    search_df = search_df.merge(ranking_df[['eval_id', 're_topk', 're_references']], on='eval_id', how='left')
    search_df['topk'] = search_df['re_topk']
    search_df['references'] = search_df['re_references']
    search_df = search_df.drop(columns=['re_topk', 're_references'])
    
    return search_df

def evaluate_reranker(search_list, model_name, output_path, batch_size):
    search_df = pd.DataFrame(search_list)
    print("Load Model and Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reranker 모델에 넣어줄 input
    rerank_inputs = search_df.apply(make_rerank_inputs, axis=1)
    rerank_inputs = [item for inputs in rerank_inputs for item in inputs]
    rerank_batchs = make_batch_list(rerank_inputs, batch_size)

    # 추론
    list_result = inference_reranker(model, tokenizer, device, rerank_batchs)
    
    # rerank 결과 제출 형식에 맞게 재구성
    search_df = make_submission(search_df, list_result)
    
    # 결과 저장
    search_df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Save rerank result at {output_path}")
    rerank_list = file_control.read_jsonl(output_path)
    
    # 결과 평가
    def keystoint(x):
        return {int(k): v for k, v in x.items()}

    with open("data/gt_eval_dict.json", 'r') as f:
        gt_eval_dict = json.load(f, object_hook=keystoint)

    # pred_df = pd.DataFrame(rerank_list).loc[0:100]
    # gt_df = file_control.read_jsonl("data/gt_eval.jsonl", to_csv=True)
    mAP = inference.calc_map(gt_eval_dict, rerank_list[0:101])
    print("Rerank vaild mAP Score: ", mAP)
    


if __name__ == "__main__":
    # # == ColBERT Valid Score ==
    print("## Only ColBERT Valid Score")
    evaluate_colbert(eval_list, docs, index_name, 10, main_path)
    
    # == None Finetune Reranker Valid Score(Top10) ==
    print("\n\n## None Finetune Reranker Valid Score(Top10)")
    # colbert retrieval 결과 가져오기
    print("ColBERT Retrieve")
    search_list = colbert_train.make_search_result_faiss(eval_list, docs, index_name, 10, main_path)
    
    # Reranker 평가
    model_name = "Dongjin-kr/ko-reranker"
    output_path = "outputs/rerank_none_finetune01.jsonl"
    evaluate_reranker(search_list, model_name, output_path)
    
    # == Finetune Reranker Valid Score(Top10) ==
    print("\n\n## Finetune Reranker Valid Score(Top10)")
    # colbert retrieval 결과 가져오기
    print("ColBERT Retrieve")
    search_list = colbert_train.make_search_result_faiss(eval_list, docs, index_name, 10, main_path)
    
    # Reranker 평가
    model_name = "models/ko_reranker"
    output_path = "outputs/rerank_finetune01.jsonl"
    evaluate_reranker(search_list, model_name, output_path)

