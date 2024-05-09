import json
import os
import pandas as pd
from tqdm import tqdm
import random

def read_jsonl(path, to_csv=False):
    with open(path) as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))

    if to_csv:
        return pd.DataFrame(json_list)
    else:
        return json_list
    
    
def conbine_json(folder_path, output_path):
    files = os.listdir(folder_path)

    json_list = []
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            # print(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            json_list.append(json_data)
            
    with open(output_path, 'w') as f:
        for entry in json_list:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
            
def delete_jsons(folder_path):
    files = os.listdir(folder_path)

    count = 0
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            count += 1
    print(f"{count} files is deleted")
    
    
def save_to_jsonl(save_path, data_list):
    with open(save_path, "w") as f:
        for data in data_list:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"jsonl file is saved at {save_path}")
            
            
def read_ranking(ranking_path):
    rankings = []
    with open(ranking_path) as f:
        for line in tqdm(f, desc="Loading ranking data"):
            columns = line.strip().split("\t")
            rankings.append(columns)
            
    print(f"{len(rankings)} ranking datas are loaded\n")
    return rankings


def read_queries(queries_path):
    queries = []
    with open(queries_path) as f:
        for line in tqdm(f, desc="Loading query data"):
            columns = line.strip().split("\t")
            queries.append(columns)
    print(f"{len(queries)} queries are loaded\n")
    return queries


def find_model_path(model_root) -> str:
    # 모델 경로 찾기
    for root, dirs, files in os.walk(model_root):
        for file in files:
            if file == "model.safetensors":
                checkpoint_path = os.path.dirname(os.path.join(root, file))
                print("checkpoint_path: ", checkpoint_path)
                
    return checkpoint_path

def find_ranking_path(filename) -> str:
    # 랭킹 파일 경로 찾기
    path = "experiments/sc_documents"
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == filename:
                ranking_path = os.path.join(root, file)
                print("ranking_path: ", ranking_path)
                
    return ranking_path


def make_random_triples(doc_questions_path, output_queries_path, output_triples_path):
    random.seed(42)
    doc_questions_df = read_jsonl(doc_questions_path, to_csv=True)
    
    triples_data = []
    query_data = []
    q_idx = 0
    max_doc_len = len(doc_questions_df)-1

    for i in tqdm(range(len(doc_questions_df))):
        data = doc_questions_df.loc[i]
        
        for qusetion in data['questions']:
            query_data.append(qusetion)
            ran_doc_idx = random.randint(0, max_doc_len)
            
            # pos 문서가 아닌 neg 문서 고르기
            while ran_doc_idx == i: 
                ran_doc_idx = random.randint(0, max_doc_len)

            triples_data.append(f'{q_idx}, {i}, {ran_doc_idx}')
            q_idx += 1

    if not os.path.exists(os.path.dirname(output_queries_path)):
        os.makedirs(os.path.dirname(output_queries_path))
        
    with open(output_queries_path, 'w') as f:
        for i,item in enumerate(query_data):
            f.write(f'{i}\t{item}\n')
            
    if not os.path.exists(os.path.dirname(output_triples_path)):
        os.makedirs(os.path.dirname(output_triples_path))
        
    with open(output_triples_path, 'w') as f:
        for i,item in enumerate(triples_data):
            f.write(f'[{item}]\n')
            
    print(f"{len(query_data)} queries is saved at {os.path.abspath(output_queries_path)}")
    print(f"{len(triples_data)} triples is saved at {os.path.abspath(output_triples_path)}")
    

def make_HN_triples(ranking_path, queries_path, doc_questions_path, output_queries_path, output_HN_path, range_top=10, range_bottom=50):
    # range_top : neg 문서를 샘플링할 최고 랭크     **range_top < range_bottom
    # range_bottom : neg 문서를 샘플링할 최저 랭크
    random.seed(42)
    doc_questions_df = read_jsonl(doc_questions_path, to_csv=True)
    rankings = read_ranking(ranking_path) # retrieval 모델로 뽑은 query별 topk의 결과
    queries = read_queries(queries_path) # queries.tsv
    
    qid_to_topk = {} # qid to topk 결과
    for query in queries:
        qid_to_topk[int(query[0])] = []

    for rank_data in rankings:
        qid_to_topk[int(rank_data[0])].append(int(rank_data[1]))
    
    triples_data = []
    query_data = []
    q_idx = 0

    # negative 문서 선별
    for i in tqdm(range(len(doc_questions_df)), desc="making triples"):
        data = doc_questions_df.loc[i]
        
        for qusetion in data['questions']:
            top_samples = qid_to_topk[q_idx][range_top:range_bottom]
            
            if len(top_samples) >= range_bottom - range_top:
                query_data.append([q_idx, qusetion])
                
                ran_idx = random.randint(0, range_bottom - range_top - 1)
                neg_doc_id = top_samples[ran_idx]
                
                while neg_doc_id == i: 
                    ran_idx = random.randint(0, range_bottom - range_top - 1)
                    neg_doc_id = top_samples[ran_idx]
                    
                triples_data.append(f'{q_idx}, {i}, {neg_doc_id}')
                q_idx += 1
                
            else:
                q_idx += 1
                continue
    
    
    if not os.path.exists(os.path.dirname(output_queries_path)):
        os.makedirs(os.path.dirname(output_queries_path))
    
    with open(output_queries_path, 'w') as f:
        for i, item in query_data:
            f.write(f'{i}\t{item}\n')

    if not os.path.exists(os.path.dirname(output_HN_path)):
        os.makedirs(os.path.dirname(output_HN_path))
        
    with open(output_HN_path, 'w') as f:
        for i,item in enumerate(triples_data):
            f.write(f'[{item}]\n')
    
    print(f"{len(query_data)} HN queries is saved at {os.path.abspath(output_queries_path)}")
    print(f"{len(triples_data)} HN triples is saved at {os.path.abspath(output_HN_path)}")

    
def make_filtered_triples(doc_questions_path, ranking_path, queries_path, triples_path, output_queries_path, output_triples_path, topk_size=1):
    # topk_size: 검색결과 topk안에 pos 문서가 포함되지 않을 경우 필터링할 k의 크기

    doc_questions = read_jsonl(doc_questions_path)
    random.seed(42)
    qid_to_topk = {} # qid별 top k개의 문서id
    qid_to_q = {} # qid별 질의(str)
    with open(queries_path) as f:
        for line in tqdm(f, desc="Loading query data"):
            columns = line.strip().split("\t")
            qid_to_q[int(columns[0])] = columns[1]
            qid_to_topk[int(columns[0])] = []

    print(f"{len(qid_to_q)} queries are loaded\n")
    
    with open(ranking_path) as f:
        for line in tqdm(f, desc="Loading ranking data"):
            columns = line.strip().split("\t")
            
            # qid별 top k개의 문서id 추가
            if int(columns[2]) <= topk_size: 
                qid_to_topk[int(columns[0])].append(int(columns[1]))
        
    print(f"Top{topk_size} {len(qid_to_topk)} ranking datas are loaded\n")
        
        
    qid_to_pos = {} # qid별 정답 문서id
    q_id = 0
    for i, doc in enumerate(doc_questions):
        for question in doc['questions']:
            qid_to_pos[q_id] = i
            q_id+=1
            
    filtered_qid = [] # 필터링된 qid(질의와 문서간의 연관성이 높은 데이터들)
    for qid, topk in qid_to_topk.items():
        if qid_to_pos[qid] in topk:
            filtered_qid.append(qid)

    print(f"Top{topk_size} filtered reuslt")
    print(f"{len(qid_to_topk)} queries have been filtered to {len(filtered_qid)} queries\n")
    
    qid_to_triples = {}
    with open(triples_path) as f:
        for line in tqdm(f, desc="Triples datas Loading"):
            qid_to_triples[eval(line)[0]] = eval(line)
            
            
    filtered_triples = []
    filtered_queries = []
    for qid in filtered_qid:
        try:
            filtered_triples.append(qid_to_triples[qid])
            filtered_queries.append([qid, qid_to_q[qid_to_triples[qid][0]]])
        except:
            pass
        
    if not os.path.exists(os.path.dirname(output_queries_path)):
        os.makedirs(os.path.dirname(output_queries_path))
        
    with open(output_queries_path, 'w') as f:
        for i,item in filtered_queries:
            f.write(f'{i}\t{item}\n')
            
    
    print(f"{len(filtered_queries)} filtered queries are saved at {os.path.abspath(output_queries_path)}")
    
    
    if not os.path.exists(os.path.dirname(output_triples_path)):
        os.makedirs(os.path.dirname(output_triples_path))
        
    with open(output_triples_path, 'w') as f:
        for new_triple in filtered_triples:
            f.write(f'{new_triple}\n')
    
    print(f"{len(filtered_triples)} filtered triples are saved at {os.path.abspath(output_triples_path)}")
    
    
def make_rerank_triples(doc_questions_path, ranking_path, output_path, range_top=10, range_bottom=50, neg_size=40):
    # range_top : neg 문서를 샘플링할 최고 랭크     **range_top < range_bottom
    # range_bottom : neg 문서를 샘플링할 최저 랭크
    # neg_size : 샘플링할 neg 문서의 갯수   **range_bottom - range_top >= neg_size
    
    random.seed(42)
    doc_questions_df = read_jsonl(doc_questions_path)
    
    qid_to_q = {} # key: qid, value: 질의(str)
    did_to_d = {} # key: qid, value: 문서(str)
    qid_to_pos = {} # key: qid, value: 정답문서id
    qid_to_topk = {} # key: qid, value: qid에 대한 colbert 검색결과인 topk의 문서id 리스트

    q_id = 0
    for i, doc in enumerate(doc_questions_df):
        did_to_d[i] = doc['content']
        
        for question in doc['questions']:
            qid_to_q[q_id] = question
            qid_to_pos[q_id] = i
            qid_to_topk[q_id] = []
            q_id+=1
    
    print(f"{len(qid_to_q)} queries are loaded\n")
    
    rankings = []
    with open(ranking_path) as f:
        for line in tqdm(f, desc="Loading ranking data"):
            columns = line.strip().split("\t")
            qid = int(columns[0])
            did = int(columns[1])
            qid_to_topk[qid].append(did)
            
    rerank_triples = []
    for qid, topk in tqdm(qid_to_topk.items(), desc="Making rerank triples"):
        # top1의 검색결과가 정답이 아닌 데이터 필터링
        if topk[0] != qid_to_pos[qid]:
            continue
        
        # reranker 학습데이터 포멧
        rerank_data = {"query": qid_to_q[qid], "pos": [did_to_d[qid_to_pos[qid]]], "neg": []}
        
        sample_negs = random.sample(topk[range_top:range_bottom], neg_size)
        
        # neg에 pos문서가 들어간 경우 제외
        if qid_to_pos[qid] in sample_negs:
            sample_negs.remove(qid_to_pos[qid])
            
        for neg in sample_negs:
            rerank_data["neg"].append(did_to_d[neg])
            
        rerank_triples.append(rerank_data)
    
    print()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    save_to_jsonl(output_path, rerank_triples)
    print(f"{len(rerank_triples)} rerank triples is saved at {output_path}\n")