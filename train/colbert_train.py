import sys
sys.path.append("./")

import os
import json
import pandas as pd
from utils import file_control
from utils import inference

from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer
from colbert import Indexer
from colbert import Searcher


# Random 샘플링 triples
queries_path = "data/colbert_data/queries.tsv"
triple_path = "data/colbert_data/triples"

# Hard Negative triples
queries_neg_path = "data/colbert_data/queries_neg.tsv"
triple_neg_path = "data/colbert_data/triples_neg"

# 필터링된 triples
filtered_queries_path = "data/colbert_data/filtered_queries.tsv"
filtered_triples_path = "data/colbert_data/filtered_triples"

# docs_list로 collection.tsv를 대체합니다.
docs_path = "data/documents.jsonl"
docs = file_control.read_jsonl(docs_path)
docs_df = pd.DataFrame(docs)
docs_list = docs_df['content'].tolist()
docs_questions_path = "data/gpt_data/result/documents_questions_final04.jsonl"

# 평가데이터 불러오기
eval_list = file_control.read_jsonl("data/eval_query_gpt4.jsonl")



def training(main_path, model_name, checkpoint_path, triples, queries, collection):
    print("== Start Training ==")
    print("model_name: ", model_name)
    
    # Train
    with Run().context(RunConfig(nranks=1, experiment=main_path+"/"+model_name)):
        config = ColBERTConfig(
            bsize=32,
            root="experiments",
        )
        
        trainer = Trainer(
            triples=triples,
            queries=queries,
            collection=collection,
            config=config,
        )

        checkpoint_savepath = trainer.train(checkpoint=checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_savepath}...")


def indexing(main_path, model_root, index_name, collection):
    print("== Start Indexing ==")
    print("index_name: ", index_name)
    checkpoint_path = file_control.find_model_path(model_root)

    # Indexing 
    with Run().context(RunConfig(nranks=1, experiment=main_path)):
        config = ColBERTConfig(
            doc_maxlen=512,
            kmeans_niters=1,
            nbits=8,
            root="experiments",
        )
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)
        
def evaluate(main_path, index_name, eval_list, docs, search_size=10):
    print("== Start Evaluate ==")
    
    # 평가를 위한 데이터셋 만들기
    search_list = make_search_result_faiss(eval_list, docs, index_name, search_size, main_path)
    
    # 예시 출력
    # search_df = pd.DataFrame(search_list)
    # inference.print_details(search_df.loc[78], eval_list)
    
    # validation 셋으로 평가하기
    def keystoint(x):
        return {int(k): v for k, v in x.items()}

    with open("data/gt_eval_dict.json", 'r') as f:
        gt_eval_dict = json.load(f, object_hook=keystoint)
        
    mAP = inference.calc_map(gt_eval_dict, search_list[0:101])
    print("mAP Score: ", mAP)
    
    return search_list
    
def make_ranking(filename, queries, topk_size, main_path, index_name, model_root):
    print("== Start Rankings ==")
    checkpoint_path = file_control.find_model_path(model_root)
    
    # Ranking .tsv 만들기
    with Run().context(RunConfig(nranks=1, experiment=main_path)):
        config = ColBERTConfig(
            root="experiments",
            ncells=32,
            centroid_score_threshold=0.5,
            ndocs=4096,
        )

        searcher = Searcher(index=index_name, checkpoint=checkpoint_path, config=config)
        queries = Queries(queries)
        ranking = searcher.search_all(queries, k=topk_size)
        ranking.save(filename)


def make_search_result_faiss(eval_list, docs, index, size, main_path=None):
    docs_list = pd.DataFrame(docs)['content'].tolist()
    
    config = ColBERTConfig(
            root="experiments",
            ncells=32,
            centroid_score_threshold=0.4,
            ndocs=4096,
        )
    with Run().context(RunConfig(experiment=main_path)):
        searcher = Searcher(index=index, collection=docs_list, config=config, verbose=0)
    
    search_list = []
    for eval in eval_list:
        query = eval['query'] if eval['query'] else eval['msg'][-1]['content']
        result = {
            "eval_id": eval['eval_id'],
            "standalone_query": "",
            "topk": [],
            "answer": "",
            "references": [],
        }
        
        if eval['is_normal']:
            search_list.append(result)
            continue
        
        result['standalone_query'] = query
        search_result = searcher.search(query, k=size)

        # Print out the top-k retrieved passages
        for passage_id, passage_rank, passage_score in zip(*search_result):
            result["topk"].append(docs[passage_id]['docid'])
            result["references"].append({"score": passage_score, "content": searcher.collection[passage_id]})
            
        search_list.append(result)
        
    return search_list

    

if __name__ == "__main__":
    print("\n=== First Training (Random Triple) ===")
    # Random Triple 학습데이터 만들기
    file_control.make_random_triples(docs_questions_path, queries_path, triple_path)
    
    # Training(Random triple)
    main_path = "sc_documents"
    checkpoint_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_name = "multilingual-colbert_random"
    
    training(main_path, model_name, checkpoint_path, triple_path, queries_path, docs_list)
    
    # Indexing(Random triple)
    model_root = os.path.join("experiments", main_path, model_name)
    index_name = "multilingual-colbert-random-nbit8"
    
    # indexing(main_path, model_root, index_name, docs_list)
    
    # Evaluation(Random triple)
    print("Trained by random triples result")
    evaluate(main_path, index_name, eval_list, docs, 10)
    
    # Make Ranking .tsv file(Random triple)
    ranking_filename = "multilingual-colbert-random-nbit8.ranking.tsv"
    make_ranking(ranking_filename, queries_path, 100, main_path, index_name, model_root)
    
    
    
    print("\n\n=== Second Training (Hard Negative Triple) ===")
    
    # Ranking 경로 가져오기
    ranking_path = file_control.find_ranking_path("multilingual-colbert-random-nbit8.ranking.tsv")
    
    # Hard Negative Triple 학습데이터 만들기
    file_control.make_HN_triples(ranking_path, queries_path, docs_questions_path, 
                                queries_neg_path, triple_neg_path, 
                                range_top=10, range_bottom=50)
    
    # Training(Hard Negative triple)
    checkpoint_path = file_control.find_model_path(model_root)
    model_name = "multilingual-colbert_HN"
    
    training(main_path, model_name, checkpoint_path, triple_neg_path, queries_neg_path, docs_list)
    
    # Indexing(Hard Negative triple)
    model_root = os.path.join("experiments", main_path, model_name)
    index_name = "multilingual-colbert-HN-nbit8"
    
    indexing(main_path, model_root, index_name, docs_list)
    
    # Evaluation(Hard Negative triple)
    print("Trained by Hard Negative triples result")
    evaluate(main_path, index_name, eval_list, docs, 10)
    
    # Make Ranking .tsv file(Hard Negative triple)
    ranking_filename = "multilingual-colbert-HN-nbit8.ranking.tsv"
    make_ranking(ranking_filename, queries_path, 100, main_path, index_name, model_root)
    
    
    
    print("\n\n=== Third Training (Filtered Triple) ===")
    
    # Ranking 경로 가져오기
    ranking_path = file_control.find_ranking_path("multilingual-colbert-HN-nbit8.ranking.tsv")
    
    # Filtered 학습데이터 만들기
    file_control.make_filtered_triples(docs_questions_path, ranking_path,
                                        queries_neg_path, triple_neg_path,
                                        filtered_queries_path, filtered_triples_path, topk_size=1)
    
    # Training(Filtered triple)
    checkpoint_path = file_control.find_model_path(model_root)
    model_name = "multilingual-colbert_filter"
    
    training(main_path, model_name, checkpoint_path, filtered_triples_path, filtered_queries_path, docs_list)
    
    # Indexing(Filtered triple)
    model_root = os.path.join("experiments", main_path, model_name)
    index_name = "multilingual-colbert-filter-nbit8"
    
    indexing(main_path, model_root, index_name, docs_list)
    
    # Evaluation(Filtered triple)
    print("Trained by random triples result")
    search_list = evaluate(main_path, index_name, eval_list, docs, 10)
    
    file_control.save_to_jsonl("search_output/colbert_search_result.jsonl", search_list)
    print("Retrieval result is saved")
    
    # Make Ranking .tsv file(Filtered triple)
    # Reranker 모델의 데이터셋 구축용
    ranking_filename = "multilingual-colbert-filter-nbit8.ranking.tsv"
    make_ranking(ranking_filename, queries_path, 100, main_path, index_name, model_root)
    
    
    