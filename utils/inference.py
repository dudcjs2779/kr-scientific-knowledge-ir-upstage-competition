import sys
sys.path.append("/root/rag_project")

import pandas as pd
import re

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

def print_details(df, eval_df):
    if isinstance(df, pd.core.series.Series):
        if not isinstance(eval_df, pd.core.frame.DataFrame):
            eval_df = pd.DataFrame(eval_df)
            
        eval_id = df['eval_id']
        user_msg = eval_df.loc[eval_df['eval_id'] == eval_id, 'msg'].values[0]
        standalone_query = df['standalone_query']
        answer = df['answer']
        doc_ids = df['topk']
        references = df['references']
        
        print(f"## eval_id: {eval_id}")
        print(f"user_msg: {user_msg}")
        print(f"standalone_query: {standalone_query}")
        print(f"answer: {answer}\n")
        
        for idx in range(len(references)):
            print(f"{doc_ids[idx]}: {references[idx]['score']}")
            
            content = references[idx]['content']
            content = re.sub("\.", ".\n", content)
            print(content)
        
    else:
        df = df.reset_index(drop=True)
        for i in range(0, len(df)):
            eval_id = df.loc[i, 'eval_id']
            standalone_query = df.loc[i, 'standalone_query']
            answer = df.loc[i, 'answer']
            doc_ids = df.loc[i, 'topk']
            references = df.loc[i, 'references']
            
            print(f"## eval_id: {eval_id}")
            print(f"standalone_query: {standalone_query}")
            print(f"answer: {answer}\n")
            
            for idx in range(len(references)):
                print(f"{doc_ids[idx]}: {references[idx]['score']}")
                
                content = references[idx]['content']
                content = re.sub("\.", ".\n", content)
                print(content)
                
            print("="*50)
            
def print_details2(data, docs_df):
    eval_id = data['eval_id']
    query = data['standalone_query']
    pred_ids = data['topk']
    references = data['references']
    msg = data['msg']
    label_ids = data['doc_ids']
    score = data['score']

    print(f"## eval_id: {eval_id}")
    print(f"user msg: {msg}")
    print(f"standalone_query: {query}")
    print(f"score: {score}")
    print(f"정답 문서의 갯수: {len(label_ids)}")

    print("="*20 + " PRED "+"="*20)
    for idx in range(len(pred_ids)):
        star_str = ""
        if pred_ids[idx] in label_ids:
            star_str = "✅"
        print(f"*{star_str} {idx+1} {pred_ids[idx]}: {references[idx]['score']}")
        content = references[idx]['content']
        content = re.sub("\.", ".\n", content)
        print(content)
        
    print("="*20 + " GT "+"="*20)
    for idx in range(len(label_ids)):
        star_str = ""
        if label_ids[idx] in pred_ids:
            star_str = "✅"
        print(f"*{star_str} {label_ids[idx]}")
        content = docs_df[docs_df['docid'] == label_ids[idx]]['content'].values[0]
        content = re.sub("\.", ".\n", content)
        print(content)
            
def calc_map(gt, pred, get_each_result=False):
    precision_list = []
    sum_average_precision = 0
    for j in pred:
        if gt[j["eval_id"]]:
            hit_count = 0
            sum_precision = 0
            for i,docid in enumerate(j["topk"][:3]):
                if docid in gt[j["eval_id"]]:
                    hit_count += 1
                    sum_precision += hit_count/(i+1)
            average_precision = sum_precision / hit_count if hit_count > 0 else 0
        else:
            average_precision = 0 if j["topk"] else 1
        sum_average_precision += average_precision
        
        if get_each_result:
            precision_list.append(average_precision)
            
    if get_each_result:
        print("mAP Score: ", sum_average_precision/len(pred))
        return precision_list
    else:
        return sum_average_precision/len(pred)
    

