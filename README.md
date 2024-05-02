# Diaglogue Summarization | 일상 대화 요약
## Team NLP 6조

| ![김영천](https://github.com/dudcjs2779.png) | ![김윤겸](https://github.com/gyeom-yee.png) | ![김하연](https://github.com/210B.png) | ![남영진](https://github.com/NamisMe.png) | ![이소영](https://github.com/Leesoyeong.png) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김영천](https://github.com/dudcjs2779)               |            [김윤겸](https://github.com/gyeom-yee)             |            [김하연](https://github.com/devguno)             |            [남영진](https://github.com/NamisMe)             |            [이소영](https://github.com/Leesoyeong)             |
|                         Data-Centric, 후처리                        |                            Modeling, 후처리                      |                         Modeling, Augmentation                 |                            Modeling, 전처리                      |                       Modeling,  hyperparameter tuning         |

## 0. Overview
### Environment
- 컴퓨팅 환경
    - 서버를 VSCode와 SSH로 연결하여 사용 
    - NVIDIA GeForce RTX 3090 | 24GB
    - CUDA Version 12.2
- 협업환경
  * Github 
- 의사소통
  * Slack, Zoom

### Requirements
```
colbert-ir                0.2.14
elasticsearch             8.8.0
faiss-gpu                 1.7.2
Flask                     3.0.3
openai                    1.7.2
pandas                    2.2.2
tokenizers                0.15.1
torch                     1.13.1
torchaudio                2.1.0
torchelastic              0.2.2
transformers              4.37.2
```
## 1. Competiton Info

### Overview
#### **Scientific Knowledge Question Answering | 과학 지식 질의 응답 시스템 구축**

질문과 이전 대화 히스토리를 보고 참고할 문서를 검색엔진에서 추출 후 이를 활용하여, 질문에 적합한 대답을 생성하는 태스크
- **배경**
    - LLM의 한계
        - Knowledge cutoff , Hallucination

    - 이를 극복하기 위해 RAG(Retrieval Augmented Generation) 사용
        - 검색엔진 : 적합한 레퍼런스 추출
        - LLM : 답변 생성

    - 문제 해결 과정
        - 과학 상식 문서 4200여개를 미리 검색엔진에 색인
        - 대화 메시지 / 질문이 들어오면 과학 상식에 대한 질문 의도인지 아닌 지 판단
            - 과학상식 질문인 경우 → 적합한 문서 추출 후 답변 생성
            - 과학상식 문서가 아닌 경우 → 검색엔진 활용 필요 없이 적절한 답 생성

- **대회 목표**

    - RAG 시스템의 개발
    - 검색엔진이 올바른 문서를 색인했는지, 생성된 답변이 적절한지 직접 확인
    - 최종 출력은 문서 추출이 아니라, 질문에 대한 답변을 생성하는 것

- **Evaluation**
    - RAG에 대한 End-to-End 평가 대신, 적합한 레퍼런스를 얼마나 잘 추출했는지에 대한 평가만 진행
    - **Mean Average Precision(MAP)**
        - 상위 N개의 관련된 문서에 대해 전부 점수를 반영할 수 있되, 관련된 문서의 위치에 따라 점수에 차등을 줄 수있는 평가 모델
          ![image](https://github.com/UpstageAILab/upstage-ai-final-ir3/assets/88610815/28e68e47-aa91-4d38-9c43-900c1b8345f8)
        - 코드
            ```python
            def calc_map(gt, pred):    
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
                return sum_average_precision/len(pred)
            ```
          - 과학 상식 질문이 아닌, 즉 검색이 필요 없는 ground truth 항목
              -  검색 결과 없는 경우 : 1점
              -  검색 결과 있는 경우 : 0점

  
### Timeline
- Pre-Competition (04/08 ~ 04/21)
  - 1주차 (04/08 ~ 04/12) : Information Retrieval 강의 수강
  - 2주차 (04/15 ~ 04/19) : Baseline 코드 이해 및 기초 Data Processing
      - Baseline Code 실행을 통한 이해
          - Baseline Code 구조
            ```plain text
            ├─code (폴더)
            │  │  README.md (실행 방법 설명)
            │  │  requirements.txt
            │  │  install_elasticsearch.sh (Elasticsearch 설치 및 client 사용을 위한 password 초기화)
            │  │  rag_with_elasticsearch.py (검색엔진 색인, RAG 구현 및 평가 진행 코드)
            │  │  sample_submission.csv (베이스라인 코드에 대한 평가 결과)
            │  │  requirements.txt (필요한 python package)
            ```
      -  Validation 셋 구축
      -  일상대화 분류 Prompting
      -  학습데이터 구축(GPT 3.5)

- 24/04/22
  : 대회 시작

- 24/04/22 ~ 24/04/25 : 대회 전략 수립 및 실행
    - Retrieval Modeling (By Colbert)
    - Hard Negative
    - Data Filtering
    - 과학 상식 / 일반 대화 분류
    - 솔루션 제시
        - 프롬프트 '질문 재읽기' 전략 
        - 모델링 사이즈를 줄이기 위한 임베딩 양자화

- 24/04/29 ~ 24/05/02
    - ReRanking

- 24/05/02 19:00
  : 대회 종료

## 2. Components
### Directory

```
├─code
│  │  README.md
│  │  requirements.txt
│  │  install_elasticsearch.sh
│  │  rag_with_elasticsearch.py
│  │  sample_submission.csv
│  │  requirements.txt
└── data
│   ├── documents.jsonl
│   ├── eval.jsonl
```

## 3. Data descrption
### Dataset overview

- **대회 데이터셋 License**
    - CC-BY-SA 4.0
        - 원본데이터 : ARC, MMLU
            - https://paperswithcode.com/dataset/arc
            - https://paperswithcode.com/dataset/mmlu
        - 저작권자 : Upstage AI LAB
- **학습데이터**
    - 과학 상식 정보를 담고 있는 순수 색인 대상 문서 4200여개
    - 문서의 예시는 아래와 같습니다.
        
        ```
        {"docid": "42508ee0-c543-4338-878e-d98c6babee66", "src": "ko_mmlu__nutrition__test", "content": "건강한 사람이 에너지 균형을 평형 상태로 유지하는 것은 중요합니다. 에너지 균형은 에너지 섭취와 에너지 소비의 수학적 동등성을 의미합니다. 일반적으로 건강한 사람은 1-2주의 기간 동안 에너지 균형을 달성합니다. 이 기간 동안에는 올바른 식단과 적절한 운동을 통해 에너지 섭취와 에너지 소비를 조절해야 합니다. 식단은 영양가 있는 식품을 포함하고, 적절한 칼로리를 섭취해야 합니다. 또한, 운동은 에너지 소비를 촉진시키고 근육을 강화시킵니다. 이렇게 에너지 균형을 유지하면 건강을 유지하고 비만이나 영양 실조와 같은 문제를 예방할 수 있습니다. 따라서 건강한 사람은 에너지 균형을 평형 상태로 유지하는 것이 중요하며, 이를 위해 1-2주의 기간 동안 식단과 운동을 조절해야 합니다."}
        {"docid": "7a3e9dc2-2572-4954-82b4-1786e9e48f1f", "src": "ko_ai2_arc__ARC_Challenge__test", "content": "산꼭대기에서는 중력이 아주 약간 변합니다. 이는 무게에 영향을 미칩니다. 산꼭대기에서는 무게가 감소할 가능성이 가장 높습니다. 중력은 지구의 질량에 의해 결정되며, 산꼭대기에서는 지구의 질량과의 거리가 더 멀어지기 때문에 중력이 약간 감소합니다. 따라서, 산꼭대기에서는 무게가 더 가볍게 느껴질 수 있습니다."}
        ```
        
        - 'doc_id' : 문서별 id(uuid)
        - 'src' : 출처
            - 데이터를 Open Ko LLM Leaderboard에 들어가는 Ko-H4 데이터 중 MMLU, ARC 데이터를 기반으로 생성했기 때문에 출처도 두가지 카테고리를 가짐
        - 'content' : 실제 RAG에서 레퍼런스로 참고할 지식 정보
    - 파일 포맷
        - 각 line이 json인 jsonl 파일
      
- **평가데이터**
    - 200개 : 과학상식 대화
        
        ```
        {"eval_id": 0, "msg": [{"role": "user", "content": "나무의 분류에 대해 조사해 보기 위한 방법은?"}]}
        {"eval_id": 1, "msg": [{"role": "user", "content": "각 나라에서의 공교육 지출 현황에 대해 알려줘."}]}
        {"eval_id": 3, "msg": [{"role": "user", "content": "통학 버스의 가치에 대해 말해줘."}]}
        {"eval_id": 4, "msg": [{"role": "user", "content": "Dmitri Ivanovsky가 누구야?"}]}
        ```
        
    - 20개 : 멀티턴 대화
            
        ```
        {"eval_id": 2, "msg": [{"role": "user", "content": "기억 상실증 걸리면 너무 무섭겠다."}, {"role": "assistant", "content": "네 맞습니다."}, {"role": "user", "content": "어떤 원인 때문에 발생하는지 궁금해."}]}
        ```
            
    - 20개 : 일반적인 대화 메시지
        
        ```
        {"eval_id": 36, "msg": [{"role": "user", "content": "니가 대답을 잘해줘서 너무 신나!"}]}
        ```
        - eval_id : 평가 항목 ID
        - msg :  user와 assistant 간 대화 메시지 (리스트)
### Data Processing

#### 1.일상대화 분류 Prompting
- 분류 프롬프트
    ```python
    query_generate_prompt = """
    ## Role: 검색 Query 생성기
    
    ## Instruction
    - 한국어로 답변을 생성해줘.
    """
    
    # tools를 사용하여 분류합니다. "query" 변수에 검색에 이용할 쿼리를 기본적으로 생성하고 
    # is_normal_conversation 변수에는 사용자의 질의가 일상 대화인지 아닌지를 구분해 일상 대화면 1 아니면 0을 입력하도록 합니다.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_query",
                "description": "사용자의 대화 내역이 자연과학, 사회과학, 컴퓨터공학, 코딩, 수학, 의학, 정치, 사회, 지리, 경제, 일반상식 등의 지식을 요구하는 질문일 경우 검색에 사용할 query를 생성한다.",
                "parameters": {
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "사용자와의 대화를 기반으로 적절한 검색 쿼리를 간결하게 생성한다."
                        },
                        "is_normal_conversation": {
                            "type": "string",
                            "description": "사용자의 대화 내역이 자연과학, 사회과학, 컴퓨터공학, 코딩, 수학, 의학, 정치, 사회, 지리, 경제, 일반상식 등의 지식을 요구하는 질문일 경우 0 아니면 1을 입력한다."
                        }
                    },
                    "required": ["query", "is_normal_conversation"],
                    "type": "object"
                }
            }
        },
    ]림
    ```
- tools를 이용하여 분류한 결과, 2개 데이터를 제외하고 모두 잘 분류
#### 2.학습 데이터 구축(GPT 3.5)
- 키워드 추출

```python
augment_instruct_test = """
## Role
키워드 생성기

## Instructions
- 주어진 내용을 보고 중요한 키워드만 추출하거나 생성한다.
- 내용에는 없지만 관련성이 높은 키워드를 생성한다.
- 중요한 키워드를 앞쪽에 배치한다.
- JSON 포맷으로 키워드를 생성한다.

## Content
%s

## Output format
{"keywords": [$word1, $word2, $word3, ...]}

""" % (docs_df[ran]["content"])
```

- 질문생성

```python
augment_instruct = """
## Role
질문 생성기

## Instructions
- 주어진 키워드를 참고하여 질문을 5개 생성해줘.
- 주어진 레퍼런스 정보를 보고 이 정보가 도움이 될만한 질문을 생성해줘.
- 앞쪽에 위치한 키워드들은 반드시 포함하여 질문을 생성해줘.
- 최대한 다양한 질문을 생성해줘.
- 질문은 한문장으로 간결하게 구성해줘.
- 한국어로 질문을 생성해줘.
- 아래 JSON 포맷으로 생성해줘.

## Output format
{"questions": [$question1, $question2, $question3, $question4, $question5]}
""" 

user_input = """
## Keywords
%s

## Content
%s
"""% (documents_keywords[ran]['keywords'], documents_keywords[ran]['content'])
```

- 하나의 문서에 대해서 다양한 관점의 질의를 생성하기 위해서 키워드 추출 과정을 포함시켜 질의를 생성하여 학습 데이터를 구축하였습니다.  

#### 3.Validation Set 구축
- eval.jsonl 평가 데이터의 앞에서부터 101개의 데이터에 대해서 정답 문서들을 눈으로보고 라벨링하여 validation 셋을 구축
- 구축방법
    - 한국어 Dense Rretrieval 모델 1개 다국어 Dense Rretrieval 모델 1개 와 Sparse Rretrieval 모델 1개를 활용하여 각각의 top 10 결과를 보며 그럴듯한 문서를 정답으로 라벨링함.
    - Retrieval 모델로 그럴듯한 문서를 찾지 못한 경우 GPT나 인터넷 검색을 통해 적당한 Query를 재생성하여 재검색


## 4. Solutions

### Prompt Engineering



### Retrieval Modeling

#### 1.PLM sentence transformers + colbert train
#### 2.Hard Negative
#### 3.Data Filtering


### Reranking
- 이미 한국어 데이터 사전학습된 Reranker 모델을 이용해서 현재 가장 성능이 좋은 Retrieval 모델의 top10의 결과를 Reranking하였습니다.
    - [HuggingFace : Dongjin-kr/ko-reranker](https://huggingface.co/Dongjin-kr/ko-reranker)
    - BAAI/bge-reranker-larger 기반 한국어 데이터에 대한 fine-tuned model
- 결과
    - Pre-reranking : 0.7258
    - Top 10 reranking 0.8705
    - Top 100 reranking 0.8379

## 5. Result

### Leader Board

- **Public**
(수정필요)
![image](https://github.com/UpstageAILab/upstage-nlp-summarization-nlp6/assets/88610815/1fc641ea-7d93-4ce0-bc31-0577333a2731)

- **Private**
(수정필요)
![image](https://github.com/UpstageAILab/upstage-nlp-summarization-nlp6/assets/88610815/109bd134-3e85-40ba-badd-25243d0b2d02)

### Presentation

[Presentation File Link](https://docs.google.com/presentation/d/1vc2XwqVCUI35oBOTre9Nv2zSXMWpF8Ml8EmRQpalxF0/edit?usp=sharing)

## etc

### Meeting Log

[Meeting Log](https://www.notion.so/b13e70576943436bbe4d9d5791ff1a2b?pvs=21)

### Reference

- 논문
    1. [Large Language Models for Information Retrieval: A Survey(Yutao Zhu et al., 2024)](https://file.notion.so/f/f/1cc778c9-adf3-4dc6-9cdd-a46162c29bd7/c5f3d436-0523-4c6a-b983-1e65df8653e4/Large_Language_Models_for_Information_Retrieval-_A_Survey.pdf?id=3b174d97-6da7-49d8-b4a6-412775bda582&table=block&spaceId=1cc778c9-adf3-4dc6-9cdd-a46162c29bd7&expirationTimestamp=1714716000000&signature=G-yCE2GG51gXHvKdQi7jOLdiIBk4Httjgq57UT81PgQ&downloadName=Large+Language+Models+for+Information+Retrieval-+A+Survey.pdf)
    2. [ConTextual Masked Auto-Encoder for Dense Passage Retrieval(Xing Wu et al., 2022)](https://arxiv.org/abs/2208.07670)
    3. [Lost in the Middle: How Language Models Use Long Contexts(Nelson F.Liu et al., 2023)](https://arxiv.org/abs/2307.03172)
- 블로그 / Github
    1. [한국어 Reranker를 활용한 검색 증강 생성(RAG) 성능 올리기](https://aws.amazon.com/ko/blogs/tech/korean-reranker-rag/)
    2. [Korean-Reranker-Git](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/30_fine_tune/reranker-kr)
    3. [Advanced RAG와 Reranker](https://velog.io/@mmodestaa/Advanced-RAG%EC%99%80-Reranker)
    
