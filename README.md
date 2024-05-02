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
    - ReRanker Finetuning

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

#### 1.학습 데이터 구축(GPT 3.5)
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

#### 2.Validation Set 구축
- eval.jsonl 평가 데이터의 앞에서부터 101개의 데이터에 대해서 정답 문서들을 눈으로보고 라벨링하여 validation 셋을 구축
- 구축방법
    - 한국어 Dense Rretrieval 모델 1개 다국어 Dense Rretrieval 모델 1개 와 Sparse Rretrieval 모델 1개를 활용하여 각각의 top 10 결과를 보며 그럴듯한 문서를 정답으로 라벨링함.
    - Rretrieval 모델로 그럴듯한 문서를 찾지 못한 경우 GPT나 인터넷 검색을 통해 적당한 Query를 재생성하여 재검색

#### 3.일상대화 분류 Prompting

## 4. Modeling

### Retrieval Modeling

### Modeling Process
#### 1) Model-Centric : Hyperparamter Tuning 

* Grid Search / Random Search 사용

#### 2) Data-Centric : Topic Modeling
- Train Dataset의 topic 컬럼 학습
  ![Train example](https://github.com/UpstageAILab/upstage-nlp-summarization-nlp6/assets/78156719/9d5dc355-6a6f-4cdb-8819-be038aebde9f)
- 기존 Topic 모델(X) / Language 모델(O) 
  - 데이터셋의 topic 특성 상 LDA를 사용한 Extractive 방식이 아닌 Abtractive 방식 필요
  - KoBART-summarization을 사용하는 베이스라인 그대로 기용
    - 요약문 생성, Decoder 길이 제한(generate_max_length=6, decoder_max_len=6)
    - 타겟을 summary가 아닌 topic으로 학습 및 예측

      <img src=https://github.com/UpstageAILab/upstage-nlp-summarization-nlp6/assets/78156719/0f817079-6e74-45eb-b04a-4742e9778345 style="border:1px solid;">
  - 메타 데이터 학습을 위해 topic과 dialogue를 합친 dialogue_with_topic 컬럼 추가 후 해당 컬럼으로 요약문 학습 및 예측
    <img src=https://github.com/UpstageAILab/upstage-nlp-summarization-nlp6/assets/78156719/50fb9867-155e-4aa4-8071-72a4efb26e45 style="border:1px solid;">
  - topic 처리를 위한 Special token 추가: #topcic#
- Topic 추가 학습 실행 결과 성능이 향상됨✅

![image](https://github.com/UpstageAILab/upstage-nlp-summarization-nlp6/assets/78156719/f131e0cd-be97-4e77-aaad-b73121bf12ea)
![image](https://github.com/UpstageAILab/upstage-nlp-summarization-nlp6/assets/78156719/fded8d72-e903-4b89-b763-74c43a4b31b5)


## 5. Result

### Leader Board

- **Public**

![image](https://github.com/UpstageAILab/upstage-nlp-summarization-nlp6/assets/88610815/1fc641ea-7d93-4ce0-bc31-0577333a2731)

- **Private**

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
- 블로그
    1. [한국어 Reranker를 활용한 검색 증강 생성(RAG) 성능 올리기](https://aws.amazon.com/ko/blogs/tech/korean-reranker-rag/)
    2.    
    3.
