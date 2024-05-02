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

- 24/05/02
  : 대회 종료

## 2. Components

### Directory

```
├── code
│   ├── baseline.ipynb
│   ├── baseline_optuna.ipynb
│   └── baseline_with_topic.ipynb
├── configs
│   ├── config.yaml
│   ├── config.yaml
│   ├── config.yaml
│   └──paper 
└── data
│   ├── train.csv
│   ├── dev.csv
│   ├── test.csv
│   ├── sample_submission.csv
├── results
│   └──checkpoint
│   │   ├── checkpoint-1750
│   │   ├── checkpoint-2000
│   │   ├── checkpoint-2250
│   │   ├── checkpoint-2500
│   └──csv
│   │   ├── output.csv

```

## 3. Data descrption

### Dataset overview
제공되는 데이터셋은 오직 **"** **대화문과 요약문** **"** 입니다. 회의, 일상 대화 등 다양한 주제를 가진 대화문과, 이에 대한 요약문을 포함하고 있습니다.
  
  <img src="https://aistages-api-public-prod.s3.amazonaws.com/app/Files/38e20522-3af8-438f-8039-c5547212b8db.png" height="150px" width="500px">

- 데이터 정보
    - train : 12457
    - dev : 499
    - test : 250
    - hidden-test : 249
      
- 데이터 예시

  <img src="https://aistages-api-public-prod.s3.amazonaws.com/app/Files/c0c1a6e2-6fa1-448e-9fc7-f9db45701022.png" height="200px" width="600px">
    
    - fname : 대화 고유번호 입니다. 중복되는 번호가 없습니다.
    - dialogue : 최소 2명에서 최대 7명이 등장하여 나누는 대화 내용입니다. 각각의 발화자를 구분하기 위해#Person”N”#: 을 사용하며, 발화자의 대화가 끝나면 \n 으로 구분합니다. 이 구분자를 기준으로 하여 대화에 몇 명의 사람이 등장하는지 확인해보는 부분은 [EDA](https://colab.research.google.com/drive/1O3ZAcHR9q7dccasRcxvNhCZD-gIlasGV#scrollTo=usQutfBFqtuk)에서 다루고 있습니다.
    - summary : 해당 대화를 바탕으로 작성된 요약문입니다.
### Data Processing

#### 개인정보 마스킹
```
개인정보 내역을 마스킹함
전화번호 → #PhoneNumber#
주소 → #Address#
생년월일 → #DateOfBirth#
여권번호 → #PassportNumber#
사회보장번호 → #SSN#
신용카드 번호 → #CardNumber#
차량 번호 → #CarNumber#
이메일 주소 → #Email#
```
* 마스킹 후 tokenizer에 추가

### Data Augmentation

#### 방법 1
- Solar API 사용
- 데이터에서 summary 부분만 augmentation을 진행한 후 이를 학습 데이터에 추가
- 모델이 다양한 summary 스타일에 더 잘 적응하도록 만들기 위해, summary만 다르게 하고 dialogue는 기존의 것을 재사용
![열쇠를 찾았습니다](https://github.com/UpstageAILab/upstage-nlp-summarization-nlp6/assets/88610815/d8deeb84-006a-44a4-8b71-e4f68dabe5d8)
![ZWJQ4oZKVvtWSN-Hn5VxwgdhKZDbgL8MTlahURhNGgN0u6pRWaNdRshhd0YoLTlPSbqvuIOCqM6tW-3VX7XWnpBonxgx8j1SPO0-dqQ-MAxsWeCjl7E_AnIyoyrX](https://github.com/UpstageAILab/upstage-nlp-summarization-nlp6/assets/88610815/20f1cabd-abc8-4abc-93f4-9241f2a079b2)


#### 방법 2
- Solar API, GPT 4 사용
- 다양한 주제로 구성된 새로운 대화 데이터와 요약문을 생성
![uab5WiMPKaFBL2xHvDSF40DBqGH9rzVwWQZSqTUqBbUJlemWse_W8KhEEWqaFRkN31KRpO5nca7Fx2P6s4esq-Qw8wP3SY4tqIiAqc8_ahBOJvmgVubIr44Idv8o](https://github.com/UpstageAILab/upstage-nlp-summarization-nlp6/assets/88610815/ed055c9c-220e-4ba2-b3d5-7b31c8acb9ef)
![oDZLRimvItYX2eCXN1fJwyZooaDzRG7sPHE083jdeaJmsVDuOgN5mXliZZDMCj30Iz_qoJAclOmOxFl26_TSxk-Cd55H9co1CMQm1ZBtNz6h1sZQqKWI3_yInFmY](https://github.com/UpstageAILab/upstage-nlp-summarization-nlp6/assets/88610815/d2171a2d-85eb-4c09-8560-4fa230eca81e)


## 4. Modeling

### Model descrition

* Baseline 모델인 Kobart-Summarization 사용
* KoT5 모델 사용을 시도 했으나, Batch Size 이슈로 인해 성공하지 못함

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

[Presentation File Link](https://docs.google.com/presentation/d/1M8d52Sqx7eN_8Yk7zubBvoJE2KNa5dOHuB8cmO-9UBo/edit?usp=sharing)

## etc

### Meeting Log

[Meeting Log](https://www.notion.so/6-1d68ac745f7f4428a014d7e36824f567?pvs=4#9927d4c1f7704322b0cfad4d131af741)

### Reference

* 사용 모델 : Kobart Summarization

  [HuggingFace](https://huggingface.co/gogamza/kobart-summarization)
