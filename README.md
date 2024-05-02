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
  * Github, WandB
- 의사소통
  * Slack, Zoom

### Requirements
```
pandas==2.1.4
numpy==1.23.5
wandb==0.16.1
tqdm==4.66.1
pytorch_lightning==2.1.2
transformers[torch]==4.35.2
rouge==1.0.1
jupyter==1.0.0
jupyterlab==4.0.9

```
## 1. Competiton Info

### Overview
#### **Dialogue Summarization 경진대회**

  주어진 데이터를 활용하여 일상 대화에 대한 요약을 효과적으로 생성하는 모델을 개발하는 대회

- **배경**
    - 일상생활에서 대화는 **항상** 이루어지고 있습니다. 회의나 토의는 물론이고, 사소한 일상 대화 중에도 서로 다양한 주제와 입장들을 주고 받습니다. 나누는 대화를 녹음해두더라도 대화 전체를 항상 다시 들을 수는 없기 때문에 요약이 필요하고, 이를 위한 통화 비서와 같은 서비스들도 등장하고 있습니다.
    - 그러나 하나의 대화에서도 관점, 주제별로 정리하면 수 많은 요약을 만들 수 있습니다. 대화를 하는 도중에 이를 요약하게 되면 대화에 집중할 수 없으며, 대화 이후에 기억에 의존해 요약하게 되면 오해나 누락이 추가되어 주관이 많이 개입되게 됩니다.
    - 이를 돕기 위해, 우리는 이번 대회에서 **일상 대화를 바탕으로 요약문을 생성하는 모델**을 구축합니다!

- **대회 목표**

    - 경진대회의 목표는 정확하고 일반화된 모델을 개발하여 요약문을 생성하는 것입니다. 나누는 많은 대화에서 핵심적인 부분만 모델이 요약해주니, 업무 효율은 물론이고 관계도 개선될 수 있습니다. 또한, 참가자들은 모델의 성능을 평가하고 대화문과 요약문의 관계를 심층적으로 이해함으로써 자연어 딥러닝 모델링 분야에서의 실전 경험을 쌓을 수 있습니다.
    - 본 대회는 결과물 csv 확장자 파일을 제출하게 됩니다.
        - input : 249개의 대화문
        - output : 249개의 대화 요약문

- **Evaluation**

  - 예측된 요약 문장을 3개의 정답 요약 문장과 비교하여 metric의 평균 점수를 산출합니다.
  - DialogSum 데이터셋은 Multi-Reference Dataset으로 multi-reference에 대한 average를 보는 것이 중요합니다. 
  - 본 대회에서는 ROUGE-1-F1, ROUGE-2-F1, ROUGE-L-F1, 총 3가지 종류의 metric으로부터 산출된 평균 점수를 더하여 최종 점수를 계산합니다.
  - 3개의 정답 요약 문장의 metric 평균 점수를 활용하기에 metric 점수가 100점이 만점이 아니며, 3개의 정답 요약 문장 중 하나를 랜덤하게 선택하여 산출된 점수가 약 70점 정도입니다.
  
### Timeline

- 24/03/08

  : 대회 시작
- 24/03/11 ~ 24/03/15
    - Baseline 코드를 통한 모델링 과정 이해
    - EDA를 통한 인사이트 도출
    - Tokenizer 수정 실험
    - 다양한 Model 사용 실험
- 24/03/16 ~ 24/03/19
    - Data Augmentation
    - Hyperparameter Tuning
    - 후처리를 통한 Score 향상 시도
- 24/03/20

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
