# upstage-nlp-summarization
Dialogue Summarization | 일상 대화 요약

# 1. 경진대회 소개

## 1.1. 개요

- Natural Language Process 대화문 요약 대회

> Dialogue Summurization 대회로 일상 대화로 이루어진 **대화문의 요약을 생성**하는 대회
> 

## 1.2. 일정

- 대회 시작일: 2024.03.08
- 최종 제출 마감 기한: 2024.03.20 19:00

## 1.3. 평가 지표

- **ROUGE-F1** - ROUGE-Recall과 ROUGE-Precisioin의 조화 평균
    - **ROUGE**는 텍스트 요약, 기계 번역과 같은 태스크를 평가하기 위해 사용되는 대표적인 metric으로, 모델이 생성한 요약본 혹은 번역본을 사람이 만든 참조 요약본과 비교하여 점수를 계산
        - ROUGE-Recall: 참조 요약본을 구성하는 단어들 중 모델 요약본의 단어들과 얼마나 많이 겹치는지 계산한 점수
        - ROUGE-Precision: 모델 요약본을 구성하는 단어들 중 참조 요약본의 단어들과 얼마나 많이 겹치는지 계산한 점수
    - **ROUGE-N**과 **ROUGE-L**은 비교하는 단어의 단위 개수를 어떻게 정할지에 따라 구분
        - ROUGE-N은 unigram, bigram, trigram 등 문장 간 중복되는 n-gram을 비교하는 지표
            - **ROUGE-1**는 모델 요약본과 참조 요약본 간에 겹치는 **unigram**의 수를 비교
            - **ROUGE-2**는 모델 요약본과 참조 요약본 간에 겹치는 **bigram**의 수를 비교
        - **ROUGE-L**: LCS 기법을 이용해 최장 길이로 매칭되는 문자열을 측정, n-gram에서 n을 고정하지 않고, 단어의 등장 순서가 동일한 빈도수를 모두 세기 때문에 보다 유연한 성능 비교가 가능

## 1.4. 데이터 설명

- 데이터 건수
모든 데이터는 .csv 형식으로 제공되고 있으며, 각각의 데이터 건수는 다음과 같다.
    - train: 12457
    - dev: 499
    - test: 499 (open: 250, hidden: 249)
- 데이터 구성
데이터는 아래와 같은 형태이며, 최소2턴, 최대 60턴으로 대화가 구성되어 있다. 대화(*dialogue)를 보고 이에 대한 요약(*summary) 를 예측하는 것이 최종 목표입니다.
    - fname : 대화 고유번호 입니다. 중복되는 번호가 없다.
    - dialogue : 최소 2명에서 최대 7명이 등장하여 나누는 대화 내용입니다. 각각의 발화자를 구분하기 위해#Person”N”#: 을 사용하며, 발화자의 대화가 끝나면 “\n” 으로 구분합니다. 이 구분자를 기준으로 하여 대화에 몇 명의 사람이 등장하는지 확인해보는 부분은 EDA 에서 다루고 있다.
    - 대화문에 존재하는 개인정보(예: 전화번호, 주소 등)는 다음과 같이 마스킹되어 있다.
        - 예) 전화번호 -> #PhoneNumber#
    - summary : 해당 대화를 바탕으로 작성된 요약문입니다.
    
    ![data_structue.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/ae8e86a0-d355-4f13-82a2-ab52e3ee4042/data_structue.png)
    

# 2.  경진대회 수행 절차 및 방법

## 2. 1. 환경

- **컴퓨팅 환경:** 인당 RTX 3090 서버를 VSCode와 SSH로 연결하여 사용
- **협업 환경:** Github, Wandb
- **의사 소통:** Slack, Zoom

## 2.1. 수행 절차

- Step 1: 유사 경진대회 분석 및 인사이트 도출
- Step 2: EDA
- Step 3: Data Preprocessing & Data Cleansing
- Step 4: Modeling - 모델 선택, Hyper-parameter Tuning을 통한 성능 실험
- Step 5: 학습된 모델 성능을 기반으로 NLP 파이프라인 반복
- Step 7: 최종 제출 파일 선택

## 2.2. 수행 방법

- 매일 Zoom 팀 미팅을 통해 진행상황 및 아이디어 공유
- Github repository를 사용해 작업 코드 공유
- WandB(https://wandb.ai/sycj427/aistages-nlp)를 사용한 실험 결과 기록
- Slack을 통한 실시간 의견 교류

# 3. 경진대회 수행 과정

## 3.1. EDA

### 3.1.1. 텍스트 길이 확인

- 학습 데이터
    - Dialogue Mean Length: 438.77
    - Summary Mean Length: 87.40
- 검증 데이터
    - Dialogue Mean Length: 432.56
    - Summary Mean Length: 81.71
- 평가 데이터
    - Dialogue Mean Length: 449.31

### 3.1.2. 토큰 개수 (kobart tokenizer 기준) 분포 파악

- 학습 데이터
    - Dialogue
        - 최대 토큰 개수: 950
        - 최소 토큰 개수: 32
        - 평균 토큰 개수: 158.60
    - Summary
        - 최대 토큰 개수: 165
        - 최소 토큰 개수: 7
        - 평균 토큰 개수: 30.98
        
        ![Train Dialogue Token Count.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/315892e6-b401-45f2-af66-0d0db4bb4ef0/Train_Dialogue_Token_Count.png)
        
        ![Train Summary Token Count.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/4104d7b0-8d98-4cf7-aba5-7297ff3c32c1/Train_Summary_Token_Count.png)
        
- 검증 데이터
    - Dialogue
        - 최대 토큰 개수: 620
        - 최소 토큰 개수: 46
        - 평균 토큰 개수: 156.69
    - Summary
        - 최대 토큰 개수: 89
        - 최소 토큰 개수: 9
        - 평균 토큰 개수: 28.80
        
        ![Validation Dialogue Token Count.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/a417794d-5cc5-4dca-911a-4b5a438cb6de/Validation_Dialogue_Token_Count.png)
        
        ![Validation Summary Token Count.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/8cf4d4e1-5450-46da-a6ed-a4a9b5587af4/Validation_Summary_Token_Count.png)
        
- 평가 데이터
    - Dialogue
        - 최대 토큰 개수: 1041
        - 최소 토큰 개수: 38
        - 평균 토큰 개수: 163.15
        
        ![Test Dialogue Token Count.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/92ef2076-f35a-42a1-85cb-8f48a45e51e1/Test_Dialogue_Token_Count.png)
        

## 3.2. Data Processing

데이터가 대체로 잘 정제되어 있었지만 일부 오타 및 연속된 자음을 수정 및 대체하는 작업을 거쳤다.

### 3.2.1. 오타 파악 및 수정

- fname: 'train_5385', dialogue: '...#Person2#: 먼저, 이것은 19세기 초 배경ㅇ로 설정된 로맨스 소설이에요...'
- fname: 'train_7201', dialogue: '...#Person1#: 편집장이 제ㅏ 다른 잡지에서 편집자로 일했던 경험이 있다는 걸 듣고, 그가 도우미 편집자가 되고 싶냐고 물어봤어요....'
- fname: 'train_9677', dialogue: '...#Person1#: 이제 그만. 너는 아직ㅍ알맞는 사람을 만나지 못했을 뿐이고, 너는 너무 많이 일하는 것 같아. 너는 어떻게 즐기고 삶을 즐기는 법을 배워야 해....'
- fname: 'train_12181', dialogue: '...#Person1#: 아무것도 안 했어. 그는 결국 나갔어. 그런데 오늘 또 그를 봤어. 신발 가게 밖에서. 카페 근처에서. 나는 CD 가게에 들어가서 CD를 보는 척했ㄷ거든. 그런데 그도 들어왔어....'

### 3.2.2 연속된 자음 대체

연속된 자음은 대화의 감정이나 의도를 전달하기 위한 부가적인 요소로 사용될 수 있다. 그러나 이러한 문자열은 일반적으로 모델에 의해 무시될 수 있거나 잘못 이해될 수 있으므로 **해당 문자열의 의미를 잘 포착하는 감정어로 대체**하여 모델이 이러한 부분을 더 잘 이해하고 처리할 수 있도록 했다.

- fname: 'train_3154', dialogue: '...#Person1#: 속았어! ㅋㅋ.. 완전 속았어....'
- fname: 'train_5429', dialogue: '...#Person2#: ㅋㅋ'..'
    
    → ‘ㅋㅋ’를 ‘웃기다’로 대체
    

### 3.2.3. 마스킹 정보 **Special Token으로 추가**

대회 데이터셋은 여러 종류의 개인정보를 마스킹 처리하여 제공하며, 두 개의 # 사이에 마스킹된 개인정보 종류(예: 전화번호, 주소 등)를 표시한다. 예) 전화번호 -> #PhoneNumber#

- 추출한 마스킹 문자열

```python
['#PassportNumber#', 
'#CardNumber#', 
'#Person3#', 
'#DateOfBirth#', 
'#Address#', 
'#CarNumber#', 
'#Email#', 
'#Person2#', 
'#Person6#', 
'#Person1#', 
'#Person#', 
'#Person7#', 
'#Person5#', 
'#PhoneNumber#', 
'#Person4#', 
'#SSN#']
```

→ Tokenizer의 Special Token으로 추가

## 3.3. Modeling

### 3.3.1. 사용 모델

- bart 기반의 한국어 사전학습 및 파인튜닝 모델
    - digit82/kobart-summarization
    - gogamza/kobart-summarization
- t5 기반의 한국어 사전학습 및 파인튜닝 모델
    - paust/pko-t5-base
    - lcw99/t5-base-korean-text-summary
    - noahkim/KoT5_news_summarization
    - paust/pko-t5-large
    - lcw99/t5-large-korean-text-summary

> **Bart**(Bidirectional and Auto-Regressive Transformers)
> 
> - 양방향 인코더-디코더 구조를 가지고 있으며, 자가지도 학습 방식인 MLM 및 NSP로 사전 학습되어 문장 복원 및 생성 작업에 특히 뛰어남

> **T5**(Text-to-Text Transfer Transformer)
> 
> - 텍스트 입력과 출력이 쌍으로 주어진 데이터를 사용하여 사전 학습되었기 때문에, 다양한 자연어 처리 작업에 대해 더 강력할 수 있음
> - 입력 텍스트에 prefix를 추가하여 모델이 수행해야 할 작업을 지정해야 함

### 3.3.2. Modeling Process

- 가설1: eda 결과를 학습 및 평가 데이터의 토큰 개수를 고려하여 encoder_max_len, decoder_max_len, generation_max_length를 조정하면 성능이 향상될 것이다.
    - Model: digit82/kobart-summarization
    encoder_max_len: 512 → **1000**
    decoder_max_len: 100 → **200**
    per_device_train_batch_size: 16
    -> 점수 향상, LB: 42.1735
    - Model: digit82/kobart-summarization
    encoder_max_len: 1000
    decoder_max_len: 200
    generation_max_length: 100 → **200**
    per_device_train_batch_size: 16
    -> 점수 향상, LB: 42.4697
    
    ![스크린샷 2024-03-21 오전 12.24.40.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/9174db07-e006-4a74-940c-b893c242f44f/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-03-21_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.24.40.png)
    
- 가설2: learning rate를 줄이면 파라미터가 더 안정적으로 수렴하여 스코어가 향상될 것이다.
    - Model: digit82/kobart-summarization
    encoder_max_len: 1000
    decoder_max_len: 200
    generation_max_length: 200
    per_device_train_batch_size: 16
    learning_rate: 1-e5 → **5-e6**
    -> 점수 하락, LB: 41.1145
- 가설3: weight decay를 늘리면 오버피팅을 방지하여 모델의 일반화 성능이 향상될 것이다.
    - Model: digit82/kobart-summarization
    encoder_max_len: 1000
    decoder_max_len: 200
    generation_max_length: 200
    per_device_train_batch_size: 16
    weight_decay: 0.01 → **0.02**
    -> 점수 하락, LB: 41.5500
    
    ![스크린샷 2024-03-21 오전 12.32.22.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/3b697f09-e0dd-45a7-995e-033cab48c3fe/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-03-21_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.32.22.png)
    
- 가설4: Text-to-Text 방식으로 입력과 출력 사이의 상관 관계를 명시적으로 모델링하는 T5 모델이 대화문 요약 task에서 더 높은 성능을 보일 것이다.
    - Model: paust/**pko-t5-base**
    encoder_max_len: 1000
    decoder_max_len: 200
    generation_max_length: 200
    per_device_train_batch_size: 6
    -> 점수 상승, LB: 42.8954
    - Model: lcw99/**t5-base-korean-text-summary**
    encoder_max_len: 1000
    decoder_max_len: 200
    generation_max_length: 200
    per_device_train_batch_size: 6
    -> 점수 상승, LB: 42.9973
    - Model: noahkim/**KoT5_news_summarization**
    encoder_max_len: 1000
    decoder_max_len: 200
    generation_max_length: 200
    per_device_train_batch_size: 6
    -> 점수 상승, LB: 43.2997
    
    ![스크린샷 2024-03-21 오전 12.42.58.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/eb1c9a29-b569-46c5-8253-125306ba30c9/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-03-21_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.42.58.png)
    
- 가설5: batch size를 늘리거나 gradient accumulation steps를 늘려 메모리 효율적인 방식으로 더 큰 배치 크기로 학습할 수 있도록 하면 수렴 속도가 향상될 것
    - Model: noahkim/KoT5_news_summarization
    encoder_max_len: 1000
    decoder_max_len: 200
    generation_max_length: 200
    gradient_accumulation_steps: 1 -> **4**
    -> 점수 하락, LB: 42.1842
    - Model: noahkim/KoT5_news_summarization
    encoder_max_len: 1000
    decoder_max_len: 200
    generation_max_length: 200
    per_device_train_batch_size: 6 → **7**
    -> 점수 하락, LB: 42.5872
    - Model: noahkim/KoT5_news_summarization
    encoder_max_len: 1000
    decoder_max_len: 200
    generation_max_length: 200
    per_device_train_batch_size: 7
    gradient_accumulation_steps: **4**
    -> 점수 하락, LB: 42.4952
    
    ![스크린샷 2024-03-21 오전 12.58.25.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/3ea0a449-7e90-4a3f-a35d-e77664f908d6/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-03-21_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.58.25.png)
    
- 가설6: 더 복잡한 t5 기반 사전 학습 모델을 사용하면 성능이 향상될 것
    - Model: paust/**pko-t5-large**
    encoder_max_len: 1000
    decoder_max_len: 200
    generation_max_length: 200
    per_device_train_batch_size: 1
    -> 점수 상승, LB: 43.4566
    - Model: lcw99/**t5-large-korean-text-summary**
    encoder_max_len: 1000
    decoder_max_len: 200
    generation_max_length: 200
    per_device_train_batch_size: 1
    -> 점수 상승, LB: 43.7724
    
    ![스크린샷 2024-03-21 오전 1.02.38.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/71d863d2-2414-46e9-be4c-894fec378b83/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-03-21_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_1.02.38.png)
    

# 4. 경진대회 결과

## 4.1. 리더보드 순위

- Public 리더보드
    - 2위, Rouge-F1 score: 43.7724

![스크린샷 2024-03-25 오후 4.39.24.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/f0207f1e-32c7-473d-abad-3e6e6bd6e8a4/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-03-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.39.24.png)

- Private(최종) 리더보드
    - 2위, Rouge-F1 score: 41.4034

![스크린샷 2024-03-25 오후 4.39.31.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/665aabc8-73cb-4a4a-bcc8-48cc3d6fce43/1f964c53-752c-4b44-9e23-06effefcaec0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-03-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.39.31.png)

## 4.2. 결과 분석

### 스코어 향상에 도움이 된 시도 및 전략

1. 하이퍼파라미터 튜닝
    - 모델의 성능을 향상시키기 위해 하이퍼파라미터 튜닝을 수행했다. 다른 팀원은 Optuna 라이브러리를 이용해 베이즈 최적화 알고리즘을 통한 하이퍼파라미터 탐색을 시도하기도 했지만, 실험의 각 단계를 직접 제어하며 각 파라미터를 더욱 세밀하게 조정하며 기록할 수 있도록 수동으로 직접 하이퍼파라미터를 조정하는 방식을 택했다. 소수의 하이퍼파라미터에 대해 실험을 진행했으며, 각 하이퍼파라미터에 대한 실험 단계에서 성능 향상에 도움이 된 조정 사항을 채택하여 다음 실험으로 이어갔다.
2. T5 Large 모델 선택
    - BART 기반 사전 학습 모델을 사용해 기본 하이퍼파라미터 튜닝을 마치고 구조가 더 복잡한 Encoder-Decoder Transformer 모델인 T5 기반의 사전 학습 모델을 사용해 학습하기로 했다. T5 모델은 Text-to-Text Transfer Transformer의 변형으로, 다양한 텍스트 처리 작업을 하나의 일관된 형식으로 처리할 수 있는 능력을 갖추고 있으며 입력과 출력 사이의 상관 관계를 명시적으로 모델링하기 때문에 대화문 요약 작업에서 BART보다 우수한 성능을 보일 것으로 기대했다.
        
        또한 T5 Large은 T5 Base 모델보다도 큰 규모의 모델로, 많은 파라미터와 깊은 네트워크를 가지고 있다. 이 모델을 선택함으로써 모델 용량과 표현 능력이 향상되므로 더 높은 성능을 기대할 수 있었다. 결과적으로 T5 Large 모델로 학습한 것이 최종 체출 스코어를 달성하는 데 결정적인 역할을 했다. 
        

### **마주한 한계 및 아쉬웠던 점**

1. 데이터 증강을 시도해보지 못한 것
2. Decoder Transformer 구조의 LLM 모델을 사용해보지 못한 것
