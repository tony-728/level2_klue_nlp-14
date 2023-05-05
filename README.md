# level2_klue_nlp-14

# 팀 소개

### 

|김광연|김민준|김병준|김상혁|서재명|
| :-: | :-: | :-: | :-: | :-: |
|![광연님](https://user-images.githubusercontent.com/59431433/217448461-bb7a37d4-f5d4-418b-a1b9-583b561b5733.png)|![민준님](https://user-images.githubusercontent.com/59431433/217448432-a3d093c4-0145-4846-a775-00650198fc2f.png)|![병준님](https://user-images.githubusercontent.com/59431433/217448424-11666f05-dda6-406d-95e8-47b3bab7c2f6.png)|![상혁2](https://user-images.githubusercontent.com/59431433/217448849-758c8e25-87db-4902-ab06-0aa8c359500c.png)|![재명님](https://user-images.githubusercontent.com/59431433/217448416-b2ba2070-6cfb-4829-a3bd-861f526cb74a.png)|

# 프로젝트 개요
### 관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다.
### 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.  

## 활용 장비 및 재료

- ai stage GPU server 활용
    - GPU: V 100

## 데이터 셋
- trin data: 32,470
- test data: 7765
### 데이터 예시
- column 1: 샘플 순서 id
- column 2: sentence.
- column 3: subject_entity
- column 4: object_entity
- column 5: label
- column 6: 샘플 출처
