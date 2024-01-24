[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/g6ZC_OOE)
# House Price Prediction Competition 

## Team

| ![이명진](https://avatars.githubusercontent.com/u/156163982?v=4) | ![서재현](https://github.com/UpstageAILab/upstage-ml-regression-07/assets/116725865/4bacd3a4-e386-4da0-9be9-ed42caafeca0) | ![신주용](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이영훈](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이준형](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [이명진](https://github.com/UpstageAILab)             |            [서재현](https://github.com/UpstageAILab)             |            [신주용](https://github.com/UpstageAILab)             |            [이영훈](https://github.com/UpstageAILab)             |            [이준형](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

##  :clipboard: Stacks 
  <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white"> <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=for-the-badge&logo=Google colab&logoColor=white">  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=NumPy&logoColor=white"> <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"> <img src="https://img.shields.io/badge/Polars-CD792C?style=for-the-badge&logo=Polars&logoColor=white"> <img src="https://img.shields.io/badge/scikit-learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=Git&logoColor=white"> <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"> <img src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=Slack&logoColor=white"> <img src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=Notion&logoColor=white"> <img src="https://img.shields.io/badge/Zoom-0B5CFF?style=for-the-badge&logo=Zoom&logoColor=white">

## 1. Competiton Info

### Overview

- **House Price Prediction | 아파트 실거래가 예측**
- 서울시 아파트 실거래가 매매 데이터를 기반으로 아파트 가격을 예측하는 대회
  
<img width="500" alt="스크린샷 2024-01-18 오후 2 39 56" src="https://github.com/UpstageAILab/upstage-ml-regression-07/assets/46295610/3a6076ac-60ea-434a-87ba-e67ed97fa041"><img width="500" alt="스크린샷 2024-01-18 오후 2 39 42" src="https://github.com/UpstageAILab/upstage-ml-regression-07/assets/46295610/3e1136d0-6f0b-4abc-8960-ee9a427e7850">





### Timeline

- ex) January 15, 2024 - Start Date
- ex) January 25, 2024 - Final submission deadline

### Evaluation

- 해당 시점의 매매 실거래가를 예측하는 Regression 대회이며, 평가지표는 RMSE(Root Mean Squared Error)를 사용합니다.
<img width="539" alt="스크린샷 2024-01-18 오후 2 54 04" src="https://github.com/UpstageAILab/upstage-ml-regression-07/assets/46295610/e1ec2b35-cb45-4070-9289-fdfa1fd2c50b">

- RMSE는 예측된 값과 실제 값 간의 평균편차를 측정합니다. 아파트 매매의 맥락에서는 회귀 모델이 실제 거래 가격의 차이를 얼마나 잘 잡아내는지 측정합니다. 

## 2. Components

### Directory

- _Insert your directory structure_

## 3. Data descrption

### Dataset overview

주요 데이터는 .csv 형태로 제공되며, 서울시 아파트의 각 시점에서의 거래금액(만원)을 예측하는 것이 목표입니다.

학습 데이터는 아래와 같이 1,118,822개이며, 예측해야 할 거래금액(target)을 포함한 52개의 아파트의 정보에 대한 변수와 거래시점에 대한 변수가 주어집니다.

<img width="500" alt="스크린샷 2024-01-18 오후 2 58 52" src="https://github.com/UpstageAILab/upstage-ml-regression-07/assets/46295610/6ea70113-72bb-4263-8029-e7bf2b5efb1a">

학습 데이터의 기간은 2007년 1월 1일부터 2023년 6월 30일까지이며, 각 변수 명이 한글로 되어있어 어떤 정보를 나타내는 변수인지 쉽게 확인할 수 있습니다.

예시)
- 시군구 : “서울특별시 강남구 개포동” 과 같이 주소에 대한 정보입니다.
- 아파트명 : “개포더샵트리에”와 같이 아파트명에 대한 정보입니다.
- 전용면적(㎡) : “108.2017”와 같이 매매대상의 전용면적에 대한 정보입니다.
- 건축년도 : “2021”과 같이 아파트의 건축 연도를 나타내는 정보입니다.

각 변수들은 아래와 같은 결측치 비율을 가지고 있습니다.

<img width="500" alt="스크린샷 2024-01-18 오후 2 59 16" src="https://github.com/UpstageAILab/upstage-ml-regression-07/assets/46295610/cf7266b5-2eb3-4a9e-ab81-c53174cafb54">

아파트의 매매가를 결정하는데에 교통적인 요소가 영향을 줄 수 있기에 추가 데이터로 서울시 지하철역, 서울시 버스정류장의 정보가 주어집니다. 

<img width="500" alt="스크린샷 2024-01-18 오후 2 59 52" src="https://github.com/UpstageAILab/upstage-ml-regression-07/assets/46295610/4314505a-3c92-46ce-9cfe-535761fe5f10">

추가 데이터는 위도와 경도, 좌표 X와 좌표Y와 같이 거리에 대한 정보가 포함되어 있으며, 이를 활용하여 학습 데이터와 함께 사용할 수 있습니다. 

### EDA

- _Describe your EDA process and step-by-step conclusion_

### Feature engineering
 - 실험실험실험 
- _Describe feature engineering process_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- https://www.notion.so/7-9bf356d1cd69404681bd4a0ba7453c77

### Reference

- _Insert related reference_
