# 자바를 활용한 딥러닝
: 딥러닝 입문부터 DL4J를 이용한 신경망 구현과 스파크, 하둡 연동까지

## Deeplearning4J 신경망 예제 :
- 다층 퍼셉트론 (MLP Neural Nets)
- 합성곱 신경망 (Convolutional Neural Nets)
- 순환 신경망 (Recurrent Neural Nets)
- TSNE
- Word2Vec & GloVe
- 이상치 검출

---

## 빌드와 실행

예제를 빌드하려면 [Maven](https://maven.apache.org/) 을 이용하면 된다. 
```
mvn clean package
```

예제를 실행하기 위해서 `runexamples.sh` 파일을 실행하자 ([bash](https://www.gnu.org/software/bash/)가 필요하다). 예제 목록이 나타나고 실행할 예제 번호를 묻는 메시지가 출력될 것이다. `--all`을 입력하면 모든 예제가 실행된다. 다른 옵션을 보고 싶다면 `-h`를 입력해 보자. 

```
./runexamples.sh [-h | --help]
```


## 관련 문서
추가 정보를 알고 싶다면 [deeplearning4j.org](http://deeplearning4j.org/)과 [JavaDoc](http://deeplearning4j.org/doc/)을 방문하면 된다. 

코드 실행을 하며 문제가 발생한다면 로그를 남기고, 본 프로젝트에 기여하고자 하면 pull request를 날려주면 된다. 언제나 환영이다 :) 


