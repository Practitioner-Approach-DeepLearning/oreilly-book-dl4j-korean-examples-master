package org.deeplearning4j.examples.nlp.paragraphvectors;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.examples.nlp.paragraphvectors.tools.LabelSeeker;
import org.deeplearning4j.examples.nlp.paragraphvectors.tools.MeansBuilder;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.util.List;

/**
 * DL4j ParagraphVectors를 이용한 문서 분류기 예. 
 * ParagraphVectors 를 사용하는 전반적인 개념은 LDA를 쓸 때와 같음 : 토픽 공간 모델링
 *
 * 학습용으로 레이블이 붙은 카테고리를 몇 개 가지고 있고, 레이블에 없는 문서가 몇 개 있다고 가정함. 
 * 목표는 레이블이 붙이 않은 이 문서들을 적당한 카테고리로 분류하는 것. 
 *
 *
 * 참고 : cascade 방식을 활용하면 정확도를 높일 수 있지만, 기본 예제 수준에서는 벗어나기 때문에 생략한다. 
 *
 * @author raver119@gmail.com
 */
public class ParagraphVectorsClassifierExample {

    ParagraphVectors paragraphVectors;
    LabelAwareIterator iterator;
    TokenizerFactory tokenizerFactory;

    private static final Logger log = LoggerFactory.getLogger(ParagraphVectorsClassifierExample.class);

    public static void main(String[] args) throws Exception {

      ParagraphVectorsClassifierExample app = new ParagraphVectorsClassifierExample();
      app.makeParagraphVectors();
      app.checkUnlabeledData();
        /*
                출력은 다음과 같을 것이다. 

                'health' 문서의 카테고리별 분류 :
                    health: 0.29721372296220205
                    science: 0.011684473733853906
                    finance: -0.14755302887323793

                'finance' 문서의 카테고리별 분류 :
                    health: -0.17290237675941766
                    science: -0.09579267574606627
                    finance: 0.4460859189453788

                    이제 아직까지 몰랐던 문서들의 카테고리를 알게 되었다. 
         */
    }

    void makeParagraphVectors()  throws Exception {
      ClassPathResource resource = new ClassPathResource("paravec/labeled");

      // 데이터셋을 위한 반복자 생성 
      iterator = new FileLabelAwareIterator.Builder()
              .addSourceFolder(resource.getFile())
              .build();

      tokenizerFactory = new DefaultTokenizerFactory();
      tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

      // ParagraphVectors 학습 설정 
      paragraphVectors = new ParagraphVectors.Builder()
              .learningRate(0.025)
              .minLearningRate(0.001)
              .batchSize(1000)
              .epochs(20)
              .iterate(iterator)
              .trainWordVectors(true)
              .tokenizerFactory(tokenizerFactory)
              .build();

      // 모델 학습 시작 
      paragraphVectors.fit();
    }

    void checkUnlabeledData() throws FileNotFoundException {
      /*
      모델이 빌드 되었다고 가정하고, 레이블이 없는 문서를 어느 카테고리로 분류할지 확인할 것이다. 
      즉 레이블이 없는 문서를 로드하고 체크한다. 
     */
     ClassPathResource unClassifiedResource = new ClassPathResource("paravec/unlabeled");
     FileLabelAwareIterator unClassifiedIterator = new FileLabelAwareIterator.Builder()
             .addSourceFolder(unClassifiedResource.getFile())
             .build();

     /*
      레이블이 없는 문서를 확인하고 어떤 레이블을 할당할 수 있는지 확인한다. 
      일반적으로 많은 도메인에서 하나의 문서는 여러개의 레이블에 대해 서로 다른 가중치를 가지며 속할 수 있다. 
     */
     MeansBuilder meansBuilder = new MeansBuilder(
         (InMemoryLookupTable<VocabWord>)paragraphVectors.getLookupTable(),
           tokenizerFactory);
     LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
         (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

     while (unClassifiedIterator.hasNextDocument()) {
         LabelledDocument document = unClassifiedIterator.nextDocument();
         INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
         List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

         /*
          document.getLabel()은 전체 문서명을 출력하는 대신 현재 보고 있는 문서를 확인하기 위해 사용한다. 
          즉, 문서의 레이블을 제목처럼 사용해서 문서가 제대로 분류되었는지 시각화 하고 있다. 
         */
         log.info("Document '" + document.getLabel() + "' falls into the following categories: ");
         for (Pair<String, Double> score: scores) {
             log.info("        " + score.getFirst() + ": " + score.getSecond());
         }
     }

    }
}
