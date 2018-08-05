package org.deeplearning4j.examples.nlp.sequencevectors;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * SequenceVectors를 사용하여 학습 된 데이터 추상화 시퀀스 예제.
 * 텍스트 문장을 Sequence로 간주하고, VocabWords를 SequenceElements로 사용한다. 
 * 즉 데이터 시퀀스에서 분산 표현을 학습하는 방법에 대한 간단한 데모이다. 
 *
 * 다른 타입의 데이터를 학습하기 위해 기본적인 SequenceElement 클래스를 확장해서 사용할 수 있다. 단, 이 경우 모델 영속성 관리는 사용자가 직접 처리해야 한다. 
 *
 * *************************************************************************************************
 * 주의 : 본 예제는 DL4J/ND4J 버전이 rc3.8이상이어야 정상적으로 동작함.  
 * *************************************************************************************************
 * @author raver119@gmail.com
 */
public class SequenceVectorsTextExample {

    protected static final Logger logger = LoggerFactory.getLogger(SequenceVectorsTextExample.class);

    public static void main(String[] args) throws Exception {

        ClassPathResource resource = new ClassPathResource("raw_sentences.txt");
        File file = resource.getFile();

        AbstractCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();

        /*
            먼저 라인 반복자 생성
         */
        BasicLineIterator underlyingIterator = new BasicLineIterator(file);


        /*
            이제 라인을 VocabWords 시퀀스로 표현해야 한다. 
            SentenceTransformer를 사용해 보자. 
         */
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(underlyingIterator)
                .tokenizerFactory(t)
                .build();


        /*
            생성된 transformer 변수를 AbstractSequenceIterator에 넣어보자. 
         */
        AbstractSequenceIterator<VocabWord> sequenceIterator =
            new AbstractSequenceIterator.Builder<>(transformer).build();


        /*
            이제 시퀀스 반복자에서 어휘를 생성하다. 
            이 과정을 건너뛰고 set AbstractVectors.resetModel(TRUE)를 이용하면 어휘가 내부적으로 마스터 된다. 
        */
        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(sequenceIterator, 5)
                .setTargetVocabCache(vocabCache)
                .build();

        constructor.buildJointVocabulary(false, true);

        /*
            새 모델을 위한 WeightLookupTable 인스턴스 생성에 걸리는 시간. 
        */

        WeightLookupTable<VocabWord> lookupTable = new InMemoryLookupTable.Builder<VocabWord>()
                .lr(0.025)
                .vectorLength(150)
                .useAdaGrad(false)
                .cache(vocabCache)
                .build();

         /*
             재설정 모델은 AbstractVectors.resetModel()을 false로 설정한 경우에만 사용 가능하다. 
             만약 True로 설정한다면, 내부적으로 호출된다. 
        */
        lookupTable.resetWeights(true);

        /*
            이제 적합한 AbstractVectors 모델을 만들어 보자. 
         */
        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration())
                // 학습 코퍼스 각 구성요소들의 최소 발생 횟수. 이 횟수 이하인 경우 모두 무시된다. 
                // 이 값은 resetModel()가 TRUE로 내부 모델 생성인 경우에만 유효하다. 다른 경우에는 무시되고 실제 어휘 내용이 사용된다. 
                .minWordFrequency(5)

                // WeightLookupTable
                .lookupTable(lookupTable)

                // 학습 코퍼스를 커버하는 추상 반복자. 
                .iterate(sequenceIterator)

                // 모델링에 앞서 작성된 어휘 
                .vocabCache(vocabCache)

                // 배치 크기는 한번에 쓰레드 하나에서 처리되는 시퀀스의 수이다. 
                // iterations > 1 인 경우 중요한 값임. 
                .batchSize(250)

                // 배치 반복 횟수 
                .iterations(1)

                // 전체 학습 코퍼스에 대한 반복 횟수 
                .epochs(1)

                // true로 설정하면 어휘는 내부적인 수집으로 생성되고, false로 설정하면 외부에서 제공한 어휘를 사용하게 된다. 
                .resetModel(false)


                /*
                    다음 두 함수는 학습 목표를 정의한다. 적어도 하나의 목표는 true로 설정되어야 한다. 
                 */
                .trainElementsRepresentation(true)
                .trainSequencesRepresentation(false)

                /*
                    구성요소 학습 알고리즘을 특정한다(예를 들면, SkipGram 같은 것)
                 */
                .elementsLearningAlgorithm(new SkipGram<VocabWord>())

                .build();

        /*
            이제, 모든 옵션은 설정되었다. fit()을 실행하자. 
         */
        vectors.fit();

        /*
            fit()이 종료되면 바로 모델이 구축된 것으로 간주하고 테스트를 시작할 수 있다. 
            모든 유사한 컨텍스트는 SequenceElement 레이블을 통해 처리 되므로, 
            보다 복잡한 객체/관계를 모델링하기 위하여 AbstractVectors를 사용하는 경우 레이블의 고유성과 의미에 대해서는 직접 처리해야 한다. 
         */
        double sim = vectors.similarity("day", "night");
        logger.info("Day/night similarity: " + sim);

    }
}
