package org.deeplearning4j.examples.nlp.paragraphvectors;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * dl4j ParagraphVectors 구현 예제. 이 예에서는 학습 데이터에 있는 모든 문장에 대해 분산된 표현을 구축한다. 
 * LabelledDocument과 LabelAwareIterator를 이용한 레이블이 붙은 데이터를 학습하기 위해 사용된다. 
 *
 * *************************************************************************************************
 * 주의 : 본 예제는 DL4J/ND4J 버전이 rc3.8이상이어야 정상적으로 동작함.  
 * *************************************************************************************************
 *
 * @author raver119@gmail.com
 */
public class ParagraphVectorsTextExample {

    private static final Logger log = LoggerFactory.getLogger(ParagraphVectorsTextExample.class);

    public static void main(String[] args) throws Exception {
        ClassPathResource resource = new ClassPathResource("/raw_sentences.txt");
        File file = resource.getFile();
        SentenceIterator iter = new BasicLineIterator(file);

        AbstractCache<VocabWord> cache = new AbstractCache<>();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        /*
            LabelAwareIterator가 없다면 동기화 된 레이블 생성기를 사용해서 각 문서/문장/라인에 각각의 레이블을 붙일 수 있다. 
            LabelAwareIterator가 있다면 내부적으로 레이블을 붙일 수 있다. 
        */
        LabelsSource source = new LabelsSource("DOC_");

        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(5)
                .epochs(1)
                .layerSize(100)
                .learningRate(0.025)
                .labelsSource(source)
                .windowSize(5)
                .iterate(iter)
                .trainWordVectors(false)
                .vocabCache(cache)
                .tokenizerFactory(t)
                .sampling(0)
                .build();

        vec.fit();

        /*
            학습 코퍼스에는 꽤 가까운 단어가 포함된 문장이 몇 개 있다. 
            이 문장들은 벡터공간에서 인접하게 위치할 것이다. 

            line 3721: This is my way .
            line 6348: This is my case .
            line 9836: This is my house .
            line 12493: This is my world .
            line 16393: This is my work .

            위의 문장들과 전혀 관련이 없는 다음과 같은 문장도 있다. 
            line 9853: We now have one .

            문서의 인덱스는 0부터 시작한다는 것에 주의하자. 
         */

        double similarity1 = vec.similarity("DOC_9835", "DOC_12492");
        log.info("9836/12493 ('This is my house .'/'This is my world .') similarity: " + similarity1);

        double similarity2 = vec.similarity("DOC_3720", "DOC_16392");
        log.info("3721/16393 ('This is my way .'/'This is my work .') similarity: " + similarity2);

        double similarity3 = vec.similarity("DOC_6347", "DOC_3720");
        log.info("6348/3721 ('This is my case .'/'This is my way .') similarity: " + similarity3);

        // 이 경우 가능도는 매우 낮다. 
        double similarityX = vec.similarity("DOC_3720", "DOC_9852");
        log.info("3721/9853 ('This is my way .'/'We now have one .') similarity: " + similarityX +
            "(should be significantly lower)");
    }
}
