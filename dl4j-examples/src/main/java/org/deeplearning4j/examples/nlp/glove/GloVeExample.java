package org.deeplearning4j.examples.nlp.glove;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collection;

/**
 * @author raver119@gmail.com
 */
public class GloVeExample {

    private static final Logger log = LoggerFactory.getLogger(GloVeExample.class);

    public static void main(String[] args) throws Exception {
        File inputFile = new ClassPathResource("raw_sentences.txt").getFile();

        // 학습용 코퍼스를 감싸는 SentenceIterator 생성. 
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

        // 줄마다 공백으로 분리해서 단어 획득.
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Glove glove = new Glove.Builder()
                .iterate(iter)
                .tokenizerFactory(t)


                .alpha(0.75)
                .learningRate(0.1)

                // 학습 에포크 수 
                .epochs(25)

                // 가중치 함수 컷오프 
                .xMax(100)

                // 학습 코퍼스로 부터 가져온 배치 크기별로 학습이 이루어짐. 
                .batchSize(1000)

                // True이면 학습 하기 전에 배치를 섞음. 
                .shuffle(true)

                // True인 경우 단어 쌍은 양방향으로 모두 구축 됨. 
                .symmetric(true)
                .build();

        glove.fit();

        double simD = glove.similarity("day", "night");
        log.info("Day/night similarity: " + simD);

        Collection<String> words = glove.wordsNearest("day", 10);
        log.info("Nearest words to 'day': " + words);

        System.exit(0);
    }
}
