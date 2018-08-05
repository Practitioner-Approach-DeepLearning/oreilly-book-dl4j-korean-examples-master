package org.deeplearning4j.examples.nlp.word2vec;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

/**
 * 최초 어휘 구축 후 모델 가중치를 업데이트 하는 예제. 
 * w2v 모델 구축 후 새로운 코퍼스 학습이 필요하다면 본 예제를 참고하자. 
 *
 * 주의 : 새로운 단어는 어휘나 모델에 추가되지 않는다. 단지, 가중치 업데이트만 수행하기 때문에 종종 "frozen vocab training"으로 불린다. 
 *
 * @author raver119@gmail.com
 */
public class Word2VecUptrainingExample {

    private static Logger log = LoggerFactory.getLogger(Word2VecUptrainingExample.class);

    public static void main(String[] args) throws Exception {
        /*
                초기 모델 학습 단계 
         */
        String filePath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();

        log.info("Load & Vectorize Sentences....");
        // 각 라인의 양 끝 공란을 삭제
        SentenceIterator iter = new BasicLineIterator(filePath);
        // 공란을 기준으로 문장을 분리하여 라인에서 각 단어를 구분해 냄. 
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        // VocabCache와 WeightLookupTable를 수동으로 생성하는 것은 잘 사용되지 않지만 이 경우에는 필요함. 
        InMemoryLookupCache cache = new InMemoryLookupCache();
        WeightLookupTable<VocabWord> table = new InMemoryLookupTable.Builder<VocabWord>()
                .vectorLength(100)
                .useAdaGrad(false)
                .cache(cache)
                .lr(0.025f).build();

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .epochs(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .lookupTable(table)
                .vocabCache(cache)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();


        Collection<String> lst = vec.wordsNearest("day", 10);
        log.info("Closest words to 'day' on 1st run: " + lst);

        /*
            모델을 생성하고 나중에 사용할 수 있도록 저장. 
         */
        WordVectorSerializer.writeFullModel(vec, "pathToSaveModel.txt");

        /*
            시간이 지났다고 가정하고 새로운 코퍼스를 사용해서 가중치를 업데이트 해 보자. 
            추가된 코퍼스를 위해 모델을 구축하는 대신에 가중치 업데이트 모드를 이용할 수 있다. 
         */
        Word2Vec word2Vec = WordVectorSerializer.loadFullModel("pathToSaveModel.txt");

        /*
            주의 : 모델을 재저장한 후에도 이 모델을 학습하려면 SentenceIterator과 TokenizerFactory를 설정해야 한다. 
         */
        SentenceIterator iterator = new BasicLineIterator(filePath);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        word2Vec.setTokenizerFactory(tokenizerFactory);
        word2Vec.setSentenceIter(iterator);


        log.info("Word2vec uptraining...");

        word2Vec.fit();

        lst = word2Vec.wordsNearest("day", 10);
        log.info("Closest words to 'day' on 2nd run: " + lst);

        /*
            나중에 사용하도록 모델을 저장할 수 있다. 
         */
    }
}
