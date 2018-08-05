package org.deeplearning4j.examples.nlp.word2vec;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

/**
 * Created by agibsonccc on 10/9/14.
 *
 * 텍스트를 워드벡터로 처리하는 신경망. 
 * 상세한 내용을 보고 싶으면 다음 url을 방문하자. https://deeplearning4j.org/word2vec.html
 */
public class Word2VecRawTextExample {

    private static Logger log = LoggerFactory.getLogger(Word2VecRawTextExample.class);

    public static void main(String[] args) throws Exception {

        // 텍스트 파일 경로 
        String filePath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();

        log.info("Load & Vectorize Sentences....");
        // 각 라인의 양 끝 공란을 삭제
        SentenceIterator iter = new BasicLineIterator(filePath);
        // 공란을 기준으로 문장을 분리하여 라인에서 각 단어를 구분해 냄. 
        TokenizerFactory t = new DefaultTokenizerFactory();

        /*
            CommonPreprocessor는 각 토큰에 다음 정규식을 적용한다. [\d\.:,"'\(\)\[\]|/?!;]+
            효과적으로 모든 숫자, 마침표를 비롯해서 특수 기호가 삭제된다. 
            또한 모든 문자는 소문자로 변환된다. 
         */
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Writing word vectors to text file....");

        // 워드 벡터를 파일에 저장 
        WordVectorSerializer.writeWordVectors(vec, "pathToWriteto.txt");

        // "day"와 가장 가까운 단어 10개를 출력한다. 워드 벡터를 통해 작업한 결과다. 
        log.info("Closest Words:");
        Collection<String> lst = vec.wordsNearest("day", 10);
        System.out.println("10 Words closest to 'day': " + lst);

        // TODO UiServer 설정하기 
//        UiServer server = UiServer.getInstance();
//        System.out.println("Started on port " + server.getPort());
    }
}
