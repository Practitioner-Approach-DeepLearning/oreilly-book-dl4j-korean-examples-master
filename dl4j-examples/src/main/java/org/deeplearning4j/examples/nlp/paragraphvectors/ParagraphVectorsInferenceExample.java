package org.deeplearning4j.examples.nlp.paragraphvectors;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * dl4j ParagraphVectors inference 구현 예제.
 * 이전에 구축한 모델을 로드하고 학습에 사용되지 않은 원본 문장을 전달해서 벡터 표현값을 얻는다. 
 *
 * *************************************************************************************************
 * 주의 : 본 예제는 DL4J/ND4J 버전이 0.6.0 이상이어야 정상적으로 동작함. 
 * *************************************************************************************************
 *
 * @author raver119@gmail.com
 */
public class ParagraphVectorsInferenceExample {

    private static final Logger log = LoggerFactory.getLogger(ParagraphVectorsInferenceExample.class);

    public static void main(String[] args) throws Exception {
        ClassPathResource resource = new ClassPathResource("/paravec/simple.pv");
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        // 외부에서 모델 로드 
        ParagraphVectors vectors = WordVectorSerializer.readParagraphVectors(resource.getFile());
        vectors.setTokenizerFactory(t);
        vectors.getConfiguration().setIterations(1); // 더 빠른 결과를 얻기 위해 iteration을 1로 설정함. 

        /*
        // 다음과 같이 word2vec 모델을 직접 사용할 수도 있다. 
        // Google-like은 사용할 수 없다는 점에 주의하라. (허프만 트리 정보가 없기 때문에) 

        ParagraphVectors vectors = new ParagraphVectors.Builder()
            .useExistingWordVectors(word2vec)
            .build();
        */
        // 복원된 모델을 위해 형태소 분석기를 이 부분에 설정해야함.  


        INDArray inferredVectorA = vectors.inferVector("This is my world .");
        INDArray inferredVectorA2 = vectors.inferVector("This is my world .");
        INDArray inferredVectorB = vectors.inferVector("This is my way .");

        // 단어 WAY와 WORLD는 실제로 매우 밀접한 문맥에서 사용되었기 때문에 높은 유사성이 나타날 것으로 예상 됨. 
        log.info("Cosine similarity A/B: {}", Transforms.cosineSim(inferredVectorA, inferredVectorB));

        // 같은 문장에 대해 계산하기 때문에 아마 같다는 결과가 나올 것으로 예상 됨. 
        log.info("Cosine similarity A/A2: {}", Transforms.cosineSim(inferredVectorA, inferredVectorA2));
    }
}
