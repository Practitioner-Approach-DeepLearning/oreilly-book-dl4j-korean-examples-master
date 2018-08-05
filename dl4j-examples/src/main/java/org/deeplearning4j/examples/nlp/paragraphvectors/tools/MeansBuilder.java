package org.deeplearning4j.examples.nlp.paragraphvectors.tools;

import lombok.NonNull;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 이전에 학습된 ParagraphVectors 모델을 기반으로 LabelledDocument의 중심 벡터를 만드는 간단한 유틸리티 클래스.
 *
 * @author raver119@gmail.com
 */
public class MeansBuilder {
    private VocabCache<VocabWord> vocabCache;
    private InMemoryLookupTable<VocabWord> lookupTable;
    private TokenizerFactory tokenizerFactory;

    public MeansBuilder(@NonNull InMemoryLookupTable<VocabWord> lookupTable,
        @NonNull TokenizerFactory tokenizerFactory) {
        this.lookupTable = lookupTable;
        this.vocabCache = lookupTable.getVocab();
        this.tokenizerFactory = tokenizerFactory;
    }

    /**
     * 문서의 중심(mean) 벡터를 반환함. 
     *
     * @param document
     * @return
     */
    public INDArray documentAsVector(@NonNull LabelledDocument document) {
        List<String> documentAsTokens = tokenizerFactory.create(document.getContent()).getTokens();
        AtomicInteger cnt = new AtomicInteger(0);
        for (String word: documentAsTokens) {
            if (vocabCache.containsWord(word)) cnt.incrementAndGet();
        }
        INDArray allWords = Nd4j.create(cnt.get(), lookupTable.layerSize());

        cnt.set(0);
        for (String word: documentAsTokens) {
            if (vocabCache.containsWord(word))
                allWords.putRow(cnt.getAndIncrement(), lookupTable.vector(word));
        }

        INDArray mean = allWords.mean(0);

        return mean;
    }
}
