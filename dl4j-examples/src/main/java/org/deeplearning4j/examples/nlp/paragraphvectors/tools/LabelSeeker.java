package org.deeplearning4j.examples.nlp.paragraphvectors.tools;

import lombok.NonNull;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

/**
 * 가장 유사한 레이블을 찾는 원시적인 탐색기이다. 
 * ParagraphVectors는 각각의 단어가 아닌 레이블이 어떻게 할당될 것인지만 고려되기 때문에 기본적이인 wordsNearest 함수 대신에 이용된다. 
 *
 * @author raver119@gmail.com
 */
public class LabelSeeker {
    private List<String> labelsUsed;
    private InMemoryLookupTable<VocabWord> lookupTable;

    public LabelSeeker(@NonNull List<String> labelsUsed, @NonNull InMemoryLookupTable<VocabWord> lookupTable) {
        if (labelsUsed.isEmpty()) throw new IllegalStateException("You can't have 0 labels used for ParagraphVectors");
        this.lookupTable = lookupTable;
        this.labelsUsed = labelsUsed;
    }

    /**
     * 벡터를 이용해서 문서를 표현하고, 이전에 학습된 각 카테고리와 이 문서와의 거리를 반환한다. 
     * @return
     */
    public List<Pair<String, Double>> getScores(@NonNull INDArray vector) {
        List<Pair<String, Double>> result = new ArrayList<>();
        for (String label: labelsUsed) {
            INDArray vecLabel = lookupTable.vector(label);
            if (vecLabel == null) throw new IllegalStateException("Label '"+ label+"' has no known vector!");

            double sim = Transforms.cosineSim(vector, vecLabel);
            result.add(new Pair<String, Double>(label, sim));
        }
        return result;
    }
}
