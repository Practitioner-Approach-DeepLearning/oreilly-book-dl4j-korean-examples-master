package org.deeplearning4j.examples.nlp.tsne;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 9/20/14.
 *
 * 고차원 데이터셋의 차원 축소. 
 */
public class TSNEStandardExample {

    private static Logger log = LoggerFactory.getLogger(TSNEStandardExample.class);

    public static void main(String[] args) throws Exception  {
        //1 단계 : 초기화 
        int iterations = 100;
        //double 값을 가지는 n차원 배열 생성 
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        List<String> cacheList = new ArrayList<>(); // cacheList는 모든 단어를 담는데 사용되는 string 동적 배열이다. 

        //2 단계 : 텍스트 입력을 단어의 리스트로 변환 
        log.info("Load & Vectorize data....");
        File wordFile = new ClassPathResource("words.txt").getFile();   //파일 열기 
        //모든 고유 단어 벡터 데이터 가져오기 
        Pair<InMemoryLookupTable,VocabCache> vectors = WordVectorSerializer.loadTxt(wordFile);
        VocabCache cache = vectors.getSecond();
        INDArray weights = vectors.getFirst().getSyn0();    //각 단어의 가중치를 각각의 리스트로 분리 

        for(int i = 0; i < cache.numWords(); i++)   //각 단어의 문자를 각각의 리스트로 분리 
            cacheList.add(cache.wordAtIndex(i));

        //3 단계 : 사용할 이중 트리 tsne 생성 
        log.info("Build model....");
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(iterations).theta(0.5)
                .normalize(false)
                .learningRate(500)
                .useAdaGrad(false)
//                .usePca(false)
                .build();

        //4 단계 : tsne 값을 설정하고 파일에 저장하기 
        log.info("Store TSNE Coordinates for Plotting....");
        String outputFile = "target/archive-tmp/tsne-standard-coords.csv";
        (new File(outputFile)).getParentFile().mkdirs();
        tsne.plot(weights,2,cacheList,outputFile);
        //tnse는 행렬 벡터의 가중치를 사용하고 차원 두개를 가지며 단어 문자를 레이블로 가진다. 
        //이전 단계에서 생성한 outputFile에 기록됨. 
        // gnuplot 를 이용해서 그래프를 그릴수 있음. 
        // 데이터 파일의 구분자는 ","
        // 레이블 폰트는 "Times,8"로 지정하여 'tsne-standard-coords.csv' 파일을 그래프로 그려보자. 
        //!!! plot은 최근에 지원 중단 되었기 때문에 가장 마지막 줄을 다시 실행해야 할 수도 있다. 
    }



}

