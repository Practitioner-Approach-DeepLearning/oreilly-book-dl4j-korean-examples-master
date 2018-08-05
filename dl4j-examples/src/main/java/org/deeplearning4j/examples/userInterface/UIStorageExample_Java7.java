package org.deeplearning4j.examples.userInterface;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.examples.userInterface.util.UIExampleUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.J7StatsListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.sqlite.J7FileStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * JAVA 7 호환성 확인을 위한 UI 예제
 *
 * *** 참고 ***
 * 1: 꼭 JAVA 7을 사용해야 하는 것이 아니라면 표준 UIStorageExample를 사용 하면 더 빠르다. 
 * 2: UI 자체는 JAVA 8이 필요함(백엔드로 Play 프레임워크 사용). 상태를 장비에 저장하고 그 파일을 JAVA 8이 설치된 다른 장비에 복사해서 시각화도 가능함. 
 * 3: J7FileStatsStorage과 FileStatsStorage 은 호환되지 않는다. 한 가지 방식으로 저장/로드 해야함. 
 *    (J7FileStatsStorage는 JAVA 8에서도 동작하지만, FileStatsStorage은 JAVA 7에서 동작하지 않는다)
 *
 * @author Alex Black
 */
public class UIStorageExample_Java7 {

    public static void main(String[] args){

        //이 예제를 두 번 실행하세요. 한 번은 collectStats = true로 그 이후 collectStats = false로 한 번 더 실행. 
        boolean collectStats = true;

        File statsFile = new File("UIStorageExampleStats_Java7.dl4j");

        //최초 실행: 네트워크에서 학습 상태 수집 
        //수집 단계에서는 실제로 그릴 필요는 없지만, 필요하다면 그릴 수도 있음. 

        MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
        DataSetIterator trainData = UIExampleUtils.getMnistData();

        StatsStorage statsStorage = new J7FileStatsStorage(statsFile);                                      // J7 임에 유의 
        net.setListeners(new J7StatsListener(statsStorage), new ScoreIterationListener(10));

        net.fit(trainData);

        System.out.println("Done");
    }
}
