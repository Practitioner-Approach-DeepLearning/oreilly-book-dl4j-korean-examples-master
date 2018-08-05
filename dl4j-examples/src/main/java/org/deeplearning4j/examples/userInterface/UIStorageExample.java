package org.deeplearning4j.examples.userInterface;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.examples.userInterface.util.UIExampleUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * UIStorageExample은 네트워크 학습 데이터를 파일에 저장하고, 이후에 다시 로드하여 UI에 보여주는 방법을 나타내는 예제임.
 *
 * @author Alex Black
 */
public class UIStorageExample {

    public static void main(String[] args){

        //이 예제를 두 번 실행하세요. 한 번은 collectStats = true로 그 이후 collectStats = false로 한 번 더 실행. 
        boolean collectStats = true;

        File statsFile = new File("UIStorageExampleStats.dl4j");

        if(collectStats){
            //최초 실행: 네트워크에서 학습 상태 수집 
            //수집 단계에서는 실제로 그릴 필요는 없지만, 필요하다면 그릴 수도 있음. 

            MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
            DataSetIterator trainData = UIExampleUtils.getMnistData();

            StatsStorage statsStorage = new FileStatsStorage(statsFile);
            net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));

            net.fit(trainData);

            System.out.println("Done");
        } else {
            //두 번째 실행: 저장된 상테를 로드해서 시각화. http://localhost:9000/train 접속. 

            StatsStorage statsStorage = new FileStatsStorage(statsFile);    //파일이 이미 존재한다면, 파일에서 부터 데이터 로드.
            UIServer uiServer = UIServer.getInstance();
            uiServer.attach(statsStorage);
        }
    }
}
