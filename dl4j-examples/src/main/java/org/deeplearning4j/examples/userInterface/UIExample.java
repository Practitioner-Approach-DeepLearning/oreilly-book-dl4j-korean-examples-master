package org.deeplearning4j.examples.userInterface;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.examples.userInterface.util.UIExampleUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Deeplearning4j's 학습 UI를 네트워크에 연결하는 방법을 배울 수 있는 간단한 예제.
 *
 * UI포트를 변경하려면(일반적으로는 할 필요 없음) : set the org.deeplearning4j.ui.port 
 * 즉, 다음 설정을 이용해서 예제를 실행하고 9001포트를 사용하도록 JVM에 전달하면 된다. -Dorg.deeplearning4j.ui.port=9001
 *
 * @author Alex Black
 */
public class UIExample {

    public static void main(String[] args){

        //신경망과 학습데이터 
        MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
        DataSetIterator trainData = UIExampleUtils.getMnistData();

        //UI 백엔드 초기화 
        UIServer uiServer = UIServer.getInstance();

        //신경망 정보 저장 공간 설정 (경사도, 활성화, 점수 vs. 시간 등)
        //그 후 StatsListener를 이용해서 신경망에서 정보 수집.
        StatsStorage statsStorage = new InMemoryStatsStorage();             //new FileStatsStorage(File) 사용 가능(UIStorageExample 예제 참고) 
        int listenerFrequency = 1;
        net.setListeners(new StatsListener(statsStorage, listenerFrequency));

        //StatsStorage 인스턴스를 UI에 연결하여 담긴 내용을 시각화.
        uiServer.attach(statsStorage);

        //학습 시작. 
        net.fit(trainData);

        //마지막으로 브라우저를 열고 http://localhost:9000/train 에 접속. 
    }
}
