package org.deeplearning4j.examples.userInterface;

import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter;
import org.deeplearning4j.examples.userInterface.util.UIExampleUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * 다른 JVM에서 UI를 호스팅하는 방법을 보여주는 예제. 
 *
 * 이 예제의 경우 동일한 JVM에서 수행 됨. 실제로 서로 다른 JVM에서 수행하기 위해서 주석 참고. 
 *
 * 참고: UI를 꼭 분리된 JVM에서 호스팅 해야 되는 경우가 아니라면 본 예제를 사용하지 마세요. 
 *      단일 JVM인 경우 일반적인 방법보다 아래의 예제가 느릴 수 있습니다. 
 *
 * UI포트를 변경하려면(일반적으로는 할 필요 없음) : set the org.deeplearning4j.ui.port 
 * 즉, 다음 설정을 이용해서 예제를 실행하고 9001포트를 사용하도록 JVM에 전달하면 된다.  -Dorg.deeplearning4j.ui.port=9001
 *
 * @author Alex Black
 */
public class RemoteUIExample {

    public static void main(String[] args){

        //------------ 첫 번째 JVM: UI 서버를 시작하고 원격 리스너 지원 활성화. ------------
        //UI 백엔드 초기화 
        UIServer uiServer = UIServer.getInstance();
        uiServer.enableRemoteListener();        //필수: 원격 지원은 기본적으로 비활성화 되어 있음. 
        //uiServer.enableRemoteListener(new FileStatsStorage(new File("myFile.dl4j")), true);       //디스크에 저장해도 됨.


        //------------ 두 번째 JVM: 학습 수행 ------------

        //신경망과 학습 데이터 
        MultiLayerNetwork net = UIExampleUtils.getMnistNetwork();
        DataSetIterator trainData = UIExampleUtils.getMnistData();

        //원격 상태 저장소 라우터 생성. 결과를 HTTP를 이용해서 UI로 보낸다. UI는 http://localhost:9000를 가정. 
        StatsStorageRouter remoteUIRouter = new RemoteUIStatsStorageRouter("http://localhost:9000");
        net.setListeners(new StatsListener(remoteUIRouter));

        //학습 시작 
        net.fit(trainData);

        //마지막으로 브라우저를 열고 http://localhost:9000/train 에 접속. 
    }
}
