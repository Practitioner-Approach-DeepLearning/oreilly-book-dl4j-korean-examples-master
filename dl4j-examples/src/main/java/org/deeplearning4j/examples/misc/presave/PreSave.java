package org.deeplearning4j.examples.misc.presave;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * 미리 데이터셋을 저장하는 것은 중요하다.
 * 하나의 데이터 형식을 사용해야하는 다른 프레임 워크와 달리 deeplearning4j를 사용하면 임의의 데이터를로드 할 수 있으며 
 * 텍스트, 이미지, 비디오, 로그 데이터의 다양한 데이터를 사전처리 하기위한 datavec와 같은 도구도 제공한다.
 *
 * 이 예제에서는 PreSave 데이터를 저장하기 위해 데이터 선택기를 사용하는 방법을 미리 보여준다.
 * 다른 클래스의 LoadPreSavedLenetMnistExample 클래스에서는 학습용폴더 및 테스트폴더에서 데이터를 로드하기 위해 출력을 사용한다.
 *
 * 데이터 셋을 미리 저장하면 시간을 많이 절약 할 수 있다.
 * 병목이 될 때마다 처리를 다시 시도 할 때마다, 데이터를 미리 저장하면 학습 중에 처리량을 높일 수 있다.
 *
 *
 * @author Adam Gibson
 */
public class PreSave {
    private static final Logger log = LoggerFactory.getLogger(LoadPreSavedLenetMnistExample.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64; // 테스트 배치 데이터


        /*
           하나의 반복에 대한 배치 크기를 이용해 반복자를 작성해 보자.
         */
        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);
        File trainFolder = new File("trainFolder");
        trainFolder.mkdirs();
        File testFolder = new File("testFolder");
        testFolder.mkdirs();
        log.info("Saving train data to " + trainFolder.getAbsolutePath() +  " and test data to " + testFolder.getAbsolutePath());
        //저장할 파일의 인덱스를 추적하자.
        //이러한 배치 인덱스는 반복기에 의해 저장되는 미니배치를 인덱싱하는 데 사용된다.
        int trainDataSaved = 0;
        int testDataSaved = 0;
        while(mnistTrain.hasNext()) {
            // 파일의 배치에 대한 인덱스로 testDataSaved를 사용한다.
            mnistTrain.next().save(new File(trainFolder,"mnist-train-" + trainDataSaved + ".bin"));
                                                                              //^^^^^^^
                                                                              //******************
                                                                              //이것이 무엇인지 알아야한다.
                                                                              //이것은 파일이 저장될 위치이다.
                                                                              //******************************************
            trainDataSaved++;
        }

        while(mnistTest.hasNext()) {
            // 파일의 배치에 대한 인덱스로 testDataSaved를 사용한다.
            mnistTest.next().save(new File(testFolder,"mnist-test-" + testDataSaved + ".bin"));
                                                                            //^^^^^^^
                                                                            //******************
                                                                            //이것이 무엇인지 알아야한다.
                                                                            //이것은 파일이 저장될 위치이다.
                                                                            //******************************************
            testDataSaved++;
        }

        log.info("Finished pre saving test and train data");


    }

}
