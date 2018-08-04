package org.deeplearning4j.mlp;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * 스파크를 사용해 MNIST 데이터로 간단한 소규모 다층 퍼셉트론을 학습하고, 분산 관점에서 테스트셋으로 평가한다.
 *
 * 이 예제의 신경망은 작아서 굳이 스파크를 사용할 필요가 없지만 - 스파크 학습을 평가하는 방법과 구성을 보기에 적절하다.
 *
 *
 * 예제를 로컬에서 돌리고 싶다면 예제를 그대로 실행하면 된다. 이 예제는 기본적으로 스파크 로컬에서 실행하도록 구성되어 있다.
 * 주의: 스파크 로컬은 개발/테스트 용으로만 사용해야 한다. 대신 (다중 GPU 시스템 같은) 단일 머신에서 데이터를 병렬 학습시킬 때는
 * ParallelWrapper를 사용하라 (단일 머신에서 스파크를 사용할 때보다 빠르다).
 * dl4j-cuda-specific-examples에서 MultiGpuLenetMnistExample을 참고하라.
 *
 * (클러스터에서 실행하기 위해) 스파크 서브밋을 사용해 예제를 실행시키고 싶다면 "-useParkLocal false"를 애플리케이션 매개변수에 포함시키거나,
 * 예제 앞 부분의 필드를 "useSparkLocal = false"로 설정하라.
 *
 * @author 알렉스 블랙
 */
public class MnistMLPExample {
    private static final Logger log = LoggerFactory.getLogger(MnistMLPExample.class);

    @Parameter(names = "-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = true;

    @Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = 16;

    @Parameter(names = "-numEpochs", description = "Number of epochs for training")
    private int numEpochs = 15;

    public static void main(String[] args) throws Exception {
        new MnistMLPExample().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        // 명령줄 인자 다루기
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            // 사용자가 잘못 입력함 -> 사용법 출력
            jcmdr.usage();
            try { Thread.sleep(500); } catch (Exception e2) { }
            throw e;
        }

        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("DL4J Spark MLP Example");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        // 데이터를 메모리에 불러온 다음 병렬 처리
        // 올바른 사용법은 아니지만 예제 구현이 간단해진다.
        DataSetIterator iterTrain = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
        DataSetIterator iterTest = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
        List<DataSet> trainDataList = new ArrayList<>();
        List<DataSet> testDataList = new ArrayList<>();
        while (iterTrain.hasNext()) {
            trainDataList.add(iterTrain.next());
        }
        while (iterTest.hasNext()) {
            testDataList.add(iterTest.next());
        }

        JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);
        JavaRDD<DataSet> testData = sc.parallelize(testDataList);


        //----------------------------------
        // 신경망 구성 및 신경망 학습 시작
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .activation(Activation.LEAKYRELU)
            .weightInit(WeightInit.XAVIER)
            .learningRate(0.02)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(500).build())
            .layer(1, new DenseLayer.Builder().nIn(500).nOut(100).build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX).nIn(100).nOut(10).build())
            .pretrain(false).backprop(true)
            .build();

        // 스파크 학습 구성 : 옵션에 대한 설명은 http://deeplearning4j.org/spark 참고
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)    // 각 DataSet 객체는 기본적으로 입력 데이터 32개를 포함
            .averagingFrequency(5)
            .workerPrefetchNumBatches(2)            // 비동기로 입력 데이터를 워커당 2개씩 미리 가져옴
            .batchSizePerWorker(batchSizePerWorker)
            .build();

        // 스파크 신경망 생성
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, tm);

        // 학습 실행
        for (int i = 0; i < numEpochs; i++) {
            sparkNet.fit(trainData);
            log.info("Completed Epoch {}", i);
        }

        // 평가 수행 (분산)
        Evaluation evaluation = sparkNet.evaluate(testData);
        log.info("***** Evaluation *****");
        log.info(evaluation.stats());

        // 임시 학습 파일 삭제, 완료
        tm.deleteTempFiles(sc);

        log.info("***** Example Complete *****");
    }
}
