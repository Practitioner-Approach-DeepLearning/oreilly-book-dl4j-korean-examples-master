package org.deeplearning4j.stats;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.rnn.SparkLSTMCharacterExample;
import org.deeplearning4j.spark.api.stats.SparkTrainingStats;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.stats.EventStats;
import org.deeplearning4j.spark.stats.StatsUtils;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 이 예제는 DL4J의 스파크 학습 벤치마킹 / 디버깅 / 타이밍 기능을 사용하는 법을 보여준다.
 * 자세한 내용은 https://deeplearning4j.org/spark#sparkstats을 참고하라.
 *
 * 이 도구는 성능 문제가 있는지 판별하고 디버그하기 위해 스파크 학습의 다양한 측면에서 통계를 수집한다.
 *
 * 이 예제에서는 SparkLSTMCharacterExample의 신경망 구성과 데이터를 사용한다.
 *
 *
 * 예제를 로컬에서 돌리고 싶다면 예제를 그대로 실행하면 된다. 이 예제는 기본적으로 스파크 로컬에서 실행하도록 구성되어 있다.
 *
 * (클러스터에서 실행하기 위해) 스파크 서브밋을 사용해 예제를 실행시키고 싶다면 "-useParkLocal false"를 애플리케이션 매개변수에 포함시키거나,
 * 예제 앞 부분의 필드를 "useSparkLocal = false"로 설정하라.
 *
 * 주의: 인터넷에 접근할 수 없는 일부 클러스터에서는 "Error querying NTP server" 오류 메세지와 함께 예제가 실패할 수 있다.
 * 참고: https://deeplearning4j.org/spark#sparkstatsntp
 *
 * @author 알렉스 블랙
 */
public class TrainingStatsExample {
    private static final Logger log = LoggerFactory.getLogger(TrainingStatsExample.class);

    @Parameter(names="-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = true;

    public static void main(String[] args) throws Exception {
        new TrainingStatsExample().entryPoint(args);
    }

    private void entryPoint(String[] args) throws Exception {
        // 명령줄 인자 다루기
        JCommander jcmdr = new JCommander(this);
        try{
            jcmdr.parse(args);
        } catch(ParameterException e){
            // 사용자가 잘못 입력함 -> 사용법 출력
            jcmdr.usage();
            try{ Thread.sleep(500); } catch(Exception e2){ }
            throw e;
        }


        // 신경망 구성 설정
        MultiLayerConfiguration config = getConfiguration();

        // 스파크 관련 구성 설정
        int examplesPerWorker = 8;      //쉽게 말해 각 워커(익스큐터)가 처리할 미니 배치 크기
        int averagingFrequency = 3;     //매개 변수가 평균화되는 빈도

        // 스파크 구성 및 컨텍스트 설정
        SparkConf sparkConf = new SparkConf();
        if(useSparkLocal){
            sparkConf.setMaster("local[*]");
            log.info("Using Spark Local");
        }
        sparkConf.setAppName("DL4J Spark Stats Example");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //데이터 가져오기. 자세한 설명은 SparkLSTMCharacterExample 참고
        JavaRDD<DataSet> trainingData = SparkLSTMCharacterExample.getTrainingData(sc);

        // TrainingMaster를 설정. TrainingMaster는 스파크로 학습되는 과정을 제어한다.
        // 여기서는 기본적인 파라미터 평균화를 사용한다.
        int examplesPerDataSetObject = 1;   // 데이터를 미리 배치화하지 않았다. 따라서 각 DataSet 객체는 입력 데이터 한개를 포함한다.
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
                .workerPrefetchNumBatches(2)    // 비동기로 배치를 최대 2개 미리 가져옴
                .averagingFrequency(averagingFrequency)
                .batchSizePerWorker(examplesPerWorker)
                .build();

        // 스파크 신경망 생성
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, config, tm);

        // *** 학습 통계를 수집하도록 신경망 설정. 기본적으로는 학습 통계를 수집하지 않음 ***
        sparkNetwork.setCollectTrainingStats(true);

        // 에포크 하나에 대한 학습을 수행
        sparkNetwork.fit(trainingData);

        // 다 사용한 임시 학습 파일 삭제, (에포크 여러개에 대한 학습을 수행할 경우 재사용됨)
        tm.deleteTempFiles(sc);

        // 통계 가져오기
        SparkTrainingStats stats = sparkNetwork.getSparkTrainingStats();
        Set<String> statsKeySet = stats.getKeySet();    // 통계 유형별 키
        System.out.println("--- Collected Statistics ---");
        for(String s : statsKeySet){
            System.out.println(s);
        }

        // 데모 목적: 통계 하나를 가져와 출력
        String first = statsKeySet.iterator().next();
        List<EventStats> firstStatEvents = stats.getValue(first);
        EventStats es = firstStatEvents.get(0);
        log.info("Training stats example:");
        log.info("Machine ID:     " + es.getMachineID());
        log.info("JVM ID:         " + es.getJvmID());
        log.info("Thread ID:      " + es.getThreadID());
        log.info("Start time ms:  " + es.getStartTime());
        log.info("Duration ms:    " + es.getDurationMs());

        // 학습 과정에서 계산된 다양한 통계 차트를 포함하는 HTML 내보내기
        StatsUtils.exportStatsAsHtml(stats, "SparkStats.html",sc);
        log.info("Training stats exported to {}", new File("SparkStats.html").getAbsolutePath());

        log.info("****************Example finished********************");
    }


    // 학습할 신경망 구성
    private static MultiLayerConfiguration getConfiguration(){
        int lstmLayerSize = 200;					// 각 GravesLSTM 계층의 유닛 수
        int tbpttLength = 50;                       // 단기 BPTT의 길이. 파라미터 업데이트를 50자까지 수행

        Map<Character, Integer> CHAR_TO_INT = SparkLSTMCharacterExample.getCharToInt();
        int nIn = CHAR_TO_INT.size();
        int nOut = CHAR_TO_INT.size();

        // 신경망 구성 설정
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .learningRate(0.1)
            .updater(Updater.RMSPROP).rmsDecay(0.95)
            .seed(12345)
            .regularization(true).l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayerSize).activation(Activation.TANH).build())
            .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize).activation(Activation.TANH).build())
            .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)        // MCXENT + 소프트맥스를 사용해 분류
                .nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
            .pretrain(false).backprop(true)
            .build();

        return conf;
    }
}
