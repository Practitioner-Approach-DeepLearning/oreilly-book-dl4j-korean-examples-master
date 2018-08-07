package org.deeplearning4j.examples.dataExamples;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Adam Gibson
 */
public class CSVExample {

    private static Logger log = LoggerFactory.getLogger(CSVExample.class);

    public static void main(String[] args) throws  Exception {

        //먼저, 레코드 리더를 사용해 데이터셋을 가져온다. CSVRecordReader는 로드/파싱을 수행한다.
        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

        // 이제 RecordReaderDataSetIterator가 DataSet 객체로의 변환을 처리해 신경망에서 사용할 준비가 되었다.
        int labelIndex = 4;     // 각 행에 값 5개가 있는 iris.txt CSV: 입력 특징 4개 다음에 정수 레이블(클래스) 색인이 위치한다. 레이블은 각 행의 5번째 값(색인 4)이다.
        int numClasses = 3;     // 붓꽃 데이터셋에는 클래스 3개(붓꽃 유형)가 존재한다. 각 클래스 정수 값은 0, 1, 2이다.
        int batchSize = 150;    // 붓꽃 데이터셋은 총 150개의 입력데이터가 있다. 이를 한 DataSet 객체에 로드할 것이다(데이터셋이 클 때는 추천하지 않는다).

        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  //Use 65% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        // 데이터를 정규화하는데 NormalizerStandardize(평균 0, 단위 분산 제공)를 사용한다.
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           // 학습 데이터에서 통계 (평균/표준편차)를 수집. 이 단계에서는 입력 데이터를 수정하지는 않음
        normalizer.transform(trainingData);     // 학습 데이터에 정규화 적용
        normalizer.transform(testData);         // 테스트 데이터에 정규화 적용. 학습 데이터셋에서 계산된 통계를 이용


        final int numInputs = 4;
        int outputNum = 3;
        int iterations = 1000;
        long seed = 6;


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .learningRate(0.1)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                .build())
            .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(3).nOut(outputNum).build())
            .backprop(true).pretrain(false)
            .build();

        // 모델 실행
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        model.fit(trainingData);

        // 테스트셋으로 모델 평가
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatureMatrix());
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());
    }

}

