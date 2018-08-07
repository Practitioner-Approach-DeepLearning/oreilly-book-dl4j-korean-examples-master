package org.deeplearning4j.examples.dataExamples;

import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.meta.Prediction;
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
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

/**
 * 이 예제는 기본적인 CSV 예제에 다음을 추가했다.
 * (a) 메타데이터 추적 - 예를 들어 데이터가 어떤 입력 데이터로부터 생성됐는가?
 * (b) 평가 정보 추가 - 예측 오차에 관한 메타데이터를 가져옴
 *
 * @author 알렉스 블랙
 */
public class CSVExampleEvaluationMetaData {

    public static void main(String[] args) throws  Exception {
        // 먼저, 레코드 리더를 사용해 데이터셋을 가져온다. CSVExample과 동일하다 - 자세한 내용은 해당 예제 참조
        RecordReader recordReader = new CSVRecordReader(0, ",");
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        int labelIndex = 4;
        int numClasses = 3;
        int batchSize = 150;

        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        iterator.setCollectMetaData(true);  // 메타데이터를 수집하고 DataSet 객체에 저장하도록 반복자에 지시한다.
        DataSet allData = iterator.next();
        allData.shuffle(123);
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  // 데이터의 65%를 학습에 사용

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        // 학습 및 테스트셋에서 입력 데이터의 메타데이터를 보자.
        List<RecordMetaData> trainMetaData = trainingData.getExampleMetaData(RecordMetaData.class);
        List<RecordMetaData> testMetaData = testData.getExampleMetaData(RecordMetaData.class);

        // 수집된 메타데이터를 사용해 학습 및 테스틑셋에 어떤 입력 데이터가 있는지 자세히 보여준다.
        System.out.println("  +++++ Training Set Examples MetaData +++++");
        String format = "%-20s\t%s";
        for(RecordMetaData recordMetaData : trainMetaData){
            System.out.println(String.format(format, recordMetaData.getLocation(), recordMetaData.getURI()));
            // 대안으로 사용 가능: recordMetaData.getReaderClass()
        }
        System.out.println("\n\n  +++++ Test Set Examples MetaData +++++");
        for(RecordMetaData recordMetaData : testMetaData){
            System.out.println(recordMetaData.getLocation());
        }


        // CSVExample과 동일하게 데이터 정규화
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           // 학습 데이터에서 통계 (평균/표준편차)를 수집. 이 단계에서는 입력 데이터를 수정하지는 않음
        normalizer.transform(trainingData);     // 학습 데이터에 정규화 적용
        normalizer.transform(testData);         // 테스트 데이터에 정규화 적용. 학습 데이터셋에서 계산된 통계를 이용


        // 간단한 모델을 구성한다. 평가/오차를 표시하기 위해 최적의 구성을 사용하지는 않는다.
        final int numInputs = 4;
        int outputNum = 3;
        int iterations = 50;
        long seed = 6;

        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .learningRate(0.1)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build())
            .backprop(true).pretrain(false)
            .build();

        // 모델 피팅
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        model.fit(trainingData);

        // 테스트셋으로 모델 평가
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatureMatrix());
        eval.eval(testData.getLabels(), output, testMetaData);          // 테스트셋 메타 데이터를 전달하고 있다
        System.out.println(eval.stats());

        // Evaluation 객체에서 예측 오차 목록을 가져온다.
        // 이와 같은 예측 오차는 iterator.setCollectMetaData(true)를 호출한 후에만 사용할 수 있다.
        List<Prediction> predictionErrors = eval.getPredictionErrors();
        System.out.println("\n\n+++++ Prediction Errors +++++");
        for(Prediction p : predictionErrors){
            System.out.println("Predicted class: " + p.getPredictedClass() + ", Actual class: " + p.getActualClass()
                + "\t" + p.getRecordMetaData(RecordMetaData.class).getLocation());
        }

        // 데이터의 하위 집합을 DataSet 객체에 로드할 수도 있다.
        List<RecordMetaData> predictionErrorMetaData = new ArrayList<>();
        for( Prediction p : predictionErrors ) predictionErrorMetaData.add(p.getRecordMetaData(RecordMetaData.class));
        DataSet predictionErrorExamples = iterator.loadFromMetaData(predictionErrorMetaData);
        normalizer.transform(predictionErrorExamples);  // 이 서브셋에 정규화 적용

        // 원본 데이터를 로드할 수도 있다.
        List<Record> predictionErrorRawData = recordReader.loadFromMetaData(predictionErrorMetaData);

        // 원본 데이터, 정규화 된 데이터, 레이블, 네트워크 예측과 함께 예측 오차를 출력
        for(int i=0; i<predictionErrors.size(); i++ ){
            Prediction p = predictionErrors.get(i);
            RecordMetaData meta = p.getRecordMetaData(RecordMetaData.class);
            INDArray features = predictionErrorExamples.getFeatures().getRow(i);
            INDArray labels = predictionErrorExamples.getLabels().getRow(i);
            List<Writable> rawData = predictionErrorRawData.get(i).getRecord();

            INDArray networkPrediction = model.output(features);

            System.out.println(meta.getLocation() + ": "
                + "\tRaw Data: " + rawData
                + "\tNormalized: " + features
                + "\tLabels: " + labels
                + "\tPredictions: " + networkPrediction);
        }


        // 기타 유용한 평가 방법:
        List<Prediction> list1 = eval.getPredictions(1,2);                  // 예측: 실제 클래스 1, 예측 클래스 2
        List<Prediction> list2 = eval.getPredictionByPredictedClass(2);     //예측 클래스 2에 대한 모든 예측
        List<Prediction> list3 = eval.getPredictionsByActualClass(2);       //실제 클래스 2에 대한 모든 예측
    }
}
