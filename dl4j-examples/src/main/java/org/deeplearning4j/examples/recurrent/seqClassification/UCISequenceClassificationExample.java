package org.deeplearning4j.examples.recurrent.seqClassification;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.api.berkeley.Pair;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Sequence Classification Example Using a LSTM Recurrent Neural Network
 *
 * This example learns how to classify univariate time series as belonging to one of six categories.
 * Categories are: Normal, Cyclic, Increasing trend, Decreasing trend, Upward shift, Downward shift
 *
 * Data is the UCI Synthetic Control Chart Time Series Data Set
 * Details:     https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series
 * Data:        https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data
 * Image:       https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/data.jpeg
 *
 * This example proceeds as follows:
 * 1. Download and prepare the data (in downloadUCIData() method)
 *    (a) Split the 600 sequences into train set of size 450, and test set of size 150
 *    (b) Write the data into a format suitable for loading using the CSVSequenceRecordReader for sequence classification
 *        This format: one time series per file, and a separate file for the labels.
 *        For example, train/features/0.csv is the features using with the labels file train/labels/0.csv
 *        Because the data is a univariate time series, we only have one column in the CSV files. Normally, each column
 *        would contain multiple values - one time step per row.
 *        Furthermore, because we have only one label for each time series, the labels CSV files contain only a single value
 *
 * 2. Load the training data using CSVSequenceRecordReader (to load/parse the CSV files) and SequenceRecordReaderDataSetIterator
 *    (to convert it to DataSet objects, ready to train)
 *    For more details on this step, see: http://deeplearning4j.org/usingrnns#data
 *
 * 3. Normalize the data. The raw data contain values that are too large for effective training, and need to be normalized.
 *    Normalization is conducted using NormalizerStandardize, based on statistics (mean, st.dev) collected on the training
 *    data only. Note that both the training data and test data are normalized in the same way.
 *
 * 4. Configure the network
 *    The data set here is very small, so we can't afford to use a large network with many parameters.
 *    We are using one small LSTM layer and one RNN output layer
 *
 * 5. Train the network for 40 epochs
 *    At each epoch, evaluate and print the accuracy and f1 on the test set
 *
 * @author Alex Black
 */
/**
 * LSTM 반복적 인 신경망을 이용한 시퀀스 분류 예
 *
 * 이 예제는 단 변수 시계열을 여섯 가지 범주 중 하나로 분류하는 방법을 학습한다.
 * 카테고리는 다음과 같다: Normal, Cyclic, Increasing trend, Decreasing trend, Upward shift, Downward shift
 * 데이터는 UCI 합성 제어 차트 시계열 데이터셋이다.
 * 상세내용:     https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series
 * 데이터:        https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data
 * 이미지들:       https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/data.jpeg
 *
 * 이 예제의 순서는 다음과 같다.
 * 1. 데이터를 다운로드하고 준비한다 (downloadUCIData()메소드 안에서 수행)
 *    (a) 600개의 데이터를 450 세트의 학습 데이터셋으로 분할하고 150개를 테스트 데이터셋으로 분할한다.
 *    (b) 데이터 분류를 위해 CSVSequenceRecordReader를 사용하여 로드에 적합한 형식으로 데이터를 작성하자.
 *        이 형식은 파일 당 하나의 시계열과 레이블에 대한 별도의 파일이다.
 *        예를 들어 train / features / 0.csv는 train / labels / 0.csv라는 레이블 파일과 함께 사용되는 기능이다
 *        데이터는 단 변량 시계열이므로 CSV 파일에는 하나의 열만 있다. 일반적으로 각 열에는 여러 값 (행당 하나의 시간 단계)이 포함된다.
 *        또한 각 시계열마다 하나의 레이블 만 있기 때문에 레이블 CSV 파일에는 단일 값만 포함된다.
 *
 * 2. CSVSequenceRecordReader를 사용하여 데이터를 로드하고 SequenceRecordReaderDataSetIterator로 DataSet 개체로 변환하여 준비를 완료한다.
 *    이 단계에 대한 자세한 내용은 다음을 참조하자. 
 *    http://deeplearning4j.org/usingrnns#data
 *
 * 3. 데이터 정규화. 원시 데이터에는 효과적인 교육에 비해 너무 큰 값이 포함되어 있으므로 정규화 해야 한다.
 *    정규화는 교육 데이터에서만 수집 된 통계 (평균, stdev)를 기반으로 NormalizerStandardize를 사용하여 수행된다.
 *    트레이닝 데이터와 테스트 데이터는 동일한 방식으로 정규화된다.
 *
 * 4. 신경망 설정
 *    여기에 설정된 데이터는 매우 작기 때문에 매개 변수가 많은 대규모 신경망을 사용할 여력이 없다. 
 *    하나의 작은 LSTM 레이어와 하나의 RNN 출력 레이어를 사용하고 있다.
 *
 * 5. 신경망을 40에포크동안 학습시킨다
 *    각 에포크마다 테스트셋의 정확도를 평가하고 출력한다.
 *
 * @author Alex Black
 */
public class UCISequenceClassificationExample {
    private static final Logger log = LoggerFactory.getLogger(UCISequenceClassificationExample.class);

    //'baseDir': 데이터의 기본 디렉토리. 데이터를 다른 곳에 저장하려면이 값을 변경하자.
    private static File baseDir = new File("src/main/resources/uci/");
    private static File baseTrainDir = new File(baseDir, "train");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "test");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");

    public static void main(String[] args) throws Exception {
        downloadUCIData();

        // ----- 학습용 데이터 로드 -----
        //train/features/0.csv에서 train/features/449.csv 까지 450 개의 기능에 대한 학습용 파일이 있다.
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));

        int miniBatchSize = 10;
        int numLabelClasses = 6;
        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        //학습용데이터 정규화
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);              //학습자료 통계 수집
        trainData.reset();

        //이전에 수집 된 통계를 사용하여 바로 정상화하자. 'trainData'이터레이터에서 반환 된 각 DataSet을 정규화한다.
        trainData.setPreProcessor(normalizer);


        // ----- 테스트 데이터 로드 -----
        //학습 데이터와 동일한 프로세스
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, 149));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, 149));

        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        testData.setPreProcessor(normalizer);   //학습 데이터와 똑같은 정규화 과정을 사용하고 있다.


        // ----- 신경망 설정 -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)    //향상된 반복성을위한 난수 생성기 시드 설정
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .learningRate(0.005)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //항상 필요한 것은 아니지만이 데이터셋에 도움이 된다.
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(10).nOut(numLabelClasses).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(20));   //20 회 반복 할 때마다 점수 (손실 함수 값)를 출력하자.


        // ----- 신경망을 학습하여 각 에포크마다 테스트셋의 성능 평가 -----
        int nEpochs = 40;
        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainData);

            //테스트셋에 대한 평가
            Evaluation evaluation = net.evaluate(testData);
            log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));

            testData.reset();
            trainData.reset();
        }

        log.info("----- Example Complete -----");
    }


    //이 방법은 데이터를 다운로드하고 "한 줄에 하나의 시간"형식을 적합한 형식으로 변환한다.
    //DataVec (CsvSequenceRecordReader) 및 DL4J가 읽을 수있는 CSV 시퀀스 형식.
    private static void downloadUCIData() throws Exception {
        if (baseDir.exists()) return;    //데이터가 이미 있다. 다시 다운로드하지 말자.

        String url = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data";
        String data = IOUtils.toString(new URL(url));

        String[] lines = data.split("\n");

        //디렉토리 생성
        baseDir.mkdir();
        baseTrainDir.mkdir();
        featuresDirTrain.mkdir();
        labelsDirTrain.mkdir();
        baseTestDir.mkdir();
        featuresDirTest.mkdir();
        labelsDirTest.mkdir();

        int lineCount = 0;
        List<Pair<String, Integer>> contentAndLabels = new ArrayList<>();
        for (String line : lines) {
            String transposed = line.replaceAll(" +", "\n");

            //레이블 : 처음 100 개의 예 (선)는 레이블 0, 두 번째 100은 레이블 1 이다.
            contentAndLabels.add(new Pair<>(transposed, lineCount++ / 100));
        }

        //무작위 배정 및 학습 / 테스트 분할 수행 :
        Collections.shuffle(contentAndLabels, new Random(12345));

        int nTrain = 450;   //75% 학습, 25% 테스트
        int trainCount = 0;
        int testCount = 0;
        for (Pair<String, Integer> p : contentAndLabels) {
            //적절한 위치에서 읽을 수있는 형식으로 출력을 작성.
            File outPathFeatures;
            File outPathLabels;
            if (trainCount < nTrain) {
                outPathFeatures = new File(featuresDirTrain, trainCount + ".csv");
                outPathLabels = new File(labelsDirTrain, trainCount + ".csv");
                trainCount++;
            } else {
                outPathFeatures = new File(featuresDirTest, testCount + ".csv");
                outPathLabels = new File(labelsDirTest, testCount + ".csv");
                testCount++;
            }

            FileUtils.writeStringToFile(outPathFeatures, p.getFirst());
            FileUtils.writeStringToFile(outPathLabels, p.getSecond().toString());
        }
    }
}
