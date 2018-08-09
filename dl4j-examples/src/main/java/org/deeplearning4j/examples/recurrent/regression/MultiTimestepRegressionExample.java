package org.deeplearning4j.examples.recurrent.regression;


import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;

/**
 * 이 예제는 Jason Brownlee의 케라스(keras) 예제를 참고했다, 아래 내용으로 확인이 가능하다.
 * http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
 *
 * LSTM을 사용하여 다중 시간 단계 회귀를 보여준다.
 */
public class MultiTimestepRegressionExample {
    private static final Logger LOGGER = LoggerFactory.getLogger(MultiTimestepRegressionExample.class);

    private static File baseDir = new File("dl4j-examples/src/main/resources/rnnRegression");
    private static File baseTrainDir = new File(baseDir, "multiTimestepTrain");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "multiTimestepTest");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");


    public static void main(String[] args) throws Exception {

        //학습, 테스트 및 단계에 대한 예제 세트 수
        int trainSize = 100;
        int testSize = 20;
        int numberOfTimesteps = 20;

        //여러 시간 단계 데이터 준비, 자세한 정보를 보려면 메소드 주석보기
        List<String> rawStrings = prepareTrainAndTest(trainSize, testSize, numberOfTimesteps);

        //miniBatch Size가 trainSize 및 testSize로 나눌 수 있는지 확인한다. rnn TimeStep은 다른 크기의 예제를 허용하지 않기 떄문에 miniBatchSize는 고정으로 둔다.
        int miniBatchSize = 10;

        // ----- 학습데이터 로드 -----
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/train_%d.csv", 0, trainSize-1));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/train_%d.csv", 0, trainSize-1));

        DataSetIterator trainDataIter = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        //학습 셋 일반화
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fitLabel(true);
        normalizer.fit(trainDataIter);              //교육 자료 통계 수집
        trainDataIter.reset();


        // ----- 테스트 데이터 로드 -----
        //교육 데이터와 동일한 프로세스이다.
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/test_%d.csv", trainSize, trainSize+testSize-1));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/test_%d.csv", trainSize, trainSize+testSize-1));

        DataSetIterator testDataIter = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        trainDataIter.setPreProcessor(normalizer);
        testDataIter.setPreProcessor(normalizer);


        // ----- 신경망 설정 -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(140)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .learningRate(0.15)
            .list()
            .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10)
                .build())
            .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY).nIn(10).nOut(1).build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(20));

        // ----- 신경망을 학습하여 각 에포크마다 테스트셋의 성능 평가 -----
        int nEpochs = 50;

        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainDataIter);
            trainDataIter.reset();
            LOGGER.info("Epoch " + i + " complete. Time series evaluation:");

            //단일 열 입력에 대한 회귀 분석 실행
            RegressionEvaluation evaluation = new RegressionEvaluation(1);

            //평가 실행. 25k 정도이기 때문에, 시간이 좀 걸릴 수 있다.
            while(testDataIter.hasNext()){
                DataSet t = testDataIter.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                INDArray predicted = net.output(features,false);

                evaluation.evalTimeSeries(lables,predicted);
            }

            System.out.println(evaluation.stats());

            testDataIter.reset();
        }

        /**
         * 이 지점 아래의 모든 코드는 플로팅에만 필요하다.
         */

        //학습 데이터가 있는 rrnTimeStemp 초기화 및 테스트 데이터 예측
        while (trainDataIter.hasNext()) {
            DataSet t = trainDataIter.next();
            net.rnnTimeStep(t.getFeatureMatrix());
        }

        trainDataIter.reset();

        DataSet t = testDataIter.next();
        INDArray predicted  = net.rnnTimeStep(t.getFeatureMatrix());
        normalizer.revertLabels(predicted);

        //플로팅을 위해 원시 문자열 데이터를 IndArrays로 변환
        INDArray trainArray = createIndArrayFromStringList(rawStrings, 0, trainSize);
        INDArray testArray = createIndArrayFromStringList(rawStrings, trainSize, testSize);

        //데이터 없이 플롯 만들기
        XYSeriesCollection c = new XYSeriesCollection();
        createSeries(c, trainArray, 0, "Train data");
        createSeries(c, testArray, trainSize-1, "Actual test data");
        createSeries(c, predicted, trainSize-1, "Predicted test data");

        plotDataset(c);

        LOGGER.info("----- Example Complete -----");
    }


    /**
     * 문자열 목록에서 INDArray를 만든다.
     * 플로팅 용도로 사용
     */
    private static INDArray createIndArrayFromStringList(List<String> rawStrings, int startIndex, int length) {
        List<String> stringList = rawStrings.subList(startIndex,startIndex+length);
        double[] primitives = new double[stringList.size()];

        for (int i = 0; i < stringList.size(); i++) {
            primitives[i] = Double.valueOf(stringList.get(i));
        }

        return Nd4j.create(new int[]{1,length},primitives);
    }

    /**
     * 플로팅 목적으로 다른 시계열을 만드는 데 사용된다.
     */
    private static XYSeriesCollection createSeries(XYSeriesCollection seriesCollection, INDArray data, int offset, String name) {
        int nRows = data.shape()[2];
        XYSeries series = new XYSeries(name);
        for (int i = 0; i < nRows; i++) {
            series.add(i + offset, data.getDouble(i));
        }

        seriesCollection.addSeries(series);

        return seriesCollection;
    }

    /**
     * 제공된 데이터 세트의 xy 플롯을 생성하자.
     */
    private static void plotDataset(XYSeriesCollection c) {

        String title = "Regression example";
        String xAxisLabel = "Timestep";
        String yAxisLabel = "Number of passengers";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = false;
        boolean urls = false;
        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);

        // 추가 맞춤 설정을 위한 플롯에 대한 참조 얻기...
        final XYPlot plot = chart.getXYPlot();

        // 초기 창에서 시계열에 맞게 자동 확대 / 축소
        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setAutoRange(true);

        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        RefineryUtilities.centerFrameOnScreen(f);
        f.setVisible(true);
    }

    /**
     * 이 방법을 사용하면 CSV 파일을 기반으로 여러 단계의 문제에 대해 예상되는 구조의 데이터를 사전 처리 할 수 있다. 
     * 이 예제에서는 단일 열 CSV를 입력으로 사용하지만 예제는 다중 열 입력과 함께 사용하기 쉽도록 수정해야한다.
     * @return
     * @throws IOException
     */
    private static List<String> prepareTrainAndTest(int trainSize, int testSize, int numberOfTimesteps) throws IOException {
        Path rawPath = Paths.get(baseDir.getAbsolutePath() + "/passengers_raw.csv");

        List<String> rawStrings = Files.readAllLines(rawPath, Charset.defaultCharset());

        //새 파일을 생성하기 전에 모든 파일을 제거하자.
        FileUtils.cleanDirectory(featuresDirTrain);
        FileUtils.cleanDirectory(labelsDirTrain);
        FileUtils.cleanDirectory(featuresDirTest);
        FileUtils.cleanDirectory(labelsDirTest);

        for (int i = 0; i < trainSize; i++) {
            Path featuresPath = Paths.get(featuresDirTrain.getAbsolutePath() + "/train_" + i + ".csv");
            Path labelsPath = Paths.get(labelsDirTrain + "/train_" + i + ".csv");
            int j;
            for (j = 0; j < numberOfTimesteps; j++) {
                Files.write(featuresPath,rawStrings.get(i+j).concat(System.lineSeparator()).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
            }
            Files.write(labelsPath,rawStrings.get(i+j).concat(System.lineSeparator()).getBytes(),StandardOpenOption.APPEND, StandardOpenOption.CREATE);
        }

        for (int i = testSize; i < testSize+trainSize; i++) {
            Path featuresPath = Paths.get(featuresDirTest + "/test_" + i + ".csv");
            Path labelsPath = Paths.get(labelsDirTest + "/test_" + i + ".csv");
            int j;
            for (j = 0; j < numberOfTimesteps; j++) {
                Files.write(featuresPath,rawStrings.get(i+j).concat(System.lineSeparator()).getBytes(),StandardOpenOption.APPEND, StandardOpenOption.CREATE);
            }
            Files.write(labelsPath,rawStrings.get(i+j).concat(System.lineSeparator()).getBytes(),StandardOpenOption.APPEND, StandardOpenOption.CREATE);
        }

        return rawStrings;
    }
}
