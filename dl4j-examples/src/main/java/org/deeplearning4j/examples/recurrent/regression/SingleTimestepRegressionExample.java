package org.deeplearning4j.examples.recurrent.regression;


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

/**
 * 이 예제는 Jason Brownlee가 작성한 카라스(karas)를 활용하여 회귀분석을 참고했다. 아래 내용으로 확인이 가능하다.
 * http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
 *
 * LSTM을 사용한 단일 시간 단계 회귀를 보여준다.
 */

public class SingleTimestepRegressionExample {
    private static final Logger LOGGER = LoggerFactory.getLogger(SingleTimestepRegressionExample.class);

    private static File baseDir = new File("dl4j-examples/src/main/resources/rnnRegression");

    public static void main(String[] args) throws Exception {

        int miniBatchSize = 32;

        // ----- 학습데이터 호출 -----
        SequenceRecordReader trainReader = new CSVSequenceRecordReader(0, ";");
        trainReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath() + "/passengers_train_%d.csv", 0, 0));

        //회귀 분석의 경우 numPossibleLabels가 사용되지 않습니다. -1로 설정하자.
        DataSetIterator trainIter = new SequenceRecordReaderDataSetIterator(trainReader, miniBatchSize, -1, 1, true);

        SequenceRecordReader testReader = new CSVSequenceRecordReader(0, ";");
        testReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath() + "/passengers_test_%d.csv", 0, 0));
        DataSetIterator testIter = new SequenceRecordReaderDataSetIterator(testReader, miniBatchSize, -1, 1, true);

        //단일 데이터셋만 가지고 있으므로 이터레이터에서 데이터 세트를 만든다.
        DataSet trainData = trainIter.next();
        DataSet testData = testIter.next();

        //레이블을 포함한 데이터 표준화 (fitLabel = true)
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fitLabel(true);
        normalizer.fit(trainData);              //교육 자료 통계 수집

        normalizer.transform(trainData);
        normalizer.transform(testData);

        // ----- 신경망 설정 -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(140)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .learningRate(0.0015)
            .list()
            .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10)
                .build())
            .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY).nIn(10).nOut(1).build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.setListeners(new ScoreIterationListener(20));

        // ----- 신경망을 학습하여 각 에포크마다 테스트 세트 성능 평가 -----
        int nEpochs = 300;

        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainData);
            LOGGER.info("Epoch " + i + " complete. Time series evaluation:");

            //단일 열 입력에 대한 회귀 분석 실행
            RegressionEvaluation evaluation = new RegressionEvaluation(1);
            INDArray features = testData.getFeatureMatrix();

            INDArray lables = testData.getLabels();
            INDArray predicted = net.output(features, false);

            evaluation.evalTimeSeries(lables, predicted);

            //로그 작성자가 통계의 열을 이동 시키므로 여기에서 sout을 수행하자.
            System.out.println(evaluation.stats());
        }

        //학습 데이터가있는 rrnTimeStemp 초기화 및 테스트 데이터 예측
        net.rnnTimeStep(trainData.getFeatureMatrix());
        INDArray predicted = net.rnnTimeStep(testData.getFeatureMatrix());

        //플로팅에 대한 데이터를 원래 값으로 되돌린다.
        normalizer.revert(trainData);
        normalizer.revert(testData);
        normalizer.revertLabels(predicted);

        //데이터없이 플롯 만들기
        XYSeriesCollection c = new XYSeriesCollection();
        createSeries(c, trainData.getFeatures(), 0, "Train data");
        createSeries(c, testData.getFeatures(), 99, "Actual test data");
        createSeries(c, predicted, 100, "Predicted test data");

        plotDataset(c);

        LOGGER.info("----- Example Complete -----");
    }

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

        // 추가 사용자 정의를 위해 플롯에 대한 참조를 얻어보자 ...
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
}

