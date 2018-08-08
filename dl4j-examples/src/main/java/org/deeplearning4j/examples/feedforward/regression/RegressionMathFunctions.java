package org.deeplearning4j.examples.feedforward.regression;

import org.deeplearning4j.examples.feedforward.regression.function.MathFunction;
import org.deeplearning4j.examples.feedforward.regression.function.SinXDivXMathFunction;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**예제: 특정 수학 함수를 재현할 수 있는 신경망을 훈련하고 결과를 표시한다.
 * 신경망 출력은 'plotFrequency' 에포크마다 플로팅 된다. 따라서 플롯은 학습이 진행될수록 신경망 예측의 정확성을 보여준다.
 * 여기서는 다양한 수학 함수를 구현한다.
 * 회귀 분석을 위해 신경망 출력 계층에서 항등 함수를 사용한다.
 *
 * @author 알렉스 블랙
 */
public class RegressionMathFunctions {

    // 재현성을 위해 난수 생성기 시드를 고정
    public static final int seed = 12345;
    // 미니 배치당 반복 횟수
    public static final int iterations = 1;
    // 에포크 횟수 (전체 데이터 처리)
    public static final int nEpochs = 2000;
    // 신경망 출력을 얼마나 자주 플로팅해야하는가?
    public static final int plotFrequency = 500;
    // 데이터 좌표 개수
    public static final int nSamples = 1000;
    // 배치 크기: 즉, 에포크는 nSample/batchSize번 매개변수 업데이트한다
    public static final int batchSize = 100;
    // 신경망 학습률
    public static final double learningRate = 0.01;
    public static final Random rng = new Random(seed);
    public static final int numInputs = 1;
    public static final int numOutputs = 1;


    public static void main(final String[] args){

        // 서로 다른 신경망에서 다양한 기능을 수행하려면 다음 두 옵션을 전환해라
        final MathFunction fn = new SinXDivXMathFunction();
        final MultiLayerConfiguration conf = getDeepDenseLayerNetworkConfiguration();

        // 학습 데이터 생성
        final INDArray x = Nd4j.linspace(-10,10,nSamples).reshape(nSamples, 1);
        final DataSetIterator iterator = getTrainingData(x,fn,batchSize,rng);

        // 신경망 생성
        final MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));


        // 신경망을 전체 데이터셋에서 학습하고 주기적으로 평가
        final INDArray[] networkPredictions = new INDArray[nEpochs/ plotFrequency];
        for( int i=0; i<nEpochs; i++ ){
            iterator.reset();
            net.fit(iterator);
            if((i+1) % plotFrequency == 0) networkPredictions[i/ plotFrequency] = net.output(x, false);
        }

        // 대상 데이터 및 신경망 예측을 플롯
        plot(fn,x,fn.getFunctionValues(x),networkPredictions);
    }

    /** 노드 50개를 가진 2개의 은닉 계층으로 구성된 신경망을 반환
     */
    private static MultiLayerConfiguration getDeepDenseLayerNetworkConfiguration() {
        final int numHiddenNodes = 50;
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.TANH).build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.TANH).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();
    }

    /** 학습용 DataSetIterator 생성
     * @param x X 값
     * @param function 평가할 함수
     * @param batchSize 배치 크기(DataSetIterator.next() 호출마다 불려지는 입력 데이터 개수)
     * @param rng 난수 생성기 (재현성을 위해 외부에서 난수 생성기 전달)
     */
    private static DataSetIterator getTrainingData(final INDArray x, final MathFunction function, final int batchSize, final Random rng) {
        final INDArray y = function.getFunctionValues(x);
        final DataSet allData = new DataSet(x,y);

        final List<DataSet> list = allData.asList();
        Collections.shuffle(list,rng);
        return new ListDataSetIterator(list,batchSize);
    }

    // 데이터를 플롯
    private static void plot(final MathFunction function, final INDArray x, final INDArray y, final INDArray... predicted) {
        final XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet,x,y,"True Function (Labels)");

        for( int i=0; i<predicted.length; i++ ){
            addSeries(dataSet,x,predicted[i],String.valueOf(i));
        }

        final JFreeChart chart = ChartFactory.createXYLineChart(
                "Regression Example - " + function.getName(),      // 차트 제목
                "X",                        // x 축 레이블
                function.getName() + "(X)", // y 축 레이블
                dataSet,                    // 데이터
                PlotOrientation.VERTICAL,
                true,                       // 범례 추가
                true,                       // 툴팁 추가
                false                       // URL 제거
        );

        final ChartPanel panel = new ChartPanel(chart);

        final JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();

        f.setVisible(true);
    }

    private static void addSeries(final XYSeriesCollection dataSet, final INDArray x, final INDArray y, final String label){
        final double[] xd = x.data().asDouble();
        final double[] yd = y.data().asDouble();
        final XYSeries s = new XYSeries(label);
        for( int j=0; j<xd.length; j++ ) s.add(xd[j],yd[j]);
        dataSet.addSeries(s);
    }
}
