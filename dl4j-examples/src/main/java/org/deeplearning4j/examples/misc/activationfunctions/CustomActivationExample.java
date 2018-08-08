package org.deeplearning4j.examples.misc.activationfunctions;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * 이 예제는 커스텀 활성화 함수를 인스턴스화하고 사용하는 방법을 보여줍니다.
 * 이 예제는 커스텀 활성화 함수에 대한 내용을 제외하고는 org.deeplearning4j.examples.feedforward.regression.RegressionSum 과 같습니다.
 */

public class CustomActivationExample {
    public static final int seed = 12345;
    public static final int iterations = 1;
    public static final int nEpochs = 500;
    public static final int nSamples = 1000;
    public static final int batchSize = 100;
    public static final double learningRate = 0.001;
    public static int MIN_RANGE = 0;
    public static int MAX_RANGE = 3;

    public static final Random rng = new Random(seed);

    public static void main(String[] args){

        DataSetIterator iterator = getTrainingData(batchSize,rng);

        //신경망 생성
        int numInput = 2;
        int numOutputs = 1;
        int nHidden = 10;
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.95)
            .list()
            //여기에 사용자 활성화 함수를 다음과 같이 다시 설정하자.
            //CustomActivation클래스의 implimentation을 참고하자

            .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
                .activation(new CustomActivation())
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(nHidden).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build()
        );
        net.init();
        net.setListeners(new ScoreIterationListener(100));

        // 전체 데이터 셋을 바탕으로 신경망 학습, 주기적으로 신경망이 개선된다.
        for( int i=0; i<nEpochs; i++ ){
            iterator.reset();
            net.fit(iterator);
        }
        // 2 개의 숫자 추가 테스트 (여기서 다른 숫자를 시도하자)
        final INDArray input = Nd4j.create(new double[] { 0.111111, 0.3333333333333 }, new int[] { 1, 2 });
        INDArray out = net.output(input, false);
        System.out.println(out);

    }

    private static DataSetIterator getTrainingData(int batchSize, Random rand){
        double [] sum = new double[nSamples];
        double [] input1 = new double[nSamples];
        double [] input2 = new double[nSamples];
        for (int i= 0; i< nSamples; i++) {
            input1[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble();
            input2[i] =  MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble();
            sum[i] = input1[i] + input2[i];
        }
        INDArray inputNDArray1 = Nd4j.create(input1, new int[]{nSamples,1});
        INDArray inputNDArray2 = Nd4j.create(input2, new int[]{nSamples,1});
        INDArray inputNDArray = Nd4j.hstack(inputNDArray1,inputNDArray2);
        INDArray outPut = Nd4j.create(sum, new int[]{nSamples, 1});
        DataSet dataSet = new DataSet(inputNDArray, outPut);
        List<DataSet> listDs = dataSet.asList();
        Collections.shuffle(listDs,rng);
        return new ListDataSetIterator(listDs,batchSize);

    }
}
