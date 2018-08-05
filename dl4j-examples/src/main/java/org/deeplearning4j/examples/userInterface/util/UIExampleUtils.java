package org.deeplearning4j.examples.userInterface.util;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

/**
 * Created by Alex on 11/11/2016.
 */
public class UIExampleUtils {

    public static MultiLayerNetwork getMnistNetwork(){

        int nChannels = 1; // 입력 채널 수
        int outputNum = 10; // 가능한 출력 수
        int iterations = 1; // 학습 반복 단계 수
        int seed = 123; //

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations) // 위와 같이 반복 학습 
            .regularization(true).l2(0.0005)
            .learningRate(0.01)
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new ConvolutionLayer.Builder(5, 5)
                //nIn 과 nOut 은 깊이를 지정함. nIn은 nChannels이고, nOut 은 적용할 필터의 갯수임.
                .nIn(nChannels)
                .stride(1, 1)
                .nOut(20)
                .activation(Activation.LEAKYRELU)
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .build())
            .layer(2, new ConvolutionLayer.Builder(5, 5)
                //이후의 계층에서는 nIn을 지정할 필요가 없음.
                .stride(1, 1)
                .nOut(50)
                .activation(Activation.LEAKYRELU)
                .build())
            .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .build())
            .layer(4, new DenseLayer.Builder().activation(Activation.LEAKYRELU).nOut(500).build())
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutionalFlat(28,28,1))
            .backprop(true).pretrain(false).build();


        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        return net;
    }

    public static DataSetIterator getMnistData(){
        try{
            return new MnistDataSetIterator(64,true,12345);
        }catch (IOException e){
            throw new RuntimeException(e);
        }
    }

}
