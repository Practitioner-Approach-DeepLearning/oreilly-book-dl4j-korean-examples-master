package org.deeplearning4j.examples.convolution;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
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
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 9/16/15에 agibsonccc가 생성.
 */
public class LenetMnistExample {
    private static final Logger log = LoggerFactory.getLogger(LenetMnistExample.class);

    public static void main(String[] args) throws Exception {
        int nChannels = 1; // 입력 채널 개수
        int outputNum = 10; // 출력 가능 개수
        int batchSize = 64; // 테스트 배치 크기
        int nEpochs = 1; // 학습 에포크 횟수
        int iterations = 1; // 학습 반복 횟수
        int seed = 123; //

        /*
            학습 반복마다 배치 크기를 사용하는 반복자 생성
         */
        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);

        /*
            신경망 구축
         */
        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations) // 학습 반복 횟수 설정
                .regularization(true).l2(0.0005)
                /*
                    학습 감쇠 및 편향을 적용하려면 아래 주석을 해제할 것
                 */
                .learningRate(.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        // nIn과 nOut으로 깊이 지정. nIn은 채널 수고 nOut은 필터 수다.
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        // 이 계층부터는 nIn을 설정할 필요가 없음
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1)) // 아래 주석 참조
                .backprop(true).pretrain(false).build();

        /*
        .setInputType(InputType.convolutionalFlat(28,28,1)) 행은 몇가지 작업을 수행한다.
        (a) 합성곱 / 부분 샘플링 / 완전 연결계층 간 전환 등을 처리하는 전처리기를 추가
        (b) 몇가지 구성 검증을 추가로 수행
        (c) 필요하다면 이전 계층의 크기를 기반으로 각 계층에 대한 nIn (합성곱 신경망의 경우 입력 뉴런 개수 또는 입력 깊이) 값을 설정
            (그러나 사용자가 수동으로 설정한 값을 덮어쓰지는 않음)

        InputTypes은 합성곱 신경망 뿐만 아니라 다른 계층 유형(순환 신경망, 다층 퍼셉트론 등)에도 사용할 수 있다.
        일반적인 이미지(ImageRecordReader를 사용하는 경우)에는 InputType.convolutional(height, width, depth)를 사용하라.
        MNIST 레코드 리더는 28x28 픽셀 그레이 스케일 (채널 1개) 이미지를 "평평한" 행 벡터 형태(1x784 벡터)로 출력하는 특수한 경우이므로
        여기서는 "convolutionalFlat" 입력 유형이 사용됐다.
        */

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(mnistTrain);
            log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while(mnistTest.hasNext()){
                DataSet ds = mnistTest.next();
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);

            }
            log.info(eval.stats());
            mnistTest.reset();
        }
        log.info("****************Example finished********************");
    }
}
