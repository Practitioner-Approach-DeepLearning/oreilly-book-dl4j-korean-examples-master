package org.deeplearning4j.examples.misc.presave;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
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
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;


/**
 *
 * YOU NEED TO RUN PreSave first
 * before using this class.
 *
 * This class demonstrates how to  use a pre saved
 * dataset to minimize time spent loading data.
 * This is critical if you want to have ANY speed
 * with deeplearning4j.
 *
 * Deeplearning4j does not force you to use a particular data format.
 * Unfortunately this flexibility means that many people get training wrong.
 *
 * With more flexibility comes more complexity. This class demonstrates how
 * to minimize time spent training while using an existing iterator and an existing dataset.
 *
 * We use an {@link AsyncDataSetIterator}  to load data in the background
 * and {@link PreSave} to pre save the data to 2 specified directories,
 * trainData and testData
 *
 *
 *
 *
 * Created by agibsonccc on 9/16/15.
 * Modified by dmichelin on 12/10/2016 to add documentation
 */
/**
 *
 * 이 클래스를 시작하기 전에 PreSave를 먼저 실행해야한다.
 *
 * 이 클래스는 어떻게 미리 저장된 데이터셋을 로딩하는데 시간을 단축시킬지 알아보는 클래스이다.
 * 이것은 deeplearning4j를 수행하는 시간을 자유자제로 다루는데 큰 도움이 될 것이다.
 * 
 * Deeplearning4j는 특정 데이터 형식을 사용하도록 강요하지 않는다.
 * 불행히도 이러한 융통성은 많은 사람들이 잘못된 방식으로 이것을 많이 사용한다는 것을 의미한다.
 *
 * 
 * 유연성이 높아지면 복잡성이 커진다. 
 * 이 클래스는 기존 반복자와 기존 데이터셋을 사용하는 동안 학습에 소요되는 시간을 최소화하는 방법을 보여준다.
 *
 * 백그라운드에서 데이터를 로드하기 위해 {@link AsyncDataSetIterator} 을 사용하고 
 * 학습데이터와 테스트 데이터를 미리 저장하기 위해 {@link PreSave}를 사용한다.
 *
 *
 *
 *
 * Created by agibsonccc on 9/16/15.
 * Modified by dmichelin on 12/10/2016 to add documentation
 */
public class LoadPreSavedLenetMnistExample {
    private static final Logger log = LoggerFactory.getLogger(LoadPreSavedLenetMnistExample.class);

    public static void main(String[] args) throws Exception {
        int nChannels = 1; //입력 채널 갯수
        int outputNum = 10; // 가능한 결과 수
        int nEpochs = 1; // 학습 에포크 수
        int iterations = 1; //학습 반복 횟수
        int seed = 123; //

        /*
            미리 저장된 데이터를 로드한다. 참고 : 먼저 PreSave를 실행해 준다.

         */
        log.info("Load data....");
        /**
         * Note the {@link ExistingMiniBatchDataSetIterator}
         * takes in a pattern of "mnist-train-%d.bin"
         * and "mnist-test-%d.bin"
         *
         * The %d is an integer. You need a %d
         * as part of the template in order to have
         * the iterator work.
         * It uses this %d integer to
         * index what number it is in the current dataset.
         * This is how pre save will save the data.
         *
         * If you still don't understand what this is, please see an example with printf in c:
         * http://www.sitesbay.com/program/c-program-print-number-pattern
         * and in java:
         * https://docs.oracle.com/javase/tutorial/java/data/numberformat.html
         */
        /**
         * {@link ExistingMiniBatchDataSetIterator}가 mnist-train-$d.bin, mnist-test-%d.bin 패턴을 가질 때 주의사항
         *
         * %d는 정수다. 반복자가 작동하려면 템플릿의 일부로 %d가 필요하다.
         * 이 %d 정수를 사용하여 현재 데이터 집합에있는 숫자를 인덱싱한다.
         * 이것은 Pre Save가 데이터를 저장하는 방법이다.
         * 
         * 이것에 대해 좀 더 알고 싶다면 아래 링크를 확인하자 (c로 구현된 버전):
         * http://www.sitesbay.com/program/c-program-print-number-pattern
         * 자바로 구현된 버전:
         * https://docs.oracle.com/javase/tutorial/java/data/numberformat.html
         */
        DataSetIterator existingTrainingData = new ExistingMiniBatchDataSetIterator(new File("trainFolder"),"mnist-train-%d.bin");
       //여기서 백그라운드로 데이터를 로드하는데 AsyncDataSetIterator를 사용한다는 점에 유의하자. 이는 데이터 로드시 디스크 병목 현상을 피하기 위해 중요하다.
        DataSetIterator mnistTrain = new AsyncDataSetIterator(existingTrainingData);
        DataSetIterator existingTestData = new ExistingMiniBatchDataSetIterator(new File("testFolder"),"mnist-test-%d.bin");
        DataSetIterator mnistTest = new AsyncDataSetIterator(existingTestData);

        /*
           신경망 구축
         */
        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations) //위와 같은 학습 반복
            .regularization(true).l2(0.0005)
                /*
                    learningRateDecayPolicy에 대한 주석을 해제해서 편향과 학습지연현상에 대한 내용을 확인하자.
                 */
            .learningRate(.01)//.biasLearningRate(0.02)
            //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new ConvolutionLayer.Builder(5, 5)
                //nIn과 nOut는 깊이를 지정한다. 여기서 nChannels가 있고 nOut은 적용 할 필터의 수이다.
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
                //이후 계층에서 nIn을 지정할 피룡는 없다. 
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
            .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
            .backprop(true).pretrain(false).build();

        /*
        Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
        .setInputType (InputType.convolutionalFlat (28,28,1))행은 몇 가지 작업을 수행한다.
        (a) 컨벌루션 / 서브 샘플링 계층과 Dense 계층 사이의 변환 같은 것을 처리하는 프리 프로세서를 추가한다.
        (b) 몇몇 추가된 설정을 확인한다.
        (c) 필요하다면 이전 레이어의 크기를 기반으로 각 레이어의 nIn (입력 뉴런 수 또는 CNN의 경우 입력 깊이) 값을 설정한다. (단, 사용자가 수동으로 설정 한 값은 무시하지 않는다.)

        InputTypes는 CNN뿐만 아니라 다른 레이어 유형 (RNN, MLP 등)에도 사용할 수 있다.
        일반적인 이미지 (ImageRecordReader를 사용하는 경우)에는 InputType.convolutional (height, width, depth)를 사용하자.
        MNIST 레코드 판독기는 28x28 픽셀 그레이 스케일 (nChannels = 1) 이미지를 "병합 된" 행 벡터 형식 
        (즉, 1x784 벡터)로 출력하는 특수한 경우이므로 여기서 사용 된 "convolutionalFlat"는 입력 유형이다.
        */

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for( int i = 0; i < nEpochs; i++ ) {
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
