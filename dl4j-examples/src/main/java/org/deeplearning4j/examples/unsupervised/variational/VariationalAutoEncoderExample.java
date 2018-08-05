package org.deeplearning4j.examples.unsupervised.variational;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.examples.unsupervised.variational.plot.PlotUtil;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * MNIST에서 변형 오토인코더 학습 예. 
 * 본 예제는 2차원 그리드 시각화를 위해 작은 크기의 은닉 상태 Z(2개 값)을 가진다. 
 *
 * 이 예제는 학습이 끝난 후, 다음 두가지를 그린다. 
 * 1. MNIST 숫자 재구성 vs 잠재공간 (latent space)
 * 2. 학습 단계별 MNIST 테스트 셋의 잠재 공간 값. (모든 N 미니배치)
 *
 * 두 그래프 모두 상단에 슬라이더가 있다. 이를 변경하여 시간의 흐름에 따라 재구성과 잠재공간의 변화를 확인할 수 있다. 
 *
 * @author Alex Black
 */
public class VariationalAutoEncoderExample {
    private static final Logger log = LoggerFactory.getLogger(VariationalAutoEncoderExample.class);

    public static void main(String[] args) throws IOException {
        int minibatchSize = 128;
        int rngSeed = 12345;
        int nEpochs = 20;                   //학습 에포크 전체 수 

        //그래프 설정
        int plotEveryNMinibatches = 100;    //그래프로 그릴 데이터를 수집하는 빈도
        double plotMin = -5;                //최솟값 (x, y축)
        double plotMax = 5;                 //최댓값 (x, y축)
        int plotNumSteps = 16;              //plotMin과 plotMax 사이의 재구성 단계 수.

        //학습용 MNIST 데이터 
        DataSetIterator trainIter = new MnistDataSetIterator(minibatchSize, true, rngSeed);

        //신경망 설정 
        Nd4j.getRandom().setSeed(rngSeed);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .iterations(1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(1e-2)
            .updater(Updater.RMSPROP).rmsDecay(0.95)
            .weightInit(WeightInit.XAVIER)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new VariationalAutoencoder.Builder()
                .activation(Activation.LEAKYRELU)
                .encoderLayerSizes(256, 256)        // 2 인코더 계층. 각 사이즈는 256
                .decoderLayerSizes(256, 256)        // 2 디코더 계층. 각 사이즈는 256
                .pzxActivationFunction("identity")  // p(z|data) 활성화 함수
                .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction()))     // p(data|z) 베르누이 분포 (2진수 또는 0에서 1사이 데이터만 가능)
                .nIn(28 * 28)                       //입력 크기: 28x28
                .nOut(2)                            //잠재변수공간 크기: p(z|x). 일반적으로 플롯을 위해 2차원 사용.
                .build())
            .pretrain(true).backprop(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        //변형 오토인코더 계층
        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
            = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);


        //플롯을 위한 테스트 데이터
        DataSet testdata = new MnistDataSetIterator(10000, false, rngSeed).next();
        INDArray testFeatures = testdata.getFeatures();
        INDArray testLabels = testdata.getLabels();
        INDArray latentSpaceGrid = getLatentSpaceGrid(plotMin, plotMax, plotNumSteps);              //plotMin과 plotMax 사이의 X/Y 값

        //나중에 플롯팅 하기 위해 데이터를 저장하는 리스트
        List<INDArray> latentSpaceVsEpoch = new ArrayList<>(nEpochs + 1);
        INDArray latentSpaceValues = vae.activate(testFeatures, false);                     //학습 시작 전 잠재 공간 값 수집, 저장
        latentSpaceVsEpoch.add(latentSpaceValues);
        List<INDArray> digitsGrid = new ArrayList<>();

        //학습 수행 
        int iterationCount = 0;
        for (int i = 0; i < nEpochs; i++) {
            log.info("Starting epoch {} of {}",(i+1),nEpochs);
            while (trainIter.hasNext()) {
                DataSet ds = trainIter.next();
                net.fit(ds);

                //모든 N=100 미니 배치마다:
                // (a) 플롯팅을 위한 테스트셋 잠재공간 값을 수집
                // (b) 그리드 각 지점의 재구성 수집
                if (iterationCount++ % plotEveryNMinibatches == 0) {
                    latentSpaceValues = vae.activate(testFeatures, false);
                    latentSpaceVsEpoch.add(latentSpaceValues);

                    INDArray out = vae.generateAtMeanGivenZ(latentSpaceGrid);
                    digitsGrid.add(out);
                }
            }

            trainIter.reset();
        }

        // MNIST 테스트셋 그리기 - 잠재공간 vs. 반복 (기본적으로 100 미니배치)
        PlotUtil.plotData(latentSpaceVsEpoch, testLabels, plotMin, plotMax, plotEveryNMinibatches);

        // 재구성 - 잠재공간 vs. 그리드
        double imageScale = 2.0;        //숫자를 확대하려면 이 값을 늘이거나 줄이면 된다. 
        PlotUtil.MNISTLatentSpaceVisualizer v = new PlotUtil.MNISTLatentSpaceVisualizer(imageScale, digitsGrid, plotEveryNMinibatches);
        v.visualize();
    }


    //2차원 그리드 (x,y) 반환: x, y 각각 plotMin에서 plotMax까지 
    private static INDArray getLatentSpaceGrid(double plotMin, double plotMax, int plotSteps) {
        INDArray data = Nd4j.create(plotSteps * plotSteps, 2);
        INDArray linspaceRow = Nd4j.linspace(plotMin, plotMax, plotSteps);
        for (int i = 0; i < plotSteps; i++) {
            data.get(NDArrayIndex.interval(i * plotSteps, (i + 1) * plotSteps), NDArrayIndex.point(0)).assign(linspaceRow);
            int yStart = plotSteps - i - 1;
            data.get(NDArrayIndex.interval(yStart * plotSteps, (yStart + 1) * plotSteps), NDArrayIndex.point(1)).assign(linspaceRow.getDouble(i));
        }
        return data;
    }
}
