package org.deeplearning4j.examples.misc.earlystopping;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**Early stopping example on a subset of MNIST
 * Idea: given a small subset of MNIST (1000 examples + 500 test set), conduct training and get the parameters that
 * have the minimum test set loss
 * This is an over-simplified example, but the principles used here should apply in more realistic cases.
 *
 * For further details on early stopping, see http://deeplearning4j.org/earlystopping.html
 *
 * @author Alex Black
 */
/**MNIST 일부를 사용해서 빠르게 중단되는 예제
 * 아이디어: 주어진 MNIST의 작은 집합 (1000개 사례 + 500개 시험셋)을 바탕으로 학습하고 최소 시험 셋 손실을 갖는 파라미터를 얻는다.
 * 이것은 지나치게 단순화된 예제이다. 하지만 여기에 사용 된 원칙이 보다 현실적인 경우에 적용되기도 한다.
 *
 * 이것의 자세한 내용을 알고 싶다면 http://deeplearning4j.org/earlystopping.html 을 참고하자
 *
 * @author Alex Black
 */

public class EarlyStoppingMNIST {

    public static void main(String[] args) throws Exception {

        //Configure network:
        int nChannels = 1;
        int outputNum = 10;
        int batchSize = 25;
        int iterations = 1;
        int seed = 123;
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .regularization(true).l2(0.0005)
            .learningRate(0.02)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.RELU)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new ConvolutionLayer.Builder(5, 5)
                .nIn(nChannels)
                .stride(1, 1)
                .nOut(20).dropOut(0.5)
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(2, new DenseLayer.Builder()
                .nOut(500).build())
            .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutionalFlat(28, 28, 1)) //LenetMnistExample 참고
            .backprop(true).pretrain(false).build();

        //Get data:
        DataSetIterator mnistTrain1024 = new MnistDataSetIterator(batchSize,1024,false,true,true,12345);
        DataSetIterator mnistTest512 = new MnistDataSetIterator(batchSize,512,false,false,true,12345);

        String tempDir = System.getProperty("java.io.tmpdir");
        String exampleDirectory = FilenameUtils.concat(tempDir, "DL4JEarlyStoppingExample/");
        EarlyStoppingModelSaver saver = new LocalFileModelSaver(exampleDirectory);
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(50)) //최대 50 에포크까지
                .evaluateEveryNEpochs(1)
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES)) //최대 20분
                .scoreCalculator(new DataSetLossCalculator(mnistTest512, true))     //시험 셋 결과를 계산
                .modelSaver(saver)
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,configuration,mnistTrain1024);

        //조기 중단 학습 시작
        EarlyStoppingResult result = trainer.fit();
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        //score vs epoch를 출력
        Map<Integer,Double> scoreVsEpoch = result.getScoreVsEpoch();
        List<Integer> list = new ArrayList<>(scoreVsEpoch.keySet());
        Collections.sort(list);
        System.out.println("Score vs. Epoch:");
        for( Integer i : list){
            System.out.println(i + "\t" + scoreVsEpoch.get(i));
        }
    }
}
