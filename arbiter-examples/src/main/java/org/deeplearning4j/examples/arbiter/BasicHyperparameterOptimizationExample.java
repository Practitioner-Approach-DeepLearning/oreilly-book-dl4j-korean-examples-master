package org.deeplearning4j.examples.arbiter;

import org.deeplearning4j.arbiter.DL4JConfiguration;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.data.DataSetIteratorProvider;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.candidategenerator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.ui.ArbiterUIServer;
import org.deeplearning4j.arbiter.optimize.ui.listener.UIOptimizationRunnerStatusListener;
import org.deeplearning4j.arbiter.saver.local.multilayer.LocalMultiLayerNetworkSaver;
import org.deeplearning4j.arbiter.scoring.multilayer.TestSetAccuracyScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * 이 예제는 아비터를 사용해 신경망 하이퍼파라미터 두 개를 임의 탐색으로 최적화하는 기본적인 하이퍼파라미터 최적화 예제다.
 * 여기서 최적화할 하이퍼파라미터는 학습률과 계층 크기로, 간단한 다층 퍼셉트론을 사용해 MNIST 데이터로 검색을 수행한다.
 *
 *  이 예제는 UI를 제공한다. 기본 UI 주소는 다음과 같다: http://localhost:8080/arbiter
 *
 * @author 알렉스 블랙
 */
public class BasicHyperparameterOptimizationExample {

    public static void main(String[] args) throws Exception {


        // 먼저, 하이퍼파라미터 구성 공간을 설정하라. MultiLayerConfiguration과 비슷하지만
        // 각 하이퍼파라미터는 고정값이나 최적화할 값을 가질 수 있다.

        ParameterSpace<Double> learningRateHyperparam = new ContinuousParameterSpace(0.0001, 0.1);  // 값은 0.0001과 0.1(포함) 사이에서 무작위로 균등하게 생성된다.
        ParameterSpace<Integer> layerSizeHyperparam = new IntegerParameterSpace(16,256);            // 정수 값은 16과 256(포함) 사이에서 무작위로 균등하게 생성된다.

        MultiLayerSpace hyperparameterSpace = new MultiLayerSpace.Builder()
            // 다음 몇가지 옵션은 모델에서 고정된다.
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .regularization(true)
            .l2(0.0001)
            // 여러 값을 테스트할 하이퍼파라미터: 학습률
            .learningRate(learningRateHyperparam)
            .addLayer( new DenseLayerSpace.Builder()
                    // 이 신경망의 고정값
                    .nIn(784)  // 입력 크기 고정: 28 x 28 = 784 픽셀 크기의 MNIST
                    .activation("relu")
                    // 추론할 하이퍼파라미터: 계층 크기
                    .nOut(layerSizeHyperparam)
                    .build())
            .addLayer( new OutputLayerSpace.Builder()
                //nIn: 이전 계층의 nOut과 동일한 값으로 설정
                .nIn(layerSizeHyperparam)
                // 마지막 하이퍼파라미터: 출력 계층 크기 고정
                .nOut(10)
                .activation("softmax")
                .lossFunction(LossFunctions.LossFunction.MCXENT)
                .build())
            .pretrain(false).backprop(true).build();


        // 이제, 몇 가지 구성 옵션을 정의해야 한다.
        // (a) 후보를 어떻게 고를 것인가? (임의 탐색 또는 격자 탐색)
        CandidateGenerator<DL4JConfiguration> candidateGenerator = new RandomSearchGenerator<>(hyperparameterSpace);    //또는: new GridSearchCandidateGenerator<>(hyperparameterSpace, 5, GridSearchCandidateGenerator.Mode.RandomOrder);

        // (b) 데이터를 어떻게 제공할 것인가? 여기서는 DataSetIterators에 대한 간단한 내장 제공 데이터 공급자를 사용할 것이다.
        int nTrainEpochs = 2;
        DataSetIterator mnistTrain = new MultipleEpochsIterator(nTrainEpochs, new MnistDataSetIterator(64,true,12345));
        DataSetIterator mnistTest = new MnistDataSetIterator(64,false,12345);
        DataProvider<DataSetIterator> dataProvider = new DataSetIteratorProvider(mnistTrain, mnistTest);

        // (c) 테스트 및 생성된 모델을 어떻게 저장할 것인가?
        //     이 예제에서는 디스크에 작업 디렉토리를 저장한다.
        //     이 예제의 결과는 다음과 같은 경로에 저장될 것이다.
        //     arbiterExample/0/, arbiterExample/1/, arbiterExample/2/, ...
        String baseSaveDirectory = "arbiterExample/";
        File f = new File(baseSaveDirectory);
        if(f.exists()) f.delete();
        f.mkdir();
        ResultSaver<DL4JConfiguration,MultiLayerNetwork,Object> modelSaver = new LocalMultiLayerNetworkSaver<>(baseSaveDirectory);

        // (d) 무엇을 최적화해야 하는가?
        //     이 예제에서는 분류 정확도를 사용해 테스트셋에 대해 최적화한다.
        ScoreFunction<MultiLayerNetwork,DataSetIterator> scoreFunction = new TestSetAccuracyScoreFunction();

        // (e) 언제 탐색을 중단해야 하는가? 종료 조건을 지정하라.
        //     이 예제에서는 15분이 지나거나 20 후보를 고르면 중단한다.
        TerminationCondition[] terminationConditions = {new MaxTimeCondition(15, TimeUnit.MINUTES), new MaxCandidatesCondition(20)};



        // 위 구성 옵션을 하나로 묶음
        OptimizationConfiguration<DL4JConfiguration, MultiLayerNetwork, DataSetIterator, Object> configuration
            = new OptimizationConfiguration.Builder<DL4JConfiguration, MultiLayerNetwork, DataSetIterator, Object>()
                .candidateGenerator(candidateGenerator)
                .dataProvider(dataProvider)
                .modelSaver(modelSaver)
                .scoreFunction(scoreFunction)
                .terminationConditions(terminationConditions)
                .build();

        // 로컬 머신에서 수행하도록 설정
        IOptimizationRunner<DL4JConfiguration,MultiLayerNetwork,Object> runner
            = new LocalOptimizationRunner<>(configuration, new MultiLayerNetworkTaskCreator<>());


        // UI 실행
        ArbiterUIServer server = ArbiterUIServer.getInstance();
        runner.addListeners(new UIOptimizationRunnerStatusListener(server));


        // 하이퍼파라미터 최적화 시작
        runner.execute();


        // 최적화 과정에 대한 기본적인 통계 출력
        StringBuilder sb = new StringBuilder();
        sb.append("Best score: ").append(runner.bestScore()).append("\n")
            .append("Index of model with best score: ").append(runner.bestScoreCandidateIndex()).append("\n")
            .append("Number of configurations evaluated: ").append(runner.numCandidatesCompleted()).append("\n");
        System.out.println(sb.toString());


        // 모든 결과를 가져와 최적의 결과에 대한 상세 내역 출력
        int indexOfBestResult = runner.bestScoreCandidateIndex();
        List<ResultReference<DL4JConfiguration,MultiLayerNetwork,Object>> allResults = runner.getResults();

        OptimizationResult<DL4JConfiguration,MultiLayerNetwork,Object> bestResult = allResults.get(indexOfBestResult).getResult();
        MultiLayerNetwork bestModel = bestResult.getResult();

        System.out.println("\n\nConfiguration of best model:\n");
        System.out.println(bestModel.getLayerWiseConfigurations().toJson());


        // 주의: 실행이 완료되어 JVM이 종료되면 UI 서버도 중단된다.
        // JVM을 활성 상태로 유지하기 위해 Thread.sleep(1 분)을 수행하면 신경망 구성을 볼 수 있다.
        Thread.sleep(60000);
        System.exit(0);
    }

}
