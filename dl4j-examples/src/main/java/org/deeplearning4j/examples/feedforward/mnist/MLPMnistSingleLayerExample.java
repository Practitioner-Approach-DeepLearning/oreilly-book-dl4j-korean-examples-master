package org.deeplearning4j.examples.feedforward.mnist;


import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/** MNIST 데이터셋(http://yann.lecun.com/exdb/mnist/)의 숫자 분류에 적용되는 간단한 다층 퍼셉트론(MLP)
 *
 * 이 파일은 하나의 입력 계층과 하나의 은닉 계층으로 구성한다.
 *
 * 입력 계층의 입력 차원은 numRows * numColumns개이며 여기서 이 변수는 이미지의 수직, 수평 픽셀수를 의미한다.
 * 이 계층은 RELU 활성화 함수를 사용한다. 이 계층의 가중치는 급격한 상승을 방지하기 위해 그자비에 초기화
 * (https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
 * 를 사용해 초기화한다. 이 계층은 출력 신호 1,000개를 은닉 계층으로 보낸다.
 *
 * 은닉 계층의 입력 차원은 1000개다. 이들은 입력 계층에서 공급된다. 또한 이 계층의 가중치는 그자비에 초기화를 사용해 초기화된다.
 * 이 계층은 정규화된 합계가 1이 될 수 있도록 10개의 출력을 모두 정규화하는 소프트맥스를 의 활성화 함수로 사용한다.
 * 정규화된 값 중 가장 높은 값을 예측된 클래스로 선택한다.
 *
 */
public class MLPMnistSingleLayerExample {

    private static Logger log = LoggerFactory.getLogger(MLPMnistSingleLayerExample.class);

    public static void main(String[] args) throws Exception {
        // 입력 이미지의 행과 열 개수
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10; // 출력 클래스 개수
        int batchSize = 128; // 에포크당 배치 크기
        int rngSeed = 123; // 재현성을 위해 난수 생성 시드 고정
        int numEpochs = 15; // 수행할 에포크 횟수

        // DataSetIterators 생성
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed) // 재현성을 위해 난수 생성 시드 포함
                // 확률적 경사 하강법을 최적화 알고리즘으로 사용
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006) // 학습률 명시
                .updater(Updater.NESTEROVS).momentum(0.9) // 학습률 변화율 지정
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder() // 그자비에 초기화 포함한 첫번째 입력 계층을 생성
                        .nIn(numRows * numColumns)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // 은닉 계층 생성
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true) // 가중치 갱신을 위해 역전파 사용
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        // 반복 마다 점수 출력
        model.setListeners(new ScoreIterationListener(1));

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            model.fit(mnistTrain);
        }


        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); // 클래스 10개에 대한 평가 객체 생성
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); // 신경망 예측 가져오기
            eval.eval(next.getLabels(), output); // 정답과 예측 비교하기
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");

    }

}
