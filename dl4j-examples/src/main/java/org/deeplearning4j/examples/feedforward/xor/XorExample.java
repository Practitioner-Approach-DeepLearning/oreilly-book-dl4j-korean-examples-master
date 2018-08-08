package org.deeplearning4j.examples.feedforward.xor;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer.Builder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * 이 기본 에제는 데이터셋을 수동으로 생성하고 기본 신경망을 학습시킨다.
 * <p>
 * 신경망은 입력 뉴런 2개, 은닉 뉴런 4개가 있는 은닉 계층 1개, 출력 뉴런 2개로 구성된다.
 * <p>
 * 여기서는 출력 뉴런을 2개로 구성했다. Evaluation 클래스는 분류당 하나의 뉴런이 필요하기 때문이다.
 *
 * @author 피터 그로즈만
 */
public class XorExample {
    public static void main(String[] args) {

        // 입력 값 나열, 입력 뉴런 각각 2개 씩 입력 받아 학습 샘플 총 4개
        INDArray input = Nd4j.zeros(4, 2);

        // 예상 출력 값이 포함된 출력 각각 2개 씩 학습 샘플 총 4개
        INDArray labels = Nd4j.zeros(4, 2);

        // 첫번째 데이터셋 생성
        // 첫번재 입력이 0, 두번째 입력도 0
        input.putScalar(new int[]{0, 0}, 0);
        input.putScalar(new int[]{0, 1}, 0);
        // 그러면 첫번째 출력은 거짓이며 두번째 출력은 0이다 (클래스 주석 참조)
        labels.putScalar(new int[]{0, 0}, 1);
        labels.putScalar(new int[]{0, 1}, 0);

        // 첫번재 입력이 1 두번째 입력도 0
        input.putScalar(new int[]{1, 0}, 1);
        input.putScalar(new int[]{1, 1}, 0);
        // 그러면 xor은 참이므로 두번째 출력 뉴런이 1이 된다
        labels.putScalar(new int[]{1, 0}, 0);
        labels.putScalar(new int[]{1, 1}, 1);

        // 위와 동일
        input.putScalar(new int[]{2, 0}, 0);
        input.putScalar(new int[]{2, 1}, 1);
        labels.putScalar(new int[]{2, 0}, 0);
        labels.putScalar(new int[]{2, 1}, 1);

        // 두 입력이 모두 1이면 xor은 다시 거짓이다. 첫번째 출력이 1이어야 한다.
        input.putScalar(new int[]{3, 0}, 1);
        input.putScalar(new int[]{3, 1}, 1);
        labels.putScalar(new int[]{3, 0}, 1);
        labels.putScalar(new int[]{3, 1}, 0);

        // 데이터셋 객체 생성
        DataSet ds = new DataSet(input, labels);

        // 신경망 구성 설정
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        // 1,000개 이상 학습하거나 학습률을 높이기 위해 얼마나 반복해야 하는가? - 시행 착오를 통해 현재 값을 도출함
        builder.iterations(10000);
        // 학습률
        builder.learningRate(0.1);
        // 난수 생성기의 시드 고정. 이 애플리케이션은 실행할 때마다 동일한 결과가 나온다. ds.shuffle()과 같은 작업을 별도로 수행하면 결과가 달라질 수 있다.
        builder.seed(123);
        // 이 신경망은 소규모 신경망이므로 사용하지 않음. 더 큰 신경망에서 사용할 경우 신경망이 학습 데이터에서 암기(특징을 습득)하는데 도움이 될 수 있다.
        builder.useDropConnect(false);
        // 오차 평면에서 이동하기 위한 표준 알고리즘이다. 이 알고리즘이 가장 잘 동작했다. LINE_GRADIENT_DESCENT나 CONJUGATE_GRADIENT도
        // 좋은 결과를 보여준다.
        builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        // 0으로 편향 초기화 - 여러번 테스트해 결정
        builder.biasInit(0);
        // ""http://deeplearning4j.org/architecture" 발췌: 신경망은 한번에 5~10개의 요소를 병렬로 수집해 입력을 보다 빠르고 정확하게 처리할 수 있다.
        // 이 예제는 데이터셋이 미니 배치 크기보다도 작기 때문에 미니 배치가 필요없다.
        builder.miniBatch(false);

        // 계층 2개로 구성된 다층 신경망 생성(입력 계층을 제외하고 출력 계층을 포함)
        ListBuilder listBuilder = builder.list();

        DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();
        // 입력 두개 연결 - 동시에 입력 뉴런의 수를 정의한다. 왜나하면 첫번째 비입력 계층이기 때문이다.
        hiddenLayerBuilder.nIn(2);
        // 출력 연결 개수, nOut은 동시에 이 계층의 뉴런 개수를 의미한다.
        hiddenLayerBuilder.nOut(4);
        // 출력을 시그모이드 함수를 통과시켜 0과 1사이 범위로 출력 값을 만든다.
        hiddenLayerBuilder.activation(Activation.SIGMOID);
        // 0과 1 사이의 임의의 값을 초기화 가중치로 설정한다.
        hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
        hiddenLayerBuilder.dist(new UniformDistribution(0, 1));

        // 계층 0으로 설정 및 구축
        listBuilder.layer(0, hiddenLayerBuilder.build());

        // MCXENT 또는 NEGATIVELOGLIKELIHOOD(수학적으로 동일함) - 이 함수는 오차-값 ('비용' 또는 '손실 함수 값')을 계산한다.
        // (이 예제처럼 상호 배타적인 클래스로) 분류하는 경우, 소프트맥스 활성화 함수와 함께 멀티 클래스 교차 엔트로피를 사용하라.
        Builder outputLayerBuilder = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
        // 이전 계층의 뉴런과 동일한 개수여야 한다.
        outputLayerBuilder.nIn(4);
        // 이 계층에 있는 뉴런 2개
        outputLayerBuilder.nOut(2);
        outputLayerBuilder.activation(Activation.SOFTMAX);
        outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
        outputLayerBuilder.dist(new UniformDistribution(0, 1));
        listBuilder.layer(1, outputLayerBuilder.build());

        // 이 신경망은 사전 학습 단계 없음
        listBuilder.pretrain(false);

        // 필수인 것 같다
        // agibsonccc 말을 인용하면: 일반적으로 오토인코더나 rbms를 사용할 때 이전 계층의 잘 튜닝된 가중치를 변경하지 않고 사전 학습/파인 튜닝을 사용하고자할 때만
        // pretrain(true)로 설정해야 한다.
        listBuilder.backprop(true);

        // 신경망을 구축하고 초기화한 다음, 모두 올바르게 구성되어 있는지 확인
        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        // 매개변수 업데이트를 100번 할 때마다 오차를 출력하는 리스너 추가
        net.setListeners(new ScoreIterationListener(100));

        // GravesLSTMCharModellingExample의 C&P
        // 신경망(및 각 계층)의 매개변수개수를 출력한다.
        Layer[] layers = net.getLayers();
        int totalNumParams = 0;
        for (int i = 0; i < layers.length; i++) {
            int nParams = layers[i].numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);

        // 여기서 실제로 학습이 이루어진다.
        net.fit(ds);

        // 모든 학습 샘플을 위한 출력 생성
        INDArray output = net.output(ds.getFeatureMatrix());
        System.out.println(output);

        // Evaluation에서  올바른(가장 높은 값을 갖는) 출력의 빈도에 관한 통계를 출력한다.
        Evaluation eval = new Evaluation(2);
        eval.eval(ds.getLabels(), output);
        System.out.println(eval.stats());

    }
}
