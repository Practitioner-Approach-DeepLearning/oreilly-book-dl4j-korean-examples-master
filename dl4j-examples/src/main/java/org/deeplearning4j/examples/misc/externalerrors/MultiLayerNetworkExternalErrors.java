package org.deeplearning4j.examples.misc.externalerrors;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 이번 예제는 출력 계층 및 레이블 배열을 사용하는 대신 오류가 외부 소스에서 발생한 MultiLayerNetwork를 학습하는 방법에 대해 알려준다.
 * 이것을 적용할 수 있는 활용 사례는 강화 학습과 새로운 알고리즘의 테스트 / 개발 이다.
 * 일부 사용 사례의 경우 다음 대안을 고려할 수 있다.
 * - 사용자 손실 함수를 만든다.
 * - 사용자 (출력) 계층을 만든다.
 * 이 두가지 모두 다 DL4J에서 구현 가능하다.
 *
 * @author Alex Black
 */
public class MultiLayerNetworkExternalErrors {

    public static void main(String[] args) {

        //Create the model
        int nIn = 4;
        int nOut = 3;
        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS)
            .learningRate(0.1)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(3).build())
            .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
            .backprop(true).pretrain(false)
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        //외부 오류에 대한 기울기 계산
        int minibatch = 32;
        INDArray input = Nd4j.rand(minibatch, nIn);
        INDArray output = model.output(input);          //전방향 전달시 이것을 사용해서 에러를 계산한다.

        INDArray externalError = Nd4j.rand(minibatch, nOut);
        Pair<Gradient, INDArray> p = model.backpropGradient(externalError);  //에러 배열을 기반으로 역전파 에러를 계산한다.

        // 기울기 업데이트 : 학습률, 모멘텀 등에 적용
        // 기울기 객체를 현재 위치에서 수정
        Gradient gradient = p.getFirst();
        int iteration = 0;
        model.getUpdater().update(model, gradient, iteration, minibatch);

        // 행 벡터 기울기 배열을 가져와 파라미터에 적용하여 모델을 업데이트한다.
        INDArray updateVector = gradient.gradient();
        model.params().subi(updateVector);
    }

}
