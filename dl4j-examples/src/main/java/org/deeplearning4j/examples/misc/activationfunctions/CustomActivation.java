package org.deeplearning4j.examples.misc.activationfunctions;

import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 이번 예제를 이용해서 커스텀 활성화 함수를 학습 파라미터를 입력하지 않고 구현하는 방법에 대해 알아보자
 * 이번에는 BaseActivationFunction을 상속하여 커스텀 활성화 함수를 만들어보고 여기에 나와있는 메소드를 구현해 보자.
 *
 * IMPORTANT: Do not forget gradient checks. Refer to these in the deeplearning4j repo,
 * 중요: 경사도를 체크하는 것을 잊지말자. deeplearning4j 저장소에서 아래의 경로를 참고하자.
 * deeplearning4j-core/src/test/java/org/deeplearning4j/gradientcheck/LossFunctionGradientCheck.java
 *
 * 활성화 함수의 형식은 https://arxiv.org/abs/1508.01292 이 링크의 내용을 기반으로 만들어 졌다.
 * "Compact Convolutional Neural Network Cascade for Face Detection" by Kalinovskii I.A. and Spitsyn V.G.
 *
 *      h(x) = 1.7159 tanh(2x/3)
 *
 * @author susaneraly
 */
public class CustomActivation extends BaseActivationFunction{

    /*
       전방향 전달법으로 구현하기 위한 작업:
       활성화 함수로 "in" 변환하자. 최상의 방법은 아래와 같이 변환을 수행하는 것이다.
       boolean 값을 활용하여 교육 및 테스트 중에 다른 동작을 지원할 수 있다.
    */
    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        //in 배열을 활성화 함수로 변환한다.
        // h(x) = 1.7159*tanh(2x/3)
        Nd4j.getExecutioner().execAndReturn(new Tanh(in.muli(2/3.0)));
        in.muli(1.7159);
        return in;
    }

    /*
       역전파전달법으로 구현하기 위한 작업:
       엡실론이 주어지면 모든 활성화 노드의 기울기가 역방향의 다음 기울기 집합을 계산한다. 올바른 경우는 제자리에서 수정하는 것이다.

        다르게 표현해보면,
            입력(in) -> 활성화 노드를 위한 선형 입력
            출력(out) -> 활성화 노드의 출력, 혹은 다른 표현으로 h(out)으로 표현 (h는 활성화 함수)
            엡실론 (epsilon) -> 활성화 노드의 출력에 대한 손실함수의 기울기, d(Loss) / dout

            h(in) = out;
            d(Loss)/d(in) = d(Loss)/d(out) * d(out)/d(in)
                        = epsilon * h'(in)
     */

    @Override
    public Pair<INDArray,INDArray> backprop(INDArray in, INDArray epsilon) {
        // dldz는 h'(in)으로 위의 설명을 나타낸다
        //
        //      h(x) = 1.7159*tanh(2x/3);
        //      h'(x) = 1.7159*[tanh(2x/3)]' * 2/3
        INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new Tanh(in.muli(2/3.0)).derivative());
        dLdz.muli(2/3.0);
        dLdz.muli(1.7159);

        //Multiply with epsilon
        dLdz.muli(epsilon);
        return new Pair<>(dLdz, null);
    }

}
