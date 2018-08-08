package org.deeplearning4j.examples.misc.customlayers.layer;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * 커스텀 계층 예제에 대한 계층 구현 클래스
 *
 * @author Alex Black
 */

public class CustomLayerImpl extends BaseLayer<CustomLayer> { //일반적인 매개 변수로 설정 클래스를 사용한다.

    public CustomLayerImpl(NeuralNetConfiguration conf) {
        super(conf);
    }


    @Override
    public INDArray preOutput(INDArray x, boolean training) {
       /*
         활성화 함수가 적용되지 전에 이 메소드는 활성화 값을 계산한다 (전방향 전달)
        Because we aren't doing anything different to a standard dense layer, we can use the existing implementation
        for this. Other network types (RNNs, CNNs etc) will require you to implement this method.
        Dense 계층과 다른 내용을 수행하지 않기 때문에 기존 것을 사용해도 된다. 다른 신경망 유형 ( RNN, CNN  등) 에서는 이 방법을 구현해야한다.
        사용자정의 계층의 경우 calcL1, calcL2, numParams 등과 같은 메서드를 구현해야 할 수도 있다.
         */

        return super.preOutput(x, training);
    }


    @Override
    public INDArray activate(boolean training) {
         /*
            이 매소드는 전방향전달을 수행한다. preOutput 메소드에 의존한다는 점에 주의하자.
            본질적으로 활성화 함수를 적용하기만 하면 된다.
            이 특별한 예제에서 두 개의 활성화 함수가 있다. 하나는 출력의 첫 번째 절반에 대한 것이고 다른 하나는 두 번째 절반에 대한 것이다.
         */

        INDArray output = preOutput(training);
        int columns = output.columns();

        INDArray firstHalf = output.get(NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2));
        INDArray secondHalf = output.get(NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns));

        IActivation activation1 = conf.getLayer().getActivationFn();
        IActivation activation2 = ((CustomLayer) conf.getLayer()).getSecondActivationFunction();

        // IActivation 함수 인스턴스는 활성화 함수를 현재 위치에서 수정한다.
        activation1.getActivation(firstHalf, training);
        activation2.getActivation(secondHalf, training);

        return output;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        /*
        이 backpropGradient 함수는 BaseLayer의 backprop gradient 구현과 비슷하다.
        주요 차이점은 두개의 활성화 함수가 이 예제에 추가되었다는 것이다.
        엡실론은 dL / da, 즉 활성화에 대한 손실 함수의 미분이다.
        활성화 배열 (즉, preOut 및 활성화 메소드의 출력)과 완전히 동일한 모양을 가진다.
        이것은 인공 신경망 문서들에서 일반적으로 사용되는 델타가 아니다.
        델타는 활성화 함수 미분을 가진 원소 단위의 곱을 수행함으로써 엡실론 ( "ε"은 dl4j의 표기법)으로부터 얻어진다.

        다음 사항에 주의하자:
        1. 결과를 위해 gradientViews를 사용한다는 점을 명심하자. gradientViews.get(...) 및 내부 편집 작업을 여기서 한다는 것에 유의하자.
            이는 DL4J가 효율을 위해 기울기에 대해 하나의 큰 배열을 사용하기 때문이다.
            이 배열(뷰)의 서브 셋은 효율적인 백 드롭 및 메모리 관리를 위해 각 계층에 분산된다.
        2. 이 메소드는 두개 값을 한쌍으로 리턴한다.
            (a) 기울기 객체 (각각 파라미터의 기울기에 대한 Map<Strong, INDArray> (다시 말하지만, 이것들은 전체 신경망 기울기 배열의 뷰 이다))
            (b) INDArray. 이 INDArray는 이후 계층으로 전달되는 엡실론 값이다. 즉,이 층의 입력에 대한 기울기이다.
        */

        INDArray activationDerivative = preOutput(true);
        int columns = activationDerivative.columns();

        INDArray firstHalf = activationDerivative.get(NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2));
        INDArray secondHalf = activationDerivative.get(NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns));

        INDArray epsilonFirstHalf = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2));
        INDArray epsilonSecondHalf = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns));

        IActivation activation1 = conf.getLayer().getActivationFn();
        IActivation activation2 = ((CustomLayer) conf.getLayer()).getSecondActivationFunction();

        //IActivation backprop method modifies the 'firstHalf' and 'secondHalf' arrays in-place, to contain dL/dz
        //IActivation backprop 메서드는 firstHalf, secondHalf 배열에 dL / dz를 포함하도록 내부 수정한다.
        activation1.backprop(firstHalf, epsilonFirstHalf);
        activation2.backprop(secondHalf, epsilonSecondHalf);

        //이 메소드의 나머지 코드는 BaseLayer.backpropGradient에서 복사하여 붙여 넣기 한 것이다.
        // INDArray delta = epsilon.muli(activationDerivative);
        if (maskArray != null) {
            activationDerivative.muliColumnVector(maskArray);
        }

        Gradient ret = new DefaultGradient();

        INDArray weightGrad = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);    //f order
        Nd4j.gemm(input, activationDerivative, weightGrad, true, false, 1.0, 0.0);
        INDArray biasGrad = gradientViews.get(DefaultParamInitializer.BIAS_KEY);
        biasGrad.assign(activationDerivative.sum(0));  //TODO: 할당 없이 수행하도록 하기

        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGrad);
        ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, biasGrad);

        INDArray epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(activationDerivative.transpose()).transpose();

        return new Pair<>(ret, epsilonNext);
    }

}
