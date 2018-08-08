package org.deeplearning4j.examples.misc.customlayers.layer;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * 커스텀 계층 예제를 위한 Layer 설정 클래스
 *
 * @author Alex Black
 */
public class CustomLayer extends FeedForwardLayer {

    private IActivation secondActivationFunction;

    public CustomLayer() {
        // JSON, YAML 형식의 설정을 사용할 수 있도록 하기 위해서 constructor에 전달되는 인자가 없어야 한다.
        // 이것이 없으면 다음과 같은 예외가 발생할 수 있다.
        //com.fasterxml.jackson.databind.JsonMappingException: No suitable constructor found for type [simple type, class org.deeplearning4j.examples.misc.customlayers.layer.CustomLayer]: can not instantiate from JSON object (missing default constructor or creator, or perhaps need to add/enable type information?)
    }

    private CustomLayer(Builder builder) {
        super(builder);
        this.secondActivationFunction = builder.secondActivationFunction;
    }

    public IActivation getSecondActivationFunction() {
        // 또한 JSON 직렬화를 위한 계층 설정 필드가 있는 경우에 setter / getter 메소드가 필요하다.
        return secondActivationFunction;
    }

    public void setSecondActivationFunction(IActivation secondActivationFunction) {
        // 또한 JSON 직렬화를 위한 계층 설정 필드가 있는 경우에 setter / getter 메소드가 필요하다.
        this.secondActivationFunction = secondActivationFunction;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        // instantiate 메소드는 설정 클래스 (즉,이 클래스)에서 구현 클래스로 이동하는지 알 수 있다.
        // 자세한 내용은 각 유형의 계층에 대해 동일하다.
        CustomLayerImpl myCustomLayer = new CustomLayerImpl(conf);
        myCustomLayer.setListeners(iterationListeners);            // 계층의 정수 인덱스가 있으면 iterationListener를 설정합니다.
        myCustomLayer.setIndex(layerIndex);

        // 파라미터 뷰 배열: Deeplearning4j에서 전체 신경망 (모든 계층)에 대한 신경망 매개 변수는 하나의 큰 배열로 나타낸다.
        // 이 파라미터 벡터의 관련 섹션은 각 계층에서 추출된다 (즉, 더 큰 배열의 하위 집합이라는 점에서 "보기"배열이다)
        // 이 행 벡터는 계층의 파라미터 수와 길이가 같다.
        myCustomLayer.setParamsViewArray(layerParamsView);

        // 계층 파라미터를 초기화하자. 예를들어 paramTable의 엔트리 (2 개의 엔트리 : [nIn, nOut]의 가중치 배열과 [1, nOut]의 편향)은 'layerParamsView'배열의 뷰이다.\
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        myCustomLayer.setParamTable(paramTable);
        myCustomLayer.setConf(conf);
        return myCustomLayer;
    }

    @Override
    public ParamInitializer initializer() {
        // 이 메소드는 이 계층 유형의 파라미터를 초기화하는 initializer를 반환한다.
        // 이 경우, DenseLayer에 사용 된 것과 동일한 DefaultParamInitializer를 사용할 수 있다.
        // 보다 복잡한 계층의 경우 커스텀 매개 변수 initializer를 구현해야 할 것이다.
        // 다양한 유형의 initializer는 아래 링크를 참고하자.
        //https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/params
        return DefaultParamInitializer.getInstance();
    }

    // 다음은 레이어를 쉽게 구성 할 수 있도록 빌더 패턴을 구현 한 것이다.
    // FeedForwardLayer.Builder 옵션을 모두 상속받는다.
    public static class Builder extends FeedForwardLayer.Builder<Builder> {

        private IActivation secondActivationFunction;

        // 이 것은 설정상의 사용자 설정 값에 대한 예제이다.
        /**
         * 이 사용자 지정 레이어 예제에 사용되는 사용자 지정 속성이다. 자세한 내용은 CustomLayerExampleReadme.md를 참고하자.
         * @param secondActivationFunction 계층의 두번째 활성화 함수
         */
        public Builder secondActivationFunction(String secondActivationFunction) {
            return secondActivationFunction(Activation.fromString(secondActivationFunction));
        }

        /**
         * 이 사용자 지정 레이어 예제에 사용되는 사용자 지정 속성이다. 자세한 내용은 CustomLayerExampleReadme.md를 참고하자.
         *
         * @param secondActivationFunction 계층의 두번째 활성화 함수
         */
        public Builder secondActivationFunction(Activation secondActivationFunction){
            this.secondActivationFunction = secondActivationFunction.getActivationFunction();
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")  //체크되지않은 cast에 대한 경고를 하지 않는다. (꼭 설정할 필요는 없다.)
        public CustomLayer build() {
            return new CustomLayer(this);
        }
    }

}
