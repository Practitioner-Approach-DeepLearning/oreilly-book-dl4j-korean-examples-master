package org.deeplearning4j.examples.misc.lossfunctions;

import lombok.EqualsAndHashCode;
import org.apache.commons.math3.ode.MainStateJacobianProvider;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;

/**
 * Created by susaneraly on 11/8/16.
 */
@EqualsAndHashCode
public class CustomLossL1L2 implements ILossFunction {

    /* 이 예제는 신경망을 훈련하는데 적용할 수 있는 맞춤형 손실 함수를 구현하는 방법에 대해 알려준다.
       모든 손실함수는 ILossFunction 인터페이스를 구현해야 한다.
       손실함수 구현방법은 아래와 같다.
       L = (y - y_hat)^2 +  |y - y_hat|
        y는  true 라벨, y_hat은 추측 결과이다.
     */

    private static Logger logger = LoggerFactory.getLogger(CustomLossL1L2.class);

    /*
    손실 함수에 따라 수정이 필요하다.
        scoreArray는 단일 데이터 요소의 손실을 계산한다. 즉, 일괄 처리 크기는 1이다. 신경망 출력의 모양과 크기를 배열로 반환한다.
        배열의 각 요소는 예측에 적용된 손실함수이며 true 값이다.
        scoreArray가 받아들이는 값.
        true 라벨 - labels
        인공 신경망의 최종 출력에 들어가는 입력 - preOutput,
        인공 신경망의 최종 출력에 사용되는 활성 함수 - activationFn
        마스크 - 라벨이 붙은 마스크 (있는 경우)
     */
    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr;
        // 인공 신경망의 출력이고 위에서 표기한 y_hat이다.
        // y_hat을 얻을 수 있는 방법 : preOutput은 활성화 함수에 의해 변환되어 신경망의 출력을 제공한다.
        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        // 스코어는 (y-y_hat)^2 + |y - y_hat| 의 합계이다.
        INDArray yMinusyHat = Transforms.abs(labels.sub(output));
        scoreArr = yMinusyHat.mul(yMinusyHat);
        scoreArr.addi(yMinusyHat);
        if (mask != null) {
            scoreArr.muliColumnVector(mask);
        }
        return scoreArr;
    }

    /*
    모든 손실함수에 대해 동일하게 유지된다.
    Compute Score는 많은 데이터 포인트에서 평균 손실 함수를 계산한다.
    단일 데이터 포인트의 손실은 모든 출력 피쳐에 대해 합산된다.
     */

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

        double score = scoreArr.sumNumber().doubleValue();

        if (average) {
            score /= scoreArr.size(0);
        }

        return score;
    }

    /*
     모든 손실 기능에 대해 동일하게 유지된다.
     Compute Score는 많은 데이터 포인트에 대한 손실 함수를 계산한다.
     단일 데이터 포인트의 손실은 모든 출력 피쳐에 대해 합계 된 손실이다.
     출력 피쳐의 샘플 x 크기 x 인 배열을 반환단다.
     */

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        return scoreArr.sum(1);
    }

    /*
        손실 함수에 따라 수정이 필요하다.
        기울기 wrt를 프리 아웃으로 계산한다. (신경망의 최종 계층에 대한 입력)
        연속적인 규칙 적용
        이 경우  L = (y - yhat)^2 + |y - yhat|, dL/dyhat = -2*(y-yhat) - sign(y-yhat), sign of y - yhat = +1 if y-yhat>= 0 else -1
        dyhat/dpreout = d(Activation(preout))/dpreout = Activation'(preout)
        dL/dpreout = dL/dyhat * dyhat/dpreout
    */

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        /*
        //NOTE: There are many ways to do this same set of operations in nd4j
        //The following is the most readable for the sake of this example, not necessarily the fastest
        //Refer to the Implementation of LossL1 and LossL2 for more efficient ways
        */
        /*
        // 참고 : nd4j에서 이와 동일한 작업 집합을 수행하는 여러 가지 방법이 있다.
        // 다음은이 예제에서 가장 읽기 쉽지만 반드시 가장 빠를 필요는 없다.
        //보다 효율적인 방법은 LossL1 및 LossL2의 구현을 참조하자.
        */
        INDArray yMinusyHat = labels.sub(output);
        INDArray dldyhat = yMinusyHat.mul(-2).sub(Transforms.sign(yMinusyHat)); //d(L)/d(yhat) -> 이것은 손실함수로 바뀔 것이다.

        //아래 모든 내용을 동일하게 유지하자.
        INDArray dLdPreOut = activationFn.backprop(preOutput.dup(), dldyhat).getFirst();
        // 항상 마스크값을 곱하자
        if (mask != null) {
            dLdPreOut.muliColumnVector(mask);
        }

        return dLdPreOut;
    }

    // 커스텀 손실 함수에 대해 동일하게 유지하자.
    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
    }


    @Override
    public String toString() {
        return "CustomLossL1L2()";
    }

}

