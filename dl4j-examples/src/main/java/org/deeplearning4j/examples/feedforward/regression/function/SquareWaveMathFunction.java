package org.deeplearning4j.examples.feedforward.regression.function;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sign;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 부호(x) 또는 실수 x에 대한 부호는 x가 음수이면 01, x가 0이면 0, x가 양수이면 1이다.
 *
 * 사인 x의 부호 함수 값은 -1, 0, 1일 수 있다.
 * 부호(사인)의 세 가지 출력은 그래프에서 정사각형과 유사한 선을 형성한다.
 */
public class SquareWaveMathFunction implements MathFunction {

    @Override
    public INDArray getFunctionValues(final INDArray x) {
        final INDArray sin = Nd4j.getExecutioner().execAndReturn(new Sin(x.dup()));
        return Nd4j.getExecutioner().execAndReturn(new Sign(sin));
    }

    @Override
    public String getName() {
        return "SquareWave";
    }
}
