package org.deeplearning4j.examples.feedforward.regression.function;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 이 예제에서는 입력된 배열(입자를 나타내는 배열)을 변환한다.
 * 무브먼트- 입력된 값이 일반적인 톱니 바퀴의 특정 간격을 나타내도록 한다.
 * 파형 함수를 사용해 출력을 계산해야 한다.
 *
 * @author Unknown
 * ERRM이 문서 추가
 */

public class SawtoothMathFunction implements MathFunction {

    @Override
    public INDArray getFunctionValues(final INDArray x) {
        final double sawtoothPeriod = 4.0;
        // 입력 데이터는 파형이 계산되는 간격
        final double[] xd2 = x.data().asDouble();
        final double[] yd2 = new double[xd2.length];
        for (int i = 0; i < xd2.length; i++) {  // sawtooth 파형 함수를 사용해 주어진 간격에서 값을 찾는다.
            yd2[i] = 2 * (xd2[i] / sawtoothPeriod - Math.floor(xd2[i] / sawtoothPeriod + 0.5));
        }
        return Nd4j.create(yd2, new int[]{xd2.length, 1});  // 열 벡터
    }

    @Override
    public String getName() {
        return "Sawtooth";
    }
}
