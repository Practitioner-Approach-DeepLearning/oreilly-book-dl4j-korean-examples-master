package org.deeplearning4j.examples.feedforward.regression.function;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.factory.Nd4j;

/**
 * x의 사인 값을 x로 나누는 함수
 */
public class SinXDivXMathFunction implements MathFunction {

    @Override
    public INDArray getFunctionValues(final INDArray x) {
        return Nd4j.getExecutioner().execAndReturn(new Sin(x.dup())).div(x);
    }

    @Override
    public String getName() {
        return "SinXDivX";
    }
}
