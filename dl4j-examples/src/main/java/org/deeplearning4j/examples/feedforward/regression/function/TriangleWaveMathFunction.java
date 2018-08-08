package org.deeplearning4j.examples.feedforward.regression.function;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Sawtooth 파형의 절대 값인, Triangle 파형의 값을 계산하는 함수.
 *
 */
public class TriangleWaveMathFunction implements MathFunction {

    @Override
    public INDArray getFunctionValues(final INDArray x) {
        final double period = 6.0;
        final double[] xd = x.data().asDouble();
        final double[] yd = new double[xd.length];
        for (int i = 0; i < xd.length; i++) {
            yd[i] = Math.abs(2 * (xd[i] / period - Math.floor(xd[i] / period + 0.5)));
        }
        return Nd4j.create(yd, new int[]{xd.length, 1});  // 열 벡터
    }

    @Override
    public String getName() {
        return "TriangleWave";
    }
}
