package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMin;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * --- Nd4j Example 5: Accumulation/Reduction Operations ---
 *
 * 이번 예제에서는, INDArray를 줄이는 방법에 대해 알아봅시다 - sum, max 와 같은 작업을 수행하면서 생기는 일에 대해
 * 알아봅시다.
 *
 * @author Alex Black
 */
public class Nd4jEx5_Accumulations {

    public static void main(String[] args) {

        /*
         * 축적하거나 줄이는 작업에는 두가지 방법이 있다. - 전체 행렬에 대한 작업 -> 스칼라 값을 반환 - 하나 혹은 그이상의 디멘젼에 대한
         * 작업 -> 하나의 배열 반환
         * 
         * 두가지 타입의 축적 관련 클레스가 있다. - 일반적인 축적: 실제 값을 반환 - min, max, sum, etc... - 인덱스 기반
         * 축적 : 인덱스값을 반환 - argmax
         * 
         */

        INDArray originalArray = Nd4j.linspace(1, 15, 15).reshape('c', 3, 5); // 예제 3을 참고하자, 1 X 15 -> 3 X 5 INDArray로
                                                                              // 변경
        System.out.println("Original array: \n" + originalArray);

        // 첫번째, 전체 행렬의 축소에 대해 새각해보자
        double minValue = originalArray.minNumber().doubleValue();
        double maxValue = originalArray.maxNumber().doubleValue();
        double sum = originalArray.sumNumber().doubleValue();
        double avg = originalArray.meanNumber().doubleValue();
        double stdev = originalArray.stdNumber().doubleValue();

        System.out.println("minValue:       " + minValue);
        System.out.println("maxValue:       " + maxValue);
        System.out.println("sum:            " + sum);
        System.out.println("average:        " + avg);
        System.out.println("standard dev.:  " + stdev);

        // 두번째, 0번쨰 디멘전을 변경하는 방법에 대해 알아보자
        // 이경우, 출력은 [1, 5] 행렬이다. 각각 출력값은 동일한 열의 최소/최대/평균이다.
        INDArray minAlong0 = originalArray.min(0);
        INDArray maxAlong0 = originalArray.max(0);
        INDArray sumAlong0 = originalArray.sum(0);
        INDArray avgAlong0 = originalArray.mean(0);
        INDArray stdevAlong0 = originalArray.std(0);

        System.out.println("\n\n\n");
        System.out.println("min along dimension 0:  " + minAlong0);
        System.out.println("max along dimension 0:  " + maxAlong0);
        System.out.println("sum along dimension 0:  " + sumAlong0);
        System.out.println("avg along dimension 0:  " + avgAlong0);
        System.out.println("stdev along dimension 0:  " + stdevAlong0);

        // 디멘젼 1을 따라 이러한 작업을 수행하면 대신 3 X 1 배열을 출력할 것이다.
        // 이 경우, 각각의 출력 값은 각각 열을 줄이는 값이다.
        // 다시 말하지만, 아래 내용이 인쇄 될 때 그것은 행 벡터처럼 보이지만 실제로는 열 벡터이다.
        INDArray avgAlong1 = originalArray.mean(1);
        System.out.println("\n\navg along dimension 1:  " + avgAlong1);
        System.out.println("Shape of avg along d1:  " + Arrays.toString(avgAlong1.shape()));

        // Index accumulations return an integer value.
        // 인덱스기반 누적은 정수 값을 반환한다.
        INDArray argMaxAlongDim0 = Nd4j.argMax(originalArray, 0); // 디멘젼 0에 대한 최대값에 대한 인덱스
        System.out.println("\n\nargmax along dimension 0:   " + argMaxAlongDim0);
        INDArray argMinAlongDim0 = Nd4j.getExecutioner().exec(new IMin(originalArray), 0); // 디멘젼 0에 대한 최소값에 대한 인덱스
        System.out.println("argmin along dimension 0:   " + argMinAlongDim0);
    }

}
