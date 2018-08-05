package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * --- Nd4j Example 6: Boolean Indexing ---
 *
 * 이 예제에서, boolean indexing을 사용하여 몇 가지 간단한 조건부 요소 별 작업을 수행하는 방법을 살펴 보자.
 *
 * @author Alex Black
 */
public class Nd4jEx6_BooleanIndexing {

    public static void main(String[] args) {

        int nRows = 3;
        int nCols = 5;
        long rngSeed = 12345;
        // -1에서 0까지의 임의의 숫자를 생성한다.
        INDArray random = Nd4j.rand(nRows, nCols, rngSeed).muli(2).subi(1);

        System.out.println("Array values:");
        System.out.println(random);

        // 예를들어, 조건부로 0.0보다 작은 값으로 0.0을 대체한다.
        INDArray randomCopy = random.dup();
        BooleanIndexing.replaceWhere(randomCopy, 0.0, Conditions.lessThan(0.0));
        System.out.println("After conditionally replacing negative values:\n" + randomCopy);

        // 혹은 조건부로 NaN 값을 대체한다.
        INDArray hasNaNs = Nd4j.create(new double[] { 1.0, 1.0, Double.NaN, 1.0 });
        BooleanIndexing.replaceWhere(hasNaNs, 0.0, Conditions.isNan());
        System.out.println("hasNaNs after replacing NaNs with 0.0:\n" + hasNaNs);

        // 혹은 조건부로 하나의 행렬에서 다른 행렬로 값을 복사한다.
        randomCopy = random.dup();
        INDArray tens = Nd4j.valueArrayOf(nRows, nCols, 10.0);
        BooleanIndexing.replaceWhere(randomCopy, tens, Conditions.lessThan(0.0));
        System.out.println(
                "Conditionally copying values from array 'tens', if original value is less than 0.0\n" + randomCopy);

        // 하나의 간단한 작업은 조건과 일치하는 값의 수를 세는 것이다.
        MatchCondition op = new MatchCondition(random, Conditions.greaterThan(0.0));
        int countGreaterThanZero = Nd4j.getExecutioner().exec(op, Integer.MAX_VALUE).getInt(0); // MAX_VALUE = "along
                                                                                                // all dimensions" or
                                                                                                // equivalently "for
                                                                                                // entire array"
        System.out.println("Number of values matching condition 'greater than 0': " + countGreaterThanZero);
    }

}
