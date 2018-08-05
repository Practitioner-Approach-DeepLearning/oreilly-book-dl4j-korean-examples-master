package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

/**
 * --- Nd4j Example 4: Additional Operations with INDArrays ---
 *
 * 이번 예제에서는, INDArray를 다루는 방법을 알아보자
 *
 * @author Alex Black
 */
public class Nd4jEx4_Ops {

    public static void main(String[] args) {

        /*
         * ND4J는 다양한 작업을 정의한다. 여기서 그것들을 어떻게 사용하는지 알아보자. - 요소를 다루는 작업 : add, multiply,
         * divide, subtract, etc add, mul, div, sub, INDArray.add(INDArray),
         * INDArray.mul(INDArray), etc - 배열 곱셈: mmul - 행/열 벡터 ops: addRowVector,
         * mulColumnVector, etc - Element-wise transforms, like tanh, scalar max
         * operations, etc
         */

        // 첫번째로, in-place와 copy의 다른 점을 알아보자
        // 다음 호출로직을 비교해 보자: myArray.add(1.0) vs myArray.addi(1.0)
        // 여기서 addi의 i의 의미는 in-place라는 의미이다.
        // in-place 작업은 원래의 배열도 변경시키지만 copy작업은 원래의 배열을 변경시키지 않는다.
        INDArray originalArray = Nd4j.linspace(1, 15, 15).reshape('c', 3, 5); // (1, 15) -> (3,5)로 변경
        INDArray copyAdd = originalArray.add(1.0);
        System.out.println("Same object returned by add:    " + (originalArray == copyAdd));
        System.out.println("Original array after originalArray.add(1.0):\n" + originalArray);
        System.out.println("copyAdd array:\n" + copyAdd);

        // in-place add 작업과 비슷한 것을 해보자.
        INDArray inPlaceAdd = originalArray.addi(1.0);
        System.out.println();
        System.out.println("Same object returned by addi:    " + (originalArray == inPlaceAdd)); // addi는 정확한 자바객체를
                                                                                                 // 리턴한다.
        System.out.println("Original array after originalArray.addi(1.0):\n" + originalArray);
        System.out.println("inPlaceAdd array:\n" + copyAdd);

        // 다음 섹션을 위해 기본 행렬을 다시 만들어보자.
        originalArray = Nd4j.linspace(1, 15, 15).reshape('c', 3, 5);
        INDArray random = Nd4j.rand(3, 5); // 예제 2에서 본 것과 같이 3X5행렬을 임의의 값으로 초기화한다.

        // 요소를 다루는 작업을 하기 때문에 배열 모양은 일치해야한다.
        // add와 addi는 스칼라에서는 정확하게 같은 방법으로 작동한다.
        INDArray added = originalArray.add(random);
        System.out.println("\n\n\nRandom values:\n" + random);
        System.out.println("Original plus random values:\n" + added);

        // 행렬의 곱샘은 쉽게 구현할 수 있다.
        INDArray first = Nd4j.rand(3, 4);
        INDArray second = Nd4j.rand(4, 5);
        INDArray mmul = first.mmul(second);
        System.out.println("\n\n\nShape of mmul array:      " + Arrays.toString(mmul.shape())); // 3x5 output as
                                                                                                // expected

        // We can do row-wise ("for each row") and column-wise ("for each column")
        // operations
        // Again, inplace vs. copy ops work the same way (i.e., addRowVector vs.
        // addiRowVector)
        // 행에 대한 작업 (각각의 행에 대한) 그리고 열에 대한 작업 ( 각각의 열에 대한)을 수행한다.
        // in-place와 copy 작업은 같은 방법으로 일어난다 (addRowVector와 addiRowVector를 비교해보자)
        INDArray row = Nd4j.linspace(0, 4, 5);
        System.out.println("\n\n\nRow:\n" + row);
        INDArray mulRowVector = originalArray.mulRowVector(row); // 'originalArray'의 각각의 행에 요소마다 행 벡터를 곱한다.
        System.out.println("Result of originalArray.mulRowVector(row)");
        System.out.println(mulRowVector);

        // 요소를 변경시키는 'tanh' 혹은 max value와 같은 것들은 아래처럼 적용시킬 수 있다.
        System.out.println("\n\n\n");
        System.out.println("Random array:\n" + random); // 예제 2와 같이 생략된 정보가 출력되는 것을 확인하자.
        System.out.println("Element-wise tanh on random array:\n" + Transforms.tanh(random));
        System.out.println("Element-wise power (x^3.0) on random array:\n" + Transforms.pow(random, 3.0));
        System.out.println("Element-wise scalar max (with scalar 0.5):\n" + Transforms.max(random, 0.5));
        // 좀 더 자세히 이것을 수행하는 방법에 대해 알아보자.
        INDArray sinx = Nd4j.getExecutioner().execAndReturn(new Sin(random.dup()));
        System.out.println("Element-wise sin(x) operation:\n" + sinx);
    }
}
