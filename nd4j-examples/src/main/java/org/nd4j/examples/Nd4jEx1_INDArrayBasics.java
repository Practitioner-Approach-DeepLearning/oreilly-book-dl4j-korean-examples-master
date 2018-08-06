package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * --- ND4J Example 1: INDArray Basics ---
 *
 * 이 예제를 통해 INDArrays의 기본적인 동작에 대해 알아보자.
 *
 * @author Alex Black
 */
public class Nd4jEx1_INDArrayBasics {

    public static void main(String[] args) {

        /*
         * 시작하기전에, INDArray에 대해 다시 확인해보자 : INDArray는 숫자형에 대한 다차원 배열이다 : 벡터, 행렬 혹은 텐서에 대한
         * 내용을 나타낼 때 사용한다. 내부적으로, 소수점 한자리 혹은 두자리의 float형 값을 각각 저장할 수 있다.
         * 
         * 여기서, INDArray의 정보를 참조하는 방법에 대해 확인할 수 있다. 이후 INDArray 생성방법과 INDArray를 활용한 좀 더
         * 많은 연산들에 대해 확인할 수 있다.
         */

        // 기본적인 2차원 배열을 생성하면서 시작해보자. 3X5행렬을 생성해보자. 행렬의 모든 값은 0.0으로 채우자
        int nRows = 3;
        int nColumns = 5;
        INDArray myArray = Nd4j.zeros(nRows, nColumns);

        // 다음으로 배열에 대한 기본적인 정보를 출력하자.
        System.out.println("Basic INDArray information:");
        System.out.println("Num. Rows:          " + myArray.rows());
        System.out.println("Num. Columns:       " + myArray.columns());
        System.out.println("Num. Dimensions:    " + myArray.rank()); // 2 dimensions -> rank 2
        System.out.println("Shape:              " + Arrays.toString(myArray.shape())); // [3,5] -> 3행, 5열의 값을 출력
        System.out.println("Length:             " + myArray.length()); // 3행 * 5열 = 전체 엔트리는 15개이다.

        // toString 메소드를 이용해서 배열 자체를 출력할 수 있다.
        System.out.println("\nArray Contents:\n" + myArray);

        // 비슷한 정보를 얻을 수 있는 다른 방법이 있다.
        System.out.println();
        System.out.println("size(0) == nRows:   " + myArray.size(0)); // .shape()[0]으로 출력해도 같은 값이 나온다.
        System.out.println("size(1) == nCols:   " + myArray.size(1)); // .shape()[1]로 출력해도 같은 값이 나온다.
        System.out.println("Is a vector:        " + myArray.isVector());
        System.out.println("Is a scalar:        " + myArray.isScalar());
        System.out.println("Is a matrix:        " + myArray.isMatrix());
        System.out.println("Is a square matrix: " + myArray.isSquare());
        ;

        // 배열을 약간 수정해보자
        // 인덱싱을 0으로 시작한다. 행은 0~2, 열은 0~4까지 사용한다.
        myArray.putScalar(0, 1, 2.0); // (0, 1)에는 2.0을 입력
        myArray.putScalar(2, 3, 5.0); // (2, 3)에는 5.0을 입력
        System.out.println("\nArray after putScalar operations:");
        System.out.println(myArray);

        // 또한 각각의 값을 얻을 수 있다.
        double val0 = myArray.getDouble(0, 1); // (0, 1)의 값을 얻는다 - 이미 입력해 놓은 2.0 값을 얻는다.
        System.out.println("\nValue at (0,1):     " + val0);

        // 마지막으로, 배열에 할 수 있는 많은 것들이 있다. 예를들어 스칼라 값을 추가하는 것이다.
        INDArray myArray2 = myArray.add(1.0); // 각각의 엔트리에 1.0을 입력한다.
        System.out.println("\nNew INDArray, after adding 1.0 to each entry:");
        System.out.println(myArray2);

        INDArray myArray3 = myArray2.mul(2.0); // 각각의 엔트리에 2.0을 곱한다.
        System.out.println("\nNew INDArray, after multiplying each entry by 2.0:");
        System.out.println(myArray3);
    }

}
