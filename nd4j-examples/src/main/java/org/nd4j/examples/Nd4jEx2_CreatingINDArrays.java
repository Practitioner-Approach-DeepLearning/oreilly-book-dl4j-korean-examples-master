package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * --- Nd4j Example 2: Creating INDArrays ---
 *
 * 이번 예제에서는, INDArrays를 만드는 여러가지 다른 방법을 보게 될 것이다.
 *
 * @author Alex Black
 */
public class Nd4jEx2_CreatingINDArrays {

    public static void main(String[] args) {

        // 서로 다른 스칼라 값으로 초기화 하면서 INDArray를 생성하는 방법을 보게 될 것이다.
        int nRows = 3;
        int nColumns = 5;
        INDArray allZeros = Nd4j.zeros(nRows, nColumns);
        System.out.println("Nd4j.zeros(nRows, nColumns)");
        System.out.println(allZeros);

        INDArray allOnes = Nd4j.ones(nRows, nColumns);
        System.out.println("\nNd4j.ones(nRows, nColumns)");
        System.out.println(allOnes);

        INDArray allTens = Nd4j.valueArrayOf(nRows, nColumns, 10.0);
        System.out.println("\nNd4j.valueArrayOf(nRows, nColumns, 10.0)");
        System.out.println(allTens);

        // double[], double[][]( 혹은 float/int로 만들어진 자바 행렬)로부터 INDArray를 생성할 수 있다.
        double[] vectorDouble = new double[] { 1, 2, 3 };
        INDArray rowVector = Nd4j.create(vectorDouble);
        System.out.println("rowVector:              " + rowVector);
        System.out.println("rowVector.shape():      " + Arrays.toString(rowVector.shape())); // (1, 3)

        INDArray columnVector = Nd4j.create(vectorDouble, new int[] { 3, 1 }); // (3, 1)을 직접 명시한다.
        System.out.println("columnVector:           " + columnVector); // print시 참고사항: 행/열 벡터가 모두 한 라인으로 출력된다.
        System.out.println("columnVector.shape():   " + Arrays.toString(columnVector.shape())); // (3, 1)

        double[][] matrixDouble = new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
        INDArray matrix = Nd4j.create(matrixDouble);
        System.out.println("\nINDArray defined from double[][]:");
        System.out.println(matrix);

        // 임의로 INDArray를 생성할 수 있다.
        // 그러나 기본적으로, 임의의 값은 INDArray.toString()을 사용하면 생략된 값으로 출력된다.
        int[] shape = new int[] { nRows, nColumns };
        INDArray uniformRandom = Nd4j.rand(shape);
        System.out.println("\n\n\nUniform random array:");
        System.out.println(uniformRandom);
        System.out.println("Full precision of random value at position (0,0): " + uniformRandom.getDouble(0, 0));

        INDArray gaussianMeanZeroUnitVariance = Nd4j.randn(shape);
        System.out.println("\nN(0,1) random array:");
        System.out.println(gaussianMeanZeroUnitVariance);

        // RNG시드를 이용하여 반복 할 수 있다.
        long rngSeed = 12345;
        INDArray uniformRandom2 = Nd4j.rand(shape, rngSeed);
        INDArray uniformRandom3 = Nd4j.rand(shape, rngSeed);
        System.out.println("\nUniform random arrays with same fixed seed:");
        System.out.println(uniformRandom2);
        System.out.println();
        System.out.println(uniformRandom3);

        // 당연히, 2차원 이상을 쉽게 표현할 수 있다.
        INDArray threeDimArray = Nd4j.ones(3, 4, 5); // 3x4x5 INDArray
        INDArray fourDimArray = Nd4j.ones(3, 4, 5, 6); // 3x4x5x6 INDArray
        INDArray fiveDimArray = Nd4j.ones(3, 4, 5, 6, 7); // 3x4x5x6x7 INDArray
        System.out.println("\n\n\nCreating INDArrays with more dimensions:");
        System.out.println("3d array shape:         " + Arrays.toString(threeDimArray.shape()));
        System.out.println("4d array shape:         " + Arrays.toString(fourDimArray.shape()));
        System.out.println("5d array shape:         " + Arrays.toString(fiveDimArray.shape()));

        // INDArray끼리 조합하여 새로운 INDArray를 만들 수 있다.
        INDArray rowVector1 = Nd4j.create(new double[] { 1, 2, 3 });
        INDArray rowVector2 = Nd4j.create(new double[] { 4, 5, 6 });

        INDArray vStack = Nd4j.vstack(rowVector1, rowVector2); // 수직 스택: [1,3]+[1,3] 은 [2,3]
        INDArray hStack = Nd4j.hstack(rowVector1, rowVector2); // 수평 스택: [1,3]+[1,3] 은 [1,6]
        System.out.println("\n\n\nCreating INDArrays from other INDArrays, using hstack and vstack:");
        System.out.println("vStack:\n" + vStack);
        System.out.println("hStack:\n" + hStack);

        // 이외에 여러가지 메서드들이 있다.
        INDArray identityMatrix = Nd4j.eye(3);
        System.out.println("\n\n\nNd4j.eye(3):\n" + identityMatrix);
        INDArray linspace = Nd4j.linspace(1, 10, 10); // 값은 1 에서 10까지, 10개의 스탭이다.
        System.out.println("Nd4j.linspace(1,10,10):\n" + linspace);
        INDArray diagMatrix = Nd4j.diag(rowVector2); // rowVector2 대각선을 따라 정사각형의 행렬을 만든다.
        System.out.println("Nd4j.diag(rowVector2):\n" + diagMatrix);

    }

}
