package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;

/**
 * --- Nd4j Example 3: Getting and setting parts of INDArrays ---
 *
 * 이번 예제는, INDArray의 부분을 얻고 다루는 방법에 대해 알아보자.
 *
 * @author Alex Black
 */
public class Nd4jEx3_GettingAndSettingSubsets {

    public static void main(String[] args) {

        // 수기로 값을 입력하여 3X5 INDArray를 작성해 보자.
        // 이것을 위해서, 1X15 행렬로 시작해보자, 그리고 'reshape'를 사용하여 3X5 INDArray를 만들자.
        INDArray originalArray = Nd4j.linspace(1, 15, 15).reshape('c', 3, 5);
        System.out.println("Original Array:");
        System.out.println(originalArray);

        // getRow, getColumn을 이용하여 행 혹은 열 각각을 얻을 수 있다.
        INDArray firstRow = originalArray.getRow(0);
        INDArray lastColumn = originalArray.getColumn(4);
        System.out.println();
        System.out.println("First row:\n" + firstRow);
        System.out.println("Last column:\n" + lastColumn);
        // 아래내용을 출력할 때 주의사항: lastColumn은 출력될 때는 행 벡터처럼 보이지만 실제로 열 벡터이다.
        System.out.println(
                "Shapes:         " + Arrays.toString(firstRow.shape()) + "\t" + Arrays.toString(lastColumn.shape()));

        // ND4J의 핵심 개념은 뷰(view)이다 : 하나의 INDArray는 다른 행렬과 같은 메모리 위치를 가리킬 수 있다.
        // 예를들어, getRow, getColumn은 originalArray의 뷰이다.
        // 따라서, 하나를 변경하면 다른 하나도 변경된다.
        firstRow.addi(1.0); // 해당 자리에 추가하는 기능: firstRow와 originalArray 양쪽 값을 모두 바꾼다.
        System.out.println("\n\n");
        System.out.println("firstRow, after addi operation:");
        System.out.println(firstRow);
        System.out.println(
                "originalArray, after firstRow.addi(1.0) operation: (note it is modified, as firstRow is a view of originalArray)");
        System.out.println(originalArray);

        // 다음 섹션을 위해 기본 행렬을 다시 만들어보자.
        originalArray = Nd4j.linspace(1, 15, 15).reshape('c', 3, 5);

        // INDArray의 인덱싱을 이용해 임의의 부분을 선택할 수 있다.
        // 모든 행들중에 처음 세개의 행
        INDArray first3Columns = originalArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 3));
        System.out.println("first 3 columns:\n" + first3Columns);
        // 다시한번, 이것 또한 뷰이다.
        first3Columns.addi(100);
        System.out.println("originalArray, after first3Columns.addi(100)");
        System.out.println(originalArray);

        // 다음 섹션을 위해 기본 행렬을 다시 만들어보자.
        originalArray = Nd4j.linspace(1, 15, 15).reshape('c', 3, 5);

        // 임의의 부분을 설정할 수 있다.
        // 2번째 인덱스의 세번째 컬럼에 0을 설정하자.
        INDArray zerosColumn = Nd4j.zeros(3, 1);
        originalArray.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.point(2) }, zerosColumn); // 모든행, 두번째
                                                                                                           // 인덱스의 열들
        System.out.println("\n\n\nOriginal array, after put operation:\n" + originalArray);

        // 다음 섹션을 위해 기본 행렬을 다시 만들어보자.
        originalArray = Nd4j.linspace(1, 15, 15).reshape('c', 3, 5);

        // 가끔, 우리는 이러한 in-place 행위를 원치않는다. 이러한 경우, 끝에 .dup() 메서드를 추가하면 된다.
        // .dup() 메서드 - 'duplicate'를 의미함 - 새롭고 분리된 행렬을 만든다.
        INDArray firstRowDup = originalArray.getRow(0).dup(); // 첫번째 행을 복사한다. firstRowDup은 originalArray의 뷰가 아니다.
        firstRowDup.addi(100);
        System.out.println("\n\n\n");
        System.out.println("firstRowDup, after .addi(100):\n" + firstRowDup);
        System.out.println("originalArray, after firstRowDup.addi(100): (note it is unmodified)\n" + originalArray);
    }
}
