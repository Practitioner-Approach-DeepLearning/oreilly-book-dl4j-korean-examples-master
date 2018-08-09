package org.deeplearning4j.examples.recurrent.seq2seq;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by susaneraly on 3/27/16.
 * 최대 자릿수가 주어진 난수 쌍을 생성하는 클래스.
 * 이 클래스는 데이터 집합 반복자에 대한 참조로 사용되거나 자체 데이터 집합 반복자를 작성하는 데 사용될 수도 있다.
 */
public class CustomSequenceIterator implements MultiDataSetIterator {

    private Random randnumG;
    private int currentBatch;
    private int [] num1Arr;
    private int [] num2Arr;
    private int [] sumArr;
    private boolean toTestSet;
    private final int seed;
    private final int batchSize;
    private final int totalBatches;
    private final int numdigits;
    private final int encoderSeqLength;
    private final int decoderSeqLength;
    private final int outputSeqLength;
    private final int timestep;

    private static final int SEQ_VECTOR_DIM = 12;

    public CustomSequenceIterator (int seed, int batchSize, int totalBatches, int numdigits, int timestep) {

        this.seed = seed;
        this.randnumG = new Random(seed);

        this.batchSize = batchSize;
        this.totalBatches = totalBatches;

        this.numdigits = numdigits;
        this.timestep = timestep;

        this.encoderSeqLength = numdigits * 2 + 1;
        this.decoderSeqLength = numdigits + 1 + 1; // (numdigits + 1) 최대 합계일 수 있다.
        this.outputSeqLength = numdigits + 1 + 1; // (numdigits + 1) 최대합계일 수 있다.

        this.currentBatch = 0;
    }
    public MultiDataSet generateTest(int testSize) {
        toTestSet = true;
        MultiDataSet testData = next(testSize);
        return testData;
    }
    public ArrayList<int[]> testFeatures (){
        ArrayList<int[]> testNums = new ArrayList<int[]>();
        testNums.add(num1Arr);
        testNums.add(num2Arr);
        return testNums;
    }
    public int[] testLabels (){
        return sumArr;
    }
    @Override
    public MultiDataSet next(int sampleSize) {
        /* 주의사항:
            아래내용은 향상된 기능으로 나중에 수정될 수 있다.
         */
        
        //0으로 모든 것을 초기화하자. 결국 원 핫 벡터로 채울 것이다.
        INDArray encoderSeq = Nd4j.zeros(sampleSize, SEQ_VECTOR_DIM, encoderSeqLength );
        INDArray decoderSeq = Nd4j.zeros(sampleSize, SEQ_VECTOR_DIM, decoderSeqLength );
        INDArray outputSeq = Nd4j.zeros(sampleSize, SEQ_VECTOR_DIM, outputSeqLength );

        //이것들은 timestep의 고정 길이 시퀀스 들이기 때문에
        //마스크는 필요하지 않다.
        INDArray encoderMask = Nd4j.ones(sampleSize, encoderSeqLength);
        INDArray decoderMask = Nd4j.ones(sampleSize, decoderSeqLength);
        INDArray outputMask = Nd4j.ones(sampleSize, outputSeqLength);

        if (toTestSet) {
            num1Arr = new int [sampleSize];
            num2Arr = new int [sampleSize];
            sumArr = new int [sampleSize];
        }

        /* ========================================================================== */
        for (int iSample = 0; iSample < sampleSize; iSample++) {
            //두개의 난수 생성
            int num1 = randnumG.nextInt((int)Math.pow(10,numdigits));
            int num2 = randnumG.nextInt((int)Math.pow(10,numdigits));
            int sum = num1 + num2;
            if (toTestSet) {
                num1Arr[iSample] = num1;
                num2Arr[iSample] = num2;
                sumArr[iSample] = sum;
            }
            /*
            인코딩 순서:
            Eg. with numdigits=4, num1=123, num2=90
                123 + 90 는  "   09+321"으로부터 인코딩 된다
                2 * numdigits + 1에 의해 주어진 고정 크기의 문자열로 변환된다 (연산자의 경우).
                역으로 위의 결과에 마스크를 적용해서 입력을 구할 수 있다.
                입력을 반전 시키면 상당한 이득을 얻는다.
                각 문자는 12 차원의 원 핫 벡터로 변환된다.(해당 자릿수의 경우 색인 0-9, "+"의 경우 10, ""의 경우 11)
            */
            int spaceFill = (encoderSeqLength) - (num1 + "+" + num2).length();
            int iPos = 0;
            //필요한 경우 공백을 채운다.
            while (spaceFill > 0) {
                //spaces encoded at index 12
                encoderSeq.putScalar(new int[] {iSample,11,iPos},1);
                iPos++;
                spaceFill--;
            }

            //숫자 2를 역순으로 채운다.
            String num2Str = String.valueOf(num2);
            for(int i = num2Str.length()-1; i >= 0; i--){
                int onehot = Character.getNumericValue(num2Str.charAt(i));
                encoderSeq.putScalar(new int[] {iSample,onehot,iPos},1);
                iPos++;
            }
            //이 경우 "+"연산자를 채우고 인덱스 11에서 인코딩한다.
            encoderSeq.putScalar(new int [] {iSample,10,iPos},1);
            iPos++;
            //num1의 숫자를 역순으로 채운다.
            String num1Str = String.valueOf(num1);
            for(int i = num1Str.length()-1; i >= 0; i--){
                int onehot = Character.getNumericValue(num1Str.charAt(i));
                encoderSeq.putScalar(new int[] {iSample,onehot,iPos},1);
                iPos++;
            }
            //나머지 시계열에 대한 마스크 입력
            //while (iPos < timestep) {
            //    encoderMask.putScalar(new []{iSample,iPos},1);
            //    iPos++;
            // }
            /*
            디코더 및 출력 시퀀스:
            */
            //합계에서 자릿수 채우기
            iPos = 0;
            char [] sumCharArr = String.valueOf(num1+num2).toCharArray();
            for(char c : sumCharArr) {
                int digit = Character.getNumericValue(c);
                outputSeq.putScalar(new int [] {iSample,digit,iPos},1);
                //공백으로 채워진 디코더 입력
                decoderSeq.putScalar(new int [] {iSample,11,iPos},1);
                iPos++;
            }
            //공백을 채운다 가능하다면
            //마지막 인덱스를 위해 남겨두자
            while (iPos < numdigits + 1) {
                //12번쨰 인덱스에서 공백이 인코딩 된다
                outputSeq.putScalar(new int [] {iSample,11,iPos}, 1);
                //공백으로 채워진 디코더 입력
                decoderSeq.putScalar(new int [] {iSample,11,iPos},1);
                iPos++;
            }
            //최종적 예측값
            outputSeq.putScalar(new int [] {iSample,10,iPos}, 1);
            decoderSeq.putScalar(new int [] {iSample,11,iPos}, 1);
        }
        //Predict "."
        /* ========================================================================== */
        INDArray[] inputs = new INDArray[]{encoderSeq, decoderSeq};
        INDArray[] inputMasks = new INDArray[]{encoderMask, decoderMask};
        INDArray[] labels = new INDArray[]{outputSeq};
        INDArray[] labelMasks = new INDArray[]{outputMask};
        currentBatch++;
        return new org.nd4j.linalg.dataset.MultiDataSet(inputs, labels, inputMasks, labelMasks);
    }

    @Override
    public void reset() {
        currentBatch = 0;
        toTestSet = false;
        randnumG = new Random(seed);
    }

    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public boolean hasNext() {
        //This generates numbers on the fly
        return currentBatch < totalBatches;
    }

    @Override
    public MultiDataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not supported");
    }
    public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {

    }
}

