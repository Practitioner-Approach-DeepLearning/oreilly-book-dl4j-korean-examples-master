package org.deeplearning4j.examples.misc.lossfunctions;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * Created by susaneraly on 11/9/16.
 * 이것은 커스텀 손실 함수를 만드는 방법에 대한 예제이다.
 * 이 예제는 커스텀 손실 함수를 제외하고 org.deeplearning4j.examples.feedforward.regression.RegressionSum
 * 과 동일하다.
 */


public class CustomLossExample {
    public static final int seed = 12345;
    public static final int iterations = 1;
    public static final int nEpochs = 500;
    public static final int nSamples = 1000;
    public static final int batchSize = 100;
    public static final double learningRate = 0.001;
    public static int MIN_RANGE = 0;
    public static int MAX_RANGE = 3;

    public static final Random rng = new Random(seed);

    public static void main(String[] args) {
        doTraining();

        // 이것은 기울기 체커를 간단히 나타낸 것이다.
        // 정확성을 보장하기 위해 한정된 차이 근사치와 비교하여 구현을 확인한다.
        // 반드시 직접 작성한 기울기 체커를 사용해야 한다.
        // 아래 코드를 사용하거나 다음을 참고하자
        // deeplearning4j/deeplearning4j-core/src/test/java/org/deeplearning4j/gradientcheck/LossFunctionGradientCheck.java
        doGradientCheck();
    }

    public static void doTraining(){

        DataSetIterator iterator = getTrainingData(batchSize,rng);

        // 신경망 생성
        int numInput = 2;
        int numOutputs = 1;
        int nHidden = 10;
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.95)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
                .activation(Activation.TANH)
                .build())
                // 즉시 커스텀 손실 함수를 아래와 같이 다시 설정
                // 구현에 대한 자세한 내용은 CustomLossL1L2 클래스를 참조하자.
            .layer(1, new OutputLayer.Builder(new CustomLossL1L2())
                .activation(Activation.IDENTITY)
                .nIn(nHidden).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build()
        );
        net.init();
        net.setListeners(new ScoreIterationListener(100));

        // 전체 데이터 집합에 대해 신경망을 교육하고 주기적으로 평가한다.
        for( int i=0; i<nEpochs; i++ ){
            iterator.reset();
            net.fit(iterator);
        }
        // 추가적으로 두개의 숫자를 더 테스트 해보자 (각각 다른 숫자로 해볼 것)
        final INDArray input = Nd4j.create(new double[] { 0.111111, 0.3333333333333 }, new int[] { 1, 2 });
        INDArray out = net.output(input, false);
        System.out.println(out);

    }

    private static DataSetIterator getTrainingData(int batchSize, Random rand){
        double [] sum = new double[nSamples];
        double [] input1 = new double[nSamples];
        double [] input2 = new double[nSamples];
        for (int i= 0; i< nSamples; i++) {
            input1[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble();
            input2[i] =  MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble();
            sum[i] = input1[i] + input2[i];
        }
        INDArray inputNDArray1 = Nd4j.create(input1, new int[]{nSamples,1});
        INDArray inputNDArray2 = Nd4j.create(input2, new int[]{nSamples,1});
        INDArray inputNDArray = Nd4j.hstack(inputNDArray1,inputNDArray2);
        INDArray outPut = Nd4j.create(sum, new int[]{nSamples, 1});
        DataSet dataSet = new DataSet(inputNDArray, outPut);
        List<DataSet> listDs = dataSet.asList();
        Collections.shuffle(listDs,rng);
        return new ListDataSetIterator(listDs,batchSize);
    }




    public static void doGradientCheck() {
        double epsilon = 1e-3;
        int totalNFailures = 0;
        double maxRelError = 5.0; // in %
        CustomLossL1L2 lossfn = new CustomLossL1L2();
        String[] activationFns = new String[]{"identity", "softmax", "relu", "tanh", "sigmoid"};
        int[] labelSizes = new int[]{1, 2, 3, 4};
        for (int i = 0; i < activationFns.length; i++) {
            System.out.println("Running checks for "+activationFns[i]);
            IActivation activation = Activation.fromString(activationFns[i]).getActivationFunction();
            List<INDArray> labelList = makeLabels(activation,labelSizes);
            List<INDArray> preOutputList = makeLabels(new ActivationIdentity(),labelSizes);
            for (int j=0; j<labelSizes.length; j++) {
                System.out.println("\tRunning check for length " + labelSizes[j]);
                INDArray label = labelList.get(j);
                INDArray preOut = preOutputList.get(j);
                INDArray grad = lossfn.computeGradient(label,preOut,activation,null);
                NdIndexIterator iterPreOut = new NdIndexIterator(preOut.shape());
                while (iterPreOut.hasNext()) {
                    int[] next = iterPreOut.next();
                    // 라벨의 각 출력 특징에 대한 총 스코어가 있는 기울기 검사
                    double before = preOut.getDouble(next);
                    preOut.putScalar(next, before + epsilon);
                    double scorePlus = lossfn.computeScore(label, preOut, activation, null, true);
                    preOut.putScalar(next, before - epsilon);
                    double scoreMinus = lossfn.computeScore(label, preOut, activation, null, true);
                    preOut.putScalar(next, before);

                    double scoreDelta = scorePlus - scoreMinus;
                    double numericalGradient = scoreDelta / (2 * epsilon);
                    double analyticGradient = grad.getDouble(next);
                    double relError = Math.abs(analyticGradient - numericalGradient) * 100 / (Math.abs(numericalGradient));
                    if( analyticGradient == 0.0 && numericalGradient == 0.0 ) relError = 0.0;
                    if (relError > maxRelError || Double.isNaN(relError)) {
                        System.out.println("\t\tParam " + Arrays.toString(next) + " FAILED: grad= " + analyticGradient + ", numericalGrad= " + numericalGradient
                            + ", relErrorPerc= " + relError + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus);
                        totalNFailures++;
                    } else {
                        System.out.println("\t\tParam " + Arrays.toString(next) + " passed: grad= " + analyticGradient + ", numericalGrad= " + numericalGradient
                            + ", relError= " + relError + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus);
                    }
                }
            }
        }
        if(totalNFailures > 0) System.out.println("DONE:\n\tGradient check failed for loss function; total num failures = " + totalNFailures);
        else System.out.println("DONE:\n\tSimple gradient check passed - This is NOT exhaustive by any means");
    }

    /* This function is a utility function for the gradient check above
        It generate labels randomly in the right range depending on the activation function
        Uses a gaussian
        identity: range is anything
        relu: range is non-negative
        softmax: range is non-negative and adds up to 1
        sigmoid: range is between 0 and 1
        tanh: range is between -1 and 1

     */
    /* 이 함수는 위의 기울기 체크를 위한 유틸형 함수이다
        활성화 함수에 따른 올바른 범위에서 임의로 라벨을 생성한다.
        가우시안 활용: 범위는 상관하지않는다.
        relu: 범위에 음수는 존재하지 않는다.
        softmax: 범위는 음수가 아니면서 최대 1을 더한다.
        sigmoid: 범위가 0~1사이다.
        tanh: 범위가 -1~1 사이다.

     */
    public static List<INDArray> makeLabels(IActivation activation,int[]labelSize) {
        // 두 개의 + ve 및 -ve 값, 0과 0이 아닌 값, 0보다 작거나 0보다 큰 softmax를 제외한 모든 것에 대한 라벨 크기이다.
        List<INDArray> returnVals = new ArrayList<>(labelSize.length);
        for (int i=0; i< labelSize.length; i++) {
            int aLabelSize = labelSize[i];
            Random r = new Random();
            double[] someVals = new double[aLabelSize];
            double someValsSum = 0;
            for (int j=0; j<aLabelSize; j++) {
                double someVal = r.nextGaussian();
                double transformVal = 0;
                switch (activation.toString()) {
                    case "identity":
                        transformVal = someVal;
                    case "softmax":
                        transformVal = someVal;
                        break;
                    case "sigmoid":
                        transformVal = Math.sin(someVal);
                        break;
                    case "tanh":
                        transformVal = Math.tan(someVal);
                        break;
                    case "relu":
                        transformVal = someVal * someVal + 4;
                        break;
                    default:
                        throw new RuntimeException("Unknown activation function");
                }
                someVals[j] = transformVal;
                someValsSum += someVals[j];
            }
            if ("sigmoid".equals(activation.toString())) {
                for (int j=0; j<aLabelSize; j++) {
                    someVals[j] /= someValsSum;
                }
            }
            returnVals.add(Nd4j.create(someVals));
        }
        return returnVals;
    }
}
