package org.deeplearning4j.examples.recurrent.seq2seq;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import java.util.ArrayList;


/**
 * Created by susaneraly on 3/27/16.
 */
public class AdditionRNN {

    /*
        이 예제는 http://arxiv.org/abs/1410.4615에 설명 된 시퀀스 RNN에 대한 시퀀스에서 모델링된다. 
        특히 시퀀스 NN에 대한 시퀀스는 더하기 연산을 위해 빌드된다. 
        두 개의 숫자와 더하기 연산자는 시퀀스로 인코딩되어 통과한다. 
        "인코더"RNN 인코더 RNN의 마지막 타임 스텝의 출력은 시계열로 재 해석되어 "디코더"RNN을 통과한다. 
        결과는 시퀀스에서 인코딩 된 합계 인 디코더 RNN의 출력이다 . 
        원 핫 벡터는 인코딩/디코딩에 사용된다.
        20 에포크로 2 자리 숫자에 85 % 이상의 정확도 제공
        NUM_DIGITS을 변경하는 간단한 작업으로 다른 숫자로 바꿔서 테스트 해볼 수 있다.
     */


    // 난수 생성기 시드
    public static final int seed = 1234;

    public static final int NUM_DIGITS =2;
    public static final int FEATURE_VEC_SIZE = 12;

    //데이터셋 사이즈 = batchSize * totalBatches
    public static final int batchSize = 10;
    public static final int totalBatches = 500;
    public static final int nEpochs = 50;
    public static final int nIterations = 1;
    public static final int numHiddenNodes = 128;

    //현재 시퀀스 길이 = max Length
    //시간별 스탭에 따라 값이 계속 바뀐다
    public static final int timeSteps = NUM_DIGITS * 2 + 1;

    public static void main(String[] args) throws Exception {

        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        //학습 데이터 반복자
        CustomSequenceIterator iterator = new CustomSequenceIterator(seed, batchSize, totalBatches, NUM_DIGITS,timeSteps);

        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
                //.regularization(true).l2(0.000005)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.5)
                .updater(Updater.RMSPROP)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(nIterations)
                .seed(seed)
                .graphBuilder()
                .addInputs("additionIn", "sumOut")
                .setInputTypes(InputType.recurrent(FEATURE_VEC_SIZE), InputType.recurrent(FEATURE_VEC_SIZE))
                .addLayer("encoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE).nOut(numHiddenNodes).activation(Activation.SOFTSIGN).build(),"additionIn")
                .addVertex("lastTimeStep", new LastTimeStepVertex("additionIn"), "encoder")
                .addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("sumOut"), "lastTimeStep")
                .addLayer("decoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE+numHiddenNodes).nOut(numHiddenNodes).activation(Activation.SOFTSIGN).build(), "sumOut","duplicateTimeStep")
                .addLayer("output", new RnnOutputLayer.Builder().nIn(numHiddenNodes).nOut(FEATURE_VEC_SIZE).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build(), "decoder")
                .setOutputs("output")
                .pretrain(false).backprop(true)
                .build();

        ComputationGraph net = new ComputationGraph(configuration);
        net.init();
        //net.setListeners(new ScoreIterationListener(200),new HistogramIterationListener(200));
        net.setListeners(new ScoreIterationListener(1));
        //net.setListeners(new HistogramIterationListener(200));
        //학습용 모델
        int iEpoch = 0;
        int testSize = 200;
        while (iEpoch < nEpochs) {
            System.out.printf("* = * = * = * = * = * = * = * = * = ** EPOCH %d ** = * = * = * = * = * = * = * = * = * = * = * = * = * =\n",iEpoch);
            net.fit(iterator);

            MultiDataSet testData = iterator.generateTest(testSize);
            ArrayList<int[]> testNums = iterator.testFeatures();
            int[] testnum1 = testNums.get(0);
            int[] testnum2 = testNums.get(1);
            int[] testSums = iterator.testLabels();
            INDArray[] prediction_array = net.output(testData.getFeatures(0),testData.getFeatures(1));
            INDArray predictions = prediction_array[0];
            INDArray answers = Nd4j.argMax(predictions,1);

            encode_decode(testnum1,testnum2,testSums,answers);

            iterator.reset();
            iEpoch++;
        }
        System.out.println("\n* = * = * = * = * = * = * = * = * = ** EPOCH " + iEpoch + " COMPLETE ** = * = * = * = * = * = * = * = * = * = * = * = * = * =");

    }

    //이것은 신경망으로부터 예측값을 더 잘 읽게 만드는 도우미 함수이다.
    private static void encode_decode(int[] num1, int[] num2, int[] sum, INDArray answers) {

        int nTests = answers.size(0);
        int wrong = 0;
        int correct = 0;
        for (int iTest=0; iTest < nTests; iTest++) {
            int aDigit = NUM_DIGITS;
            int thisAnswer = 0;
			String strAnswer = "";
            while (aDigit >= 0) {
                //System.out.println("while"+aDigit+strAnwer);
                int thisDigit = (int) answers.getDouble(iTest,aDigit);
                //System.out.println(thisDigit);
                if (thisDigit <= 9) {
                    strAnswer+= String.valueOf(thisDigit);
                	thisAnswer += thisDigit * (int) Math.pow(10,aDigit);
                }
                else {
                    //System.out.println(thisDigit+" is string " + String.valueOf(thisDigit));
					strAnswer += " ";
                    //break;
                }
                aDigit--;
            }
			String strAnswerR = new StringBuilder(strAnswer).reverse().toString();
		    strAnswerR = strAnswerR.replaceAll("\\s+","");
            if (strAnswerR.equals(String.valueOf(sum[iTest]))) {
                System.out.println(num1[iTest]+"+"+num2[iTest]+"=="+strAnswerR);
                correct ++;
            }
            else {
                System.out.println(num1[iTest]+"+"+num2[iTest]+"!="+strAnswerR+", should=="+sum[iTest]);
                wrong ++;
            }
        }
        double randomAcc = Math.pow(10,-1*(NUM_DIGITS+1)) * 100;
        System.out.println("*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*==*=*=*=*=*");
        System.out.println("WRONG: "+wrong);
        System.out.println("CORRECT: "+correct);
        System.out.println("Note randomly guessing digits in succession gives lower than a accuracy of:"+randomAcc+"%");
        System.out.println("The digits along with the spaces have to be predicted\n");
    }

}

