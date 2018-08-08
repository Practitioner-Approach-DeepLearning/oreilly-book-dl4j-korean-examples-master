package org.deeplearning4j.examples.recurrent.character;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

/**
 * 이 예제는 MultiLayerNetwork 아키텍처 대신 ComputationGraph 아키텍처를 사용한다는 점을 제외하면 GravesLSTMCharModellingExample과 거의 동일하다.
 * 자세한 내용은 해당 예제의 javadoc을 참조하자.
 * ComputationGraph 아키텍처에 대한 자세한 내용은 http://deeplearning4j.org/compgraph를 참조하자. 
 * ComputationGraph a를 사용하는 것 외에도이 버전에서는이 구성의 수행 방법을 보여주기 위해 첫 번째 레이어와 출력 레이어 사이를 건너 뛴다. 
 * 실제로 이는 연결 유형이 다음과 같은 것을 의미한다.
 * 
 * (a) 첫 번째 계층 -> 두 번째 계층 연결
 * (b) 첫 번째 계층 -> 출력 계층 연결
 * (c) 두 번째 계층 -> 출력 계층 연결
 *
 * @author Alex Black
 */
public class CompGraphLSTMExample {

    public static void main( String[] args ) throws Exception {
        int lstmLayerSize = 200;					//각 GravesLSTM 레이어의 단위 수
        int miniBatchSize = 32;						//학습 할 때 사용할 미니 배치의 크기
        int exampleLength = 1000;					//사용할 각 학습 예제 시퀀스의 길이. 이것은 시간이 지남에 따라 잘린 backpropagation을위한 길이를 늘릴 수 있다.
        int tbpttLength = 50;                       //즉, 매개 변수 업데이트를 50 자까지 수행한다.
        int numEpochs = 1;							//총 훈련 에포크 수
        int generateSamplesEveryNMinibatches = 10;  //신경망에서 샘플을 생성하는 빈도는? 1000 문자 / 50 tbptt 길이 : minibatch 당 20 매개 변수 업데이트
        int nSamplesToGenerate = 4;					//각 트레이닝 기간 이후에 생성 할 샘플 수
        int nCharactersToSample = 300;				//생성 할 각 샘플의 길이
        String generationInitialization = null;		//선택적 문자 초기화; null의 경우는 무작위의 문자가 사용된다
        // 위의 내용은 문자 시퀀스로 LSTM을 '초기화'하여 계속 / 완료하는 데 사용된다.
        // 초기화 문자는 모두 디폴트로 CharacterIterator.getMinimalCharacterSet()에 있어야 한다.
        Random rng = new Random(12345);

        //GravesLSTM 신경망을 학습하는데 사용할 수 있는 텍스트의 벡터화를 처리하는 DataSetIterator를 가져온다.
        CharacterIterator iter = GravesLSTMCharModellingExample.getShakespeareIterator(miniBatchSize, exampleLength);
        int nOut = iter.totalOutcomes();

        //신경망 구축
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .learningRate(0.1)
            .rmsDecay(0.95)
            .seed(12345)
            .regularization(true)
            .l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .graphBuilder()
            .addInputs("input") //입력에 이름을 넣자. 복수의 입력을 가지는 ComputationGraph의 경우, 이것도 입력 배열의 순서를 정의한다.
            //첫 번째 계층 : 이름이 "first"이고, 입력의 입력이 "input"
            .addLayer("first", new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
                .updater(Updater.RMSPROP).activation(Activation.TANH).build(),"input")
            //두 번째 계층: 이름은 "second"이며, 레이어의 입력은 "first"
            .addLayer("second", new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                .updater(Updater.RMSPROP)
                .activation(Activation.TANH).build(),"first")
            //출력 계층: "첫 번째"및 "두 번째"라는 두 레이어의 입력을 가진 "출력 레이어"
            .addLayer("outputLayer", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX).updater(Updater.RMSPROP)
                .nIn(2*lstmLayerSize).nOut(nOut).build(),"first","second")
            .setOutputs("outputLayer")  //출력을 나열하자. 복수의 출력을 가지는 ComputationGraph의 경우, 이것도 입력 배열의 순서를 정의한다.
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
            .pretrain(false).backprop(true)
            .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        //네트워크의 파라미터 수를 출력한다 (각 계층마다).
        int totalNumParams = 0;
        for( int i=0; i<net.getNumLayers(); i++ ){
            int nParams = net.getLayer(i).numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);

        //학습을 수행 한 다음 신경망에서 샘플을 생성하고 출력하자.
        int miniBatchNumber = 0;
        for( int i=0; i<numEpochs; i++ ){
            while(iter.hasNext()){
                DataSet ds = iter.next();
                net.fit(ds);
                if(++miniBatchNumber % generateSamplesEveryNMinibatches == 0){
                    System.out.println("--------------------");
                    System.out.println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters" );
                    System.out.println("Sampling characters from network given initialization \"" + (generationInitialization == null ? "" : generationInitialization) + "\"");
                    String[] samples = sampleCharactersFromNetwork(generationInitialization,net,iter,rng,nCharactersToSample,nSamplesToGenerate);
                    for( int j=0; j<samples.length; j++ ){
                        System.out.println("----- Sample " + j + " -----");
                        System.out.println(samples[j]);
                        System.out.println();
                    }
                }
            }

            iter.reset();	//다른 에포크를 위한 반복자 재설정
        }

        System.out.println("\n\nExample complete");
    }

    /** 초기화를 사용하여 신경망에서 샘플을 생성한다. 초기화
      * 초기화를 사용하여 확장 / 계속하려는 시퀀스로 RNN을 '초기화'할 수 있다. <br>
      * 초기화는 모든 샘플에 사용된다.
      * @param initialization String, null의 경우가 있다. null의 경우, 모든 샘플의 초기화로서 무작위의 문자를 선택한다
      * @param charactersToSample 네트워크에서 샘플링 할 문자 수 (초기화 제외)
      * @param net 하나 이상의 GravesLSTM / RNN 레이어와 softmax 출력 레이어가있는 MultiLayerNetwork
      * @param iter. CharacterIterator 인덱스에서 문자로 이동하는 데 사용된다.
      */
    private static String[] sampleCharactersFromNetwork( String initialization, ComputationGraph net,
                                                         CharacterIterator iter, Random rng, int charactersToSample, int numSamples ){
        //초기화를 설정하자. 초기화가없는 경우 임의 문자 사용
        if( initialization == null ){
            initialization = String.valueOf(iter.getRandomCharacter());
        }

        //초기화를 위한 입력 만들기
        INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
        char[] init = initialization.toCharArray();
        for( int i=0; i<init.length; i++ ){
            int idx = iter.convertCharacterToIndex(init[i]);
            for( int j=0; j<numSamples; j++ ){
                initializationInput.putScalar(new int[]{j,idx,i}, 1.0f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for( int i=0; i<numSamples; i++ ) sb[i] = new StringBuilder(initialization);

        //한 번에 한 문자 씩 네트워크 샘플 (및 피드 샘플을 입력으로 다시 입력) (모든 샘플)
        //샘플링은 여기에서 병렬로 수행된다.
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput)[0];
        output = output.tensorAlongDimension(output.size(2)-1,1,0);	//마지막 시간 단계 출력을 가져온다.

        for( int i=0; i<charactersToSample; i++ ){
            //이전 출력에서 샘플링하여 다음 입력 설정 (단일 시간 단계)
            INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns());
            //출력은 확률 분포이다. 생성하려는 각 예제에 대한 샘플을 새 입력에 추가하자.
            for( int s=0; s<numSamples; s++ ){
                double[] outputProbDistribution = new double[iter.totalOutcomes()];
                for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
                int sampledCharacterIdx = GravesLSTMCharModellingExample.sampleFromDistribution(outputProbDistribution,rng);

                nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//다음 시간 단계 입력 준비
                sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));	//샘플 된 문자를 StringBuilder에 추가 (사람이 읽을 수있는 출력).
            }

            output = net.rnnTimeStep(nextInput)[0];	//정방향 전달 한 단계 수행
        }

        String[] out = new String[numSamples];
        for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
        return out;
    }
}
