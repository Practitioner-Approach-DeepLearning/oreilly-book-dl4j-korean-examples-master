package org.deeplearning4j.examples.recurrent.character;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Random;

/**GravesLSTM 문자 모델링 예제
 * @author Alex Black

   예제: 한 번에 한 문자 씩 텍스트를 생성하려면 LSTM RNN으로 학습해야한다.
		이 예제는 Andrej Karpathy의 블로그 게시물을 참고했다.
		"The Unreasonable Effectiveness of Recurrent Neural Networks"
		http://karpathy.github.io/2015/05/21/rnn-effectiveness/

	이 예제는 Project Gutenberg에서 다운로드 한 Complete Works of William Shakespeare를 사용해서 교육하는 내용이다.
	해당 예제를 이용해서 다른 텍스트에 대해서 비교적 쉽게 구현할 수 있을 것이다.

    DL4J로 RNN을 구현하는 내용은 아래 링크들을 참고하자.
    http://deeplearning4j.org/usingrnns
    http://deeplearning4j.org/lstm
    http://deeplearning4j.org/recurrentnetwork
 */
public class GravesLSTMCharModellingExample {
	public static void main( String[] args ) throws Exception {
		int lstmLayerSize = 200;					// GravesLSTM 계층의 수
		int miniBatchSize = 32;						// 학습 할 때 사용할 미니 배치의 크기
		int exampleLength = 1000;					// 사용할 각 학습 예제 시퀀스의 길이. 증가 될 수있다.
        int tbpttLength = 50;                       // 시간에 따른 잘린 backpropagation의 길이. 즉, 매개 변수 업데이트를 50 자까지 수행한다.
		int numEpochs = 1;							// 총 훈련 에포크 수
        int generateSamplesEveryNMinibatches = 10;  // 신경망에서 샘플을 생성하는 빈도, 1000 문자 / 50 tbptt 길이 : minibatch 당 20 매개 변수 업데이트
		int nSamplesToGenerate = 4;					// 각 학습 기간 이후에 생성 할 샘플 수
		int nCharactersToSample = 300;				// 생성할 샘플의 길이
		String generationInitialization = null;		// 선택적 문자 초기화; null의 경우는 무작위의 문자가 사용된다
		// 위 내용은 문자 시퀀스로 LSTM을 '초기화'하여 학습을 수행하는데 사용된다.
		// 초기화 문자는 모두 디폴트로 CharacterIterator.getMinimalCharacterSet()에 있어야 한다.
		Random rng = new Random(12345);

		//GravesLSTM 신경망을 학습하는데 사용할 수 있는 텍스트의 벡터화를 처리하는 DataSetIterator를 가져온다.
		CharacterIterator iter = getShakespeareIterator(miniBatchSize,exampleLength);
		int nOut = iter.totalOutcomes();

		//신경망 설정
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
			.learningRate(0.1)
			.rmsDecay(0.95)
			.seed(12345)
			.regularization(true)
			.l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.RMSPROP)
			.list()
			.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
					.activation(Activation.TANH).build())
			.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.activation(Activation.TANH).build())
			.layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
					.nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
			.pretrain(false).backprop(true)
			.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		//신경망의 매개 변수 수를 출력한다 (각 계층마다).
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);

		//학습 시작, 신경망으로부터 샘플 출력
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

			iter.reset();	//다른 에포크를 위해 반복자 초기화
		}

		System.out.println("\n\nExample complete");
	}

	/** 셰익스피어 학습 데이터를 다운로드하여 로컬에 저장한다 (임시 디렉토리). 그런 다음 텍스트를 기반으로 벡터화를 수행하는 간단한 DataSetIterator를 설정하고 반환한다.
	 * @param miniBatchSize 각 학습 미니 배치의 텍스트 세그먼트 수
	 * @param sequenceLength 각 텍스트 세그먼트별로 문자 길이
	 */
	public static CharacterIterator getShakespeareIterator(int miniBatchSize, int sequenceLength) throws Exception{
		//학습데이터 원본 이름 : The Complete Works of William Shakespeare
		//5.3MB file in UTF-8 Encoding, ~5.4 백만 자
		//https://www.gutenberg.org/ebooks/100
		String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
		String tempDir = System.getProperty("java.io.tmpdir");
		String fileLocation = tempDir + "/Shakespeare.txt";	//Storage location from downloaded file
		File f = new File(fileLocation);
		if( !f.exists() ){
			FileUtils.copyURLToFile(new URL(url), f);
			System.out.println("File downloaded to " + f.getAbsolutePath());
		} else {
			System.out.println("Using existing text file at " + f.getAbsolutePath());
		}

		if(!f.exists()) throw new IOException("File does not exist: " + fileLocation);	//다운로드에 문제가 생기면 에러를 던진다

		char[] validCharacters = CharacterIterator.getMinimalCharacterSet();	//허용되는 문자는 무엇입니까? 기타는 제거된다.
		return new CharacterIterator(fileLocation, Charset.forName("UTF-8"),
				miniBatchSize, sequenceLength, validCharacters, new Random(12345));
	}

	/** 네트워크로부터의 샘플을 생성한다 (null의 경우는, 옵션). 초기화를 사용하여 확장 / 계속하려는 시퀀스로 RNN을 '초기화'할 수 있다.<br>
	 *  초기화는 모든 샘플에 사용된다.
	 * @param initialization String형, null 일 가능성이있다. null의 경우, 모든 샘플의 초기화로서 무작위의 문자를 선택한다.
	 * @param charactersToSample 신경망에서 샘플링 할 문자 수 (초기화 제외)
	 * @param net 하나 이상의 GravesLSTM / RNN 레이어와 softmax 출력 레이어가있는 MultiLayerNetwork
	 * @param iter CharacterIterator. 인덱스에서 문자로 이동하는 데 사용된다.
	 */
	private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net,
                                                        CharacterIterator iter, Random rng, int charactersToSample, int numSamples ){
		//초기화를 설정하자. 초기화가없는 경우 : 임의 문자 사용
		if( initialization == null ){
			initialization = String.valueOf(iter.getRandomCharacter());
		}

		//초기화를 위한 입력 데이터 생성
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

		//한 번에 한 문자 씩 신경망에서 샘플링 (및 피드 샘플을 입력으로 다시 입력) (모든 샘플에 적용 가능)
		//샘플링은 여기에서 병렬로 수행된다.
		net.rnnClearPreviousState();
		INDArray output = net.rnnTimeStep(initializationInput);
		output = output.tensorAlongDimension(output.size(2)-1,1,0);	//마지막 단계의 출력을 가져온다.

		for( int i=0; i<charactersToSample; i++ ){
			//이전 출력에서 샘플링하여 다음 입력 설정
			INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns());
			//출력은 확률 분포이다. 생성하려는 각 예제에 대한 샘플을 새 입력에 추가해주자.
			for( int s=0; s<numSamples; s++ ){
				double[] outputProbDistribution = new double[iter.totalOutcomes()];
				for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
				int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution,rng);

				nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//다음단계 입력 준비
				sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));	//샘플 된 문자를 StringBuilder에 추가 (사람이 읽을 수있는 출력).
			}

			output = net.rnnTimeStep(nextInput);	//정방향 전달의 한단계 수행
		}

		String[] out = new String[numSamples];
		for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
		return out;
	}

	/** 불연속 클래스에 대한 확률 분포가 주어지면, 분포로부터 샘플링하고 생성 된 클래스 인덱스를 반환한다.
	 * @param distribution 클래스에 대한 확률 분포. 1.0에 합계되어야 함
	 */
	public static int sampleFromDistribution( double[] distribution, Random rng ){
		double d = rng.nextDouble();
		double sum = 0.0;
		for( int i=0; i<distribution.length; i++ ){
			sum += distribution[i];
			if( d <= sum ) return i;
		}
		//분포가 유효한 확률 분포라면 결코 일어나지 않아야한다.
		throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
	}
}
