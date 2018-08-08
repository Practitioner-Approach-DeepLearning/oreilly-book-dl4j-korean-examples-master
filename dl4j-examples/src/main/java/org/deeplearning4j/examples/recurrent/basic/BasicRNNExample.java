package org.deeplearning4j.examples.recurrent.basic;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
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

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;

/**
 * 이 예제는 RNN을 학습하는 예제이다. 
 * 학습 후에는 RNN에 LEARNSTRING의 첫 번째 문자만 입력하면 된다. 그러면 다음 문자를 이야기할 것이다.
 *
 * @author Peter Grossmann
 */
public class BasicRNNExample {

	// 학습을 위한 문장
	public static final char[] LEARNSTRING = "Der Cottbuser Postkutscher putzt den Cottbuser Postkutschkasten.".toCharArray();

	// 모든 가능한 문자 리스트
	public static final List<Character> LEARNSTRING_CHARS_LIST = new ArrayList<Character>();

	// RNN 디멘젼
	public static final int HIDDEN_LAYER_WIDTH = 50;
	public static final int HIDDEN_LAYER_CONT = 2;
	public static final Random r = new Random(7894);

	public static void main(String[] args) {

		// LEARNSTRING_CHARS_LIST에 가능한 문자의 전용 목록을 만든다.
		LinkedHashSet<Character> LEARNSTRING_CHARS = new LinkedHashSet<Character>();
		for (char c : LEARNSTRING)
			LEARNSTRING_CHARS.add(c);
		LEARNSTRING_CHARS_LIST.addAll(LEARNSTRING_CHARS);

		// 공통 패턴
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		builder.iterations(10);
		builder.learningRate(0.001);
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		builder.seed(123);
		builder.biasInit(0);
		builder.miniBatch(false);
		builder.updater(Updater.RMSPROP);
		builder.weightInit(WeightInit.XAVIER);

		ListBuilder listBuilder = builder.list();

		// 첫 번째 차이점은, GravesLSTM.Builder를 사용해야한다는 것이다.
		for (int i = 0; i < HIDDEN_LAYER_CONT; i++) {
			GravesLSTM.Builder hiddenLayerBuilder = new GravesLSTM.Builder();
			hiddenLayerBuilder.nIn(i == 0 ? LEARNSTRING_CHARS.size() : HIDDEN_LAYER_WIDTH);
			hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH);
			// GravesLSTMCharModellingExample의 활성 함수가 RNN과 잘 작동한다.
			hiddenLayerBuilder.activation(Activation.TANH);
			listBuilder.layer(i, hiddenLayerBuilder.build());
		}

		// RNN을 위해 RnnOutputLayer을 사용할 필요가 있다.
		RnnOutputLayer.Builder outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MCXENT);
		// softmax는 출력 뉴런을 정규화하고, 모든 출력의 합이 1이 되도록 한다.
		// 이것은 샘플분포함수에 필요하다.
		outputLayerBuilder.activation(Activation.SOFTMAX);
		outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH);
		outputLayerBuilder.nOut(LEARNSTRING_CHARS.size());
		listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build());

		//빌더 종료
		listBuilder.pretrain(false);
		listBuilder.backprop(true);

		//신경망 생성
		MultiLayerConfiguration conf = listBuilder.build();
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		/*
		 * 학습데이터 생성
		 */
		// 입력 및 출력 배열 만들기 : SAMPLE_INDEX, INPUT_NEURON, SEQUENCE_POSITION
		INDArray input = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size(), LEARNSTRING.length);
		INDArray labels = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size(), LEARNSTRING.length);
		// 예제 문장을 탐색
		int samplePos = 0;
		for (char currentChar : LEARNSTRING) {
			// 작은 해킹 : currentChar가 마지막 일 때 첫 번째 char을 nextChar로 가져온다. - 꼭 필요한 것은 아니다.
			char nextChar = LEARNSTRING[(samplePos + 1) % (LEARNSTRING.length)];
			// current-char에 대한 입력 뉴런은 "samplePos"에서 1이다.
			input.putScalar(new int[] { 0, LEARNSTRING_CHARS_LIST.indexOf(currentChar), samplePos }, 1);
			// next-char에 대한 출력 뉴런은 "samplePos"에서 1이다.
			labels.putScalar(new int[] { 0, LEARNSTRING_CHARS_LIST.indexOf(nextChar), samplePos }, 1);
			samplePos++;
		}
		DataSet trainingData = new DataSet(input, labels);

		// 에포크 시작
		for (int epoch = 0; epoch < 100; epoch++) {

			System.out.println("Epoch " + epoch);

			//데이터 학습
			net.fit(trainingData);

			// 지난예제로부터 온 상태 초기화
			net.rnnClearPreviousState();

			// 초기화를 위해 첫번째 문자를 RNN에 둔다.
			INDArray testInit = Nd4j.zeros(LEARNSTRING_CHARS_LIST.size());
			testInit.putScalar(LEARNSTRING_CHARS_LIST.indexOf(LEARNSTRING[0]), 1);

			// 한 단계 실행 -> 중요 : output()이 아니라 rnnTimeStep()이 호출 되어야한다. 출력은 신경망이 다음에 무엇을 생각하는지 보여준다.
			INDArray output = net.rnnTimeStep(testInit);

			//이제 LEARNSTRING.length 보다 더 많은 문자가 예상되는 신경망을 추측해야한다.
			for (int j = 0; j < LEARNSTRING.length; j++) {

				// 처음에는 구체적인 뉴런에 대한 신경망의 최종 출력을 처리하고, 가장 높은 출력을 가진 뉴런은 가장 높은 선택을 받는다.
				double[] outputProbDistribution = new double[LEARNSTRING_CHARS.size()];
				for (int k = 0; k < outputProbDistribution.length; k++) {
					outputProbDistribution[k] = output.getDouble(k);
				}
				int sampledCharacterIdx = findIndexOfHighestValue(outputProbDistribution);

				// 선택한 출력을 출력하자
				System.out.print(LEARNSTRING_CHARS_LIST.get(sampledCharacterIdx));

				// 마지막 출력을 입력으로 사용
				INDArray nextInput = Nd4j.zeros(LEARNSTRING_CHARS_LIST.size());
				nextInput.putScalar(sampledCharacterIdx, 1);
				output = net.rnnTimeStep(nextInput);

			}
			System.out.print("\n");

		}

	}

	private static int findIndexOfHighestValue(double[] distribution) {
		int maxValueIndex = 0;
		double maxValue = 0;
		for (int i = 0; i < distribution.length; i++) {
			if(distribution[i] > maxValue) {
				maxValue = distribution[i];
				maxValueIndex = i;
			}
		}
		return maxValueIndex;
	}

}
