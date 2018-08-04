package org.deeplearning4j.rnn;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.*;

/**
 * GravesLSTM + 스파크 문자 모델링 예제
 * 예제: 한번에 문자 하나씩, 문자열을 생성하는 LSTM 순환신경망을 학습하라.
 * 이 예제는 스파크를 사용해 학습한다.
 *
 * 이 예제의 단일 머신 버전은
 * dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/character/GravesLSTMCharModellingExample.java를 참고하라.
 *
 * 예제를 로컬에서 돌리고 싶다면 예제를 그대로 실행하면 된다. 이 예제는 기본적으로 스파크 로컬에서 실행하도록 구성되어 있다.
 * 주의: 스파크 로컬은 개발/테스트 용으로만 사용해야 한다. 대신 (다중 GPU 시스템 같은) 단일 머신에서 데이터를 병렬 학습시킬 때는
 * ParallelWrapper를 사용하라 (단일 머신에서 스파크를 사용할 때보다 빠르다).
 * dl4j-cuda-specific-examples에서 MultiGpuLenetMnistExample을 참고하라.
 *
 * (클러스터에서 실행하기 위해) 스파크 서브밋을 사용해 예제를 실행시키고 싶다면 "-useParkLocal false"를 애플리케이션 매개변수에 포함시키거나,
 * 예제 앞 부분의 필드를 "useSparkLocal = false"로 설정하라.
 *
 * @author 알렉스 블랙
 */
public class SparkLSTMCharacterExample {
    private static final Logger log = LoggerFactory.getLogger(SparkLSTMCharacterExample.class);

    private static Map<Integer, Character> INT_TO_CHAR = getIntToChar();
    private static Map<Character, Integer> CHAR_TO_INT = getCharToInt();
    private static final int N_CHARS = INT_TO_CHAR.size();
    private static int nOut = CHAR_TO_INT.size();
    private static int exampleLength = 1000;                    //학습 입력 데이터 시퀀스의 길이

    @Parameter(names = "-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = true;

    @Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = 8;   // 워커(익스큐터) 당 처리할 입력 데이터 개수

    @Parameter(names = "-numEpochs", description = "Number of epochs for training")
    private int numEpochs = 1;

    public static void main(String[] args) throws Exception {
        new SparkLSTMCharacterExample().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        // 명령줄 인자 다루기
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            // 사용자가 잘못 입력함 -> 사용법 출력
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            throw e;
        }

        Random rng = new Random(12345);
        int lstmLayerSize = 200;                    // 각 GravesLSTM 계층의 유닛 수
        int tbpttLength = 50;                       // 단기 BPTT의 길이. 파라미터 업데이트를 50자까지 수행
        int nSamplesToGenerate = 4;                    // 각 학습 에포크당 생성할 샘플 수
        int nCharactersToSample = 300;                // 생성할 각 샘플의 길이
        String generationInitialization = null;        // 선택적 문자 초기화. null의 경우 무작위 문자가 사용됨
        // 위 문자 시퀀스를 사용해 LSTM을 초기화해 계속 진행/완료한다.
        // 기본적으로 초기화 문자는 모두 CharacterIterator.getMinimalCharacterSet()중에서 사용됨

        // 신경망 구성 설정
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
            .layer(0, new GravesLSTM.Builder().nIn(CHAR_TO_INT.size()).nOut(lstmLayerSize).activation(Activation.TANH).build())
            .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize).activation(Activation.TANH).build())
            .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)        // MCXENT + 소프트맥스를 사용해 분류
                .nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
            .pretrain(false).backprop(true)
            .build();


        //-------------------------------------------------------------
        // 스파크 관련 구성 설정
        /* 얼마나 자주 파라미터 평균화를 사용하는가(미니배치 수)?
        너무 자주 평균화하면 속도가 느려지고(동기화 + 직렬화 비용)
        너무 드물게 평균화하면 학습에 어려움을 겪을 수 있다 (즉, 신경망이 수렴하지 못할 수 있음) */
        int averagingFrequency = 3;

        // 스파크 구성 및 컨텍스트 설정
        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        sparkConf.setAppName("LSTM Character Example");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        JavaRDD<DataSet> trainingData = getTrainingData(sc);


        // TrainingMaster를 설정. TrainingMaster는 스파크로 학습되는 과정을 제어한다.
        // 여기서는 기본적인 파라미터 평균화를 사용한다.
        // 이러한 구성 옵션에 대한 자세한 내용은 https://deeplearning4j.org/spark#configuring 링크를 참고하라.
        int examplesPerDataSetObject = 1;
        ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
            .workerPrefetchNumBatches(2)    // 비동기로 배치를 최대 2개 미리 가져옴
            .averagingFrequency(averagingFrequency)
            .batchSizePerWorker(batchSizePerWorker)
            .build();
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, conf, tm);
        sparkNetwork.setListeners(Collections.<IterationListener>singletonList(new ScoreIterationListener(1)));

        // 학습을 수행한 다음 신경망에서 샘플을 생성하고 출력
        for (int i = 0; i < numEpochs; i++) {
            // 에포크 하나에 대한 학습을 수행. 각 에포크가 끝나면 학습된 신경망 사본 반환
            MultiLayerNetwork net = sparkNetwork.fit(trainingData);

            // 신경망에서 일부 문자 샘플(로컬로 수행)
            log.info("Sampling characters from network given initialization \"" +
                (generationInitialization == null ? "" : generationInitialization) + "\"");
            String[] samples = sampleCharactersFromNetwork(generationInitialization, net, rng, INT_TO_CHAR,
                nCharactersToSample, nSamplesToGenerate);
            for (int j = 0; j < samples.length; j++) {
                log.info("----- Sample " + j + " -----");
                log.info(samples[j]);
            }
        }

        // 임시 학습 파일 삭제, 완료
        tm.deleteTempFiles(sc);

        log.info("\n\nExample complete");
    }


    /**
     * 학습 데이터 가져오기 - JavaRDD<DataSet>
     * 아래 메소드는 특이한 방법으로 학습 데이터(모델링 문자)를 가져오고 있다.
     * 일반적으로 CSV 등의 데이터를 로드할 때 이렇게 구현해서는 안 된다.
     *
     * should  not be taken as best practice for loading data (like CSV etc) in general.
     */
    public static JavaRDD<DataSet> getTrainingData(JavaSparkContext sc) throws IOException {
        // 데이터 가져오기. 이 예제에서는 다음과 같이 작업을 수행한다.
        // File -> String -> List<String> (split into length "sequenceLength" characters) -> JavaRDD<String> -> JavaRDD<DataSet>
        List<String> list = getShakespeareAsList(exampleLength);
        JavaRDD<String> rawStrings = sc.parallelize(list);
        Broadcast<Map<Character, Integer>> bcCharToInt = sc.broadcast(CHAR_TO_INT);
        return rawStrings.map(new StringToDataSetFn(bcCharToInt));
    }


    private static class StringToDataSetFn implements Function<String, DataSet> {
        private final Broadcast<Map<Character, Integer>> ctiBroadcast;

        private StringToDataSetFn(Broadcast<Map<Character, Integer>> characterIntegerMap) {
            this.ctiBroadcast = characterIntegerMap;
        }

        @Override
        public DataSet call(String s) throws Exception {
            // 여기에서 String을 가져와 문자를 one-hot 표현으로 매핑한다.
            Map<Character, Integer> cti = ctiBroadcast.getValue();
            int length = s.length();
            INDArray features = Nd4j.zeros(1, N_CHARS, length - 1);
            INDArray labels = Nd4j.zeros(1, N_CHARS, length - 1);
            char[] chars = s.toCharArray();
            int[] f = new int[3];
            int[] l = new int[3];
            for (int i = 0; i < chars.length - 2; i++) {
                f[1] = cti.get(chars[i]);
                f[2] = i;
                l[1] = cti.get(chars[i + 1]);   // 이전 및 현재 문자가 주어진 다음 문자를 예측
                l[2] = i;

                features.putScalar(f, 1.0);
                labels.putScalar(l, 1.0);
            }
            return new DataSet(features, labels);
        }
    }

    // 이 메소드는 원본 텍스트 데이터를 (필요하다면 다운로드한 후) 불러와 sequenceLength 길이의 문자열로 분할한다.
    private static List<String> getShakespeareAsList(int sequenceLength) throws IOException {
        // 윌리엄 셰익스피어 전집
        //UTF-8 인코딩된 5.3MB 파일, 문자 약 5백4십만개
        //https://www.gutenberg.org/ebooks/100
        String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
        String tempDir = System.getProperty("java.io.tmpdir");
        String fileLocation = tempDir + "/Shakespeare.txt";    // 다운로드 파일 저장 경로
        File f = new File(fileLocation);
        if (!f.exists()) {
            FileUtils.copyURLToFile(new URL(url), f);
            System.out.println("File downloaded to " + f.getAbsolutePath());
        } else {
            System.out.println("Using existing text file at " + f.getAbsolutePath());
        }

        if (!f.exists()) throw new IOException("File does not exist: " + fileLocation);    // 다운로드 실패?

        String allData = getDataAsString(fileLocation);

        List<String> list = new ArrayList<>();
        int length = allData.length();
        int currIdx = 0;
        while (currIdx + sequenceLength < length) {
            int end = currIdx + sequenceLength;
            String substr = allData.substring(currIdx, end);
            currIdx = end;
            list.add(substr);
        }
        return list;
    }

    /**
     * 파일에서 데이터를 불러와 깨진 문자를 모두 제거한다.
     * 데이터는 매우 긴 단일 문자열로 반환된다.
     */
    private static String getDataAsString(String filePath) throws IOException {
        List<String> lines = Files.readAllLines(new File(filePath).toPath(), Charset.defaultCharset());
        StringBuilder sb = new StringBuilder();
        for (String line : lines) {
            char[] chars = line.toCharArray();
            for (int i = 0; i < chars.length; i++) {
                if (CHAR_TO_INT.containsKey(chars[i])) sb.append(chars[i]);
            }
            sb.append("\n");
        }

        return sb.toString();
    }

    /**
     * 주어진 initialization으로(null일 수 있음) 신경망으로부터 샘플을 생성한다.
     * 순환신경망을 확장/학습을 지속할 때 initialization을 순환신경망을 '초기화'하는데 사용할 수 있다.
     * initialization은 모든 샘플에서 사용된다.
     *
     * @param initialization     문자열, null일 수 있음. null인 경우 임의의 문자를 사용해 모든 샘플에 대한 initialization으로 사용한다.
     * @param charactersToSample 신경망에서 샘플링할 문자 개수 (initialization 제외)
     * @param net                MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
     */
    private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net, Random rng,
                                                        Map<Integer, Character> intToChar, int charactersToSample, int numSamples) {
        // initialization 설정. initialization이 null이라면 임의의 문자 사용
        if (initialization == null) {
            int randomCharIdx = rng.nextInt(intToChar.size());
            initialization = String.valueOf(intToChar.get(randomCharIdx));
        }

        // initialization으로 입력 생성
        INDArray initializationInput = Nd4j.zeros(numSamples, intToChar.size(), initialization.length());
        char[] init = initialization.toCharArray();
        for (int i = 0; i < init.length; i++) {
            int idx = CHAR_TO_INT.get(init[i]);
            for (int j = 0; j < numSamples; j++) {
                initializationInput.putScalar(new int[]{j, idx, i}, 1.0f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for (int i = 0; i < numSamples; i++) sb[i] = new StringBuilder(initialization);

        // 한 번에 한 문자 씩 신경망 샘플링 (샘플은 입력으로 재활용)
        // 이 예제에서는 샘플링이 병렬로 수행된다.
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension(output.size(2) - 1, 1, 0);    // 마지막 시간 단계의 출력 가져오기

        for (int i = 0; i < charactersToSample; i++) {
            // 이전 출력에서 샘플링해 다음 입력(시간 단계 한 개) 설정
            INDArray nextInput = Nd4j.zeros(numSamples, intToChar.size());
            // 출력은 확률 분포다. 생성하고자 하는 각 입력 데이터마다 출력(확률 분포)를 샘플링해 새로운 입력에 추가
            for (int s = 0; s < numSamples; s++) {
                double[] outputProbDistribution = new double[intToChar.size()];
                for (int j = 0; j < outputProbDistribution.length; j++)
                    outputProbDistribution[j] = output.getDouble(s, j);
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng);

                nextInput.putScalar(new int[]{s, sampledCharacterIdx}, 1.0f);        // 다음 시간 단계의 입력을 준비
                sb[s].append(intToChar.get(sampledCharacterIdx));    // 출력(샘플링 된 문자 색인)을 사람이 읽을 수 있는 문자로 변환해 StringBuilder에 추가
            }

            output = net.rnnTimeStep(nextInput);    // 다음 시간 단계를 수행
        }

        String[] out = new String[numSamples];
        for (int i = 0; i < numSamples; i++) out[i] = sb[i].toString();
        return out;
    }

    /**
     * 불연속 클래스에 대한 확률 분포가 주어지면 분포로부터 클래스 색인 하나를 샘플링해 반환한다.
     *
     * @param distribution 클래스 확률 분포. 합이 1.0이어야 함
     */
    private static int sampleFromDistribution(double[] distribution, Random rng) {
        double d = rng.nextDouble();
        double sum = 0.0;
        for (int i = 0; i < distribution.length; i++) {
            sum += distribution[i];
            if (d <= sum) return i;
        }
        // 확률 분포가 유효하다면 이 줄은 결코 실행되지 않는다.
        throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum);
    }

    /**
     * a-z, A-Z, 0-9, 구두점 등으로 구성된 약식 문자셋
     */
    private static char[] getValidCharacters() {
        List<Character> validChars = new LinkedList<>();
        for (char c = 'a'; c <= 'z'; c++) validChars.add(c);
        for (char c = 'A'; c <= 'Z'; c++) validChars.add(c);
        for (char c = '0'; c <= '9'; c++) validChars.add(c);
        char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
        for (char c : temp) validChars.add(c);
        char[] out = new char[validChars.size()];
        int i = 0;
        for (Character c : validChars) out[i++] = c;
        return out;
    }

    public static Map<Integer, Character> getIntToChar() {
        Map<Integer, Character> map = new HashMap<>();
        char[] chars = getValidCharacters();
        for (int i = 0; i < chars.length; i++) {
            map.put(i, chars[i]);
        }
        return map;
    }

    public static Map<Character, Integer> getCharToInt() {
        Map<Character, Integer> map = new HashMap<>();
        char[] chars = getValidCharacters();
        for (int i = 0; i < chars.length; i++) {
            map.put(chars[i], i);
        }
        return map;
    }
}
