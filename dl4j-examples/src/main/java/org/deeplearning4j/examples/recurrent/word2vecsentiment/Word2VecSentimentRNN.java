package org.deeplearning4j.examples.recurrent.word2vecsentiment;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
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
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.net.URL;

/**예제 : 영화 리뷰 (원문)가 주어지면, 그 영화 리뷰에 들어있는 단어를 기준으로 영화 리뷰를 양수 또는 음수로 분류.
 * 이것은 Word2Vec 벡터와 반복적 인 신경망 모델을 결합하여 이루어진다. 리뷰의 각 단어는 벡터화되어 (Word2Vec 모델 사용) 반복적 인 신경망으로 공급된다.
 * 학습용 데이터는 "Large Movie Review Dataset" from http://ai.stanford.edu/~amaas/data/sentiment/ 이다.
 * 이 데이터셋은 25,000 학습용 리뷰 데이터셋과 25,000개의 훈련 데이터셋으로 구성되어 있다.
 *
 * 과정:
 * 1. 데이터 다운로드
 * 2. 이미 만들어 놓은 Word2Vector 모델 사용
 * 3. 각각 리뷰를 로드한다. 단어를 벡터로 변환하고 벡터의 시퀀스를 검토한다.
 * 4. 신경망 학습
 *
 * 현재 구성을 사용하면 약. 1 epoch 이후 83 %의 정확도. 추가 튜닝을 사용하면 더 나은 성능을 얻을 수 있다.
 *
 * 주의사항 및 지시사항:
 * Google News word vector 모델 받는 방법
 * Google News word vector 모델은 해당 URL에서 확인 가능하다.: https://code.google.com/p/word2vec/
 * 다운로드 한다 : GoogleNews-vectors-negative300.bin.gz 파일  ~1.5GB
 * 그런 다음 이 위치를 가리 키도록 WORD_VECTORS_PATH 필드를 설정.
 *
 * @author Alex Black
 */
public class Word2VecSentimentRNN {

    /** 데이터 다운로드 URL */
    public static final String DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
    /** 학습, 테스트 데이터셋을 저장하는 위치  */
    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/");
    /** Google News Vector를 저장할 위치  */
    public static final String WORD_VECTORS_PATH = "/PATH/TO/YOUR/VECTORS/GoogleNews-vectors-negative300.bin.gz";


    public static void main(String[] args) throws Exception {
        if(WORD_VECTORS_PATH.startsWith("/PATH/TO/YOUR/VECTORS/")){
            throw new RuntimeException("Please set the WORD_VECTORS_PATH before running this example");
        }

        //데이터를 다운로드하고 추출한다
        downloadData();

        int batchSize = 64;     //각 미니배치별 예제 개수
        int vectorSize = 300;   //워드 벡터의 크기, Google News Vector의 경우 300
        int nEpochs = 1;        //학습을 위한 에포크 수
        int truncateReviewsToLength = 256;  //truncateReviewsToLength보다 긴 리뷰는 잘라낸다.

        //신경망 설정
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .updater(Updater.ADAM).adamMeanDecay(0.9).adamVarDecay(0.999)
            .regularization(true).l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .learningRate(2e-2)
            .list()
            .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(256)
                .activation(Activation.TANH).build())
            .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
            .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        //교육 및 테스트 용 DataSetIterators
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
        SentimentExampleIterator train = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, true);
        SentimentExampleIterator test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false);

        System.out.println("Starting training");
        for (int i = 0; i < nEpochs; i++) {
            net.fit(train);
            train.reset();
            System.out.println("Epoch " + i + " complete. Starting evaluation:");

            //평가를 실행한다. 리뷰의 크기 때문에 시간이 좀 걸릴 수 있다.
            Evaluation evaluation = new Evaluation();
            while (test.hasNext()) {
                DataSet t = test.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = net.output(features, false, inMask, outMask);

                evaluation.evalTimeSeries(lables, predicted, outMask);
            }
            test.reset();

            System.out.println(evaluation.stats());
        }

        //교육 후 : 단일 예제를 로드하고 예측을 생성한다.
        File firstPositiveReviewFile = new File(FilenameUtils.concat(DATA_PATH, "aclImdb/test/pos/0_10.txt"));
        String firstPositiveReview = FileUtils.readFileToString(firstPositiveReviewFile);

        INDArray features = test.loadFeaturesFromString(firstPositiveReview, truncateReviewsToLength);
        INDArray networkOutput = net.output(features);
        int timeSeriesLength = networkOutput.size(2);
        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

        System.out.println("\n\n-------------------------------");
        System.out.println("First positive review: \n" + firstPositiveReview);
        System.out.println("\n\nProbabilities at last time step:");
        System.out.println("p(positive): " + probabilitiesAtLastWord.getDouble(0));
        System.out.println("p(negative): " + probabilitiesAtLastWord.getDouble(1));

        System.out.println("----- Example complete -----");
    }

    private static void downloadData() throws Exception {
        //필요하다면 디렉토리도 생성한다.
        File directory = new File(DATA_PATH);
        if(!directory.exists()) directory.mkdir();

        //파일 다운로드
        String archizePath = DATA_PATH + "aclImdb_v1.tar.gz";
        File archiveFile = new File(archizePath);
        String extractedPath = DATA_PATH + "aclImdb";
        File extractedFile = new File(extractedPath);

        if( !archiveFile.exists() ){
            System.out.println("Starting data download (80MB)...");
            FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile);
            System.out.println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath());
            //output 디렉토리에 tar.gz파일의 압축을 풀어준다.
            extractTarGz(archizePath, DATA_PATH);
        } else {
            //아카이브 (.tar.gz)가 있고 데이터가 이미 추출되었다고 가정한다.
            System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
            if( !extractedFile.exists()){
            	//output 디렉토리에 tar.gz파일의 압축을 풀어준다.
            	extractTarGz(archizePath, DATA_PATH);
            } else {
            	System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
            }
        }
    }

    private static final int BUFFER_SIZE = 4096;
    private static void extractTarGz(String filePath, String outputPath) throws IOException {
        int fileCount = 0;
        int dirCount = 0;
        System.out.print("Extracting files");
        try(TarArchiveInputStream tais = new TarArchiveInputStream(
                new GzipCompressorInputStream( new BufferedInputStream( new FileInputStream(filePath))))){
            TarArchiveEntry entry;

            /** getNextEntry 메소드를 사용하여 tar 항목 읽기 **/
            while ((entry = (TarArchiveEntry) tais.getNextEntry()) != null) {
                //System.out.println("Extracting file: " + entry.getName());

                //필요하다면 디렉토리 생성
                if (entry.isDirectory()) {
                    new File(outputPath + entry.getName()).mkdirs();
                    dirCount++;
                }else {
                    int count;
                    byte data[] = new byte[BUFFER_SIZE];

                    FileOutputStream fos = new FileOutputStream(outputPath + entry.getName());
                    BufferedOutputStream dest = new BufferedOutputStream(fos,BUFFER_SIZE);
                    while ((count = tais.read(data, 0, BUFFER_SIZE)) != -1) {
                        dest.write(data, 0, count);
                    }
                    dest.close();
                    fileCount++;
                }
                if(fileCount % 1000 == 0) System.out.print(".");
            }
        }

        System.out.println("\n" + fileCount + " files and " + dirCount + " directories extracted to: " + outputPath);
    }
}
