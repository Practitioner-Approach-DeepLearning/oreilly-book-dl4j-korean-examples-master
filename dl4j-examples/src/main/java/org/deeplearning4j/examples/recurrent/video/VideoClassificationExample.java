package org.deeplearning4j.examples.recurrent.video;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.codec.reader.CodecRecordReader;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

/**
 * 비디오의 각 프레임을 분류하기 위해 CNN, RNN, 맥스풀링, Dense 계층 (feed forward)를 조합해서 사용하는 방법에 대한 예제이다.
 * 구체적으로 각 비디오에는 여러 프레임 동안 지속되는 모양 (원, 사각형, 선 등)이 프레임에 포함되어 있다. 각 비디오는 프레임에 임의의 숫자의 모양들을 가지고 있다.
 * 프레임에 모양이 남는다면 그것들을 분류 해야한다.
 *
 * 이 예제는 몇몇 인위적으로 만들어진 부분이 있지만 비디오 프래임들을 분류하기 위해 데이터를 불러오고 신경망을 구성한다.
 *
 * *******************************************************
 * 경고: 이 예제는 많은 데이터셋을 만들어낸다.
 * 이 예제는 예제를 다 실행했더라도 자동으로 삭제되지 않는다.
 * *******************************************************
 * @author Alex Black
 */
public class VideoClassificationExample {

    public static final int N_VIDEOS_TO_GENERATE = 500;
    public static final int V_WIDTH = 130;
    public static final int V_HEIGHT = 130;
    public static final int V_NFRAMES = 150;

    public static void main(String[] args) throws Exception {

        int miniBatchSize = 10;
        boolean generateData = true;

        String tempDir = System.getProperty("java.io.tmpdir");
        String dataDirectory = FilenameUtils.concat(tempDir, "DL4JVideoShapesExample/");   //만들어진 데이터셋을 저장할 위치

        //데이터 생성: .mp4 형식의 비디오 숫자와 레이블을 위한 .txt 파일들을 입력으로 받는다
        if (generateData) {
            System.out.println("Starting data generation...");
            generateData(dataDirectory);
            System.out.println("Data generation complete");
        }

        //신경망 구성
        Updater updater = Updater.ADAGRAD;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .regularization(true).l2(0.001) //모든계층에서 l2 정규화를 사용한다.
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.04)
                .list()
                .layer(0, new ConvolutionLayer.Builder(10, 10)
                        .nIn(3) //3채널은 RGB를 뜻한다.
                        .nOut(30)
                        .stride(4, 4)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.RELU)
                        .updater(updater)
                        .build())   //출력: (130-10+0)/4+1 = 31 -> 31*31*30
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2).build())   //(31-3+0)/2+1 = 15
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nIn(30)
                        .nOut(10)
                        .stride(2, 2)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.RELU)
                        .updater(updater)
                        .build())   //출력: (15-3+0)/2+1 = 7 -> 7*7*10 = 490
                .layer(3, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nIn(490)
                        .nOut(50)
                        .weightInit(WeightInit.RELU)
                        .updater(updater)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .learningRate(0.01)
                        .build())
                .layer(4, new GravesLSTM.Builder()
                        .activation(Activation.SOFTSIGN)
                        .nIn(50)
                        .nOut(50)
                        .weightInit(WeightInit.XAVIER)
                        .updater(updater)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .learningRate(0.008)
                        .build())
                .layer(5, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(50)
                        .nOut(4)    //4가지의 가능한 모양: circle, square, arc, line
                        .updater(updater)
                        .weightInit(WeightInit.XAVIER)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build())
                .inputPreProcessor(0, new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, 3))
                .inputPreProcessor(3, new CnnToFeedForwardPreProcessor(7, 7, 10))
                .inputPreProcessor(4, new FeedForwardToRnnPreProcessor())
                .pretrain(false).backprop(true)
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(V_NFRAMES / 5)
                .tBPTTBackwardLength(V_NFRAMES / 5)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        System.out.println("Number of parameters in network: " + net.numParams());
        for( int i=0; i<net.getnLayers(); i++ ){
            System.out.println("Layer " + i + " nParams = " + net.getLayer(i).numParams());
        }

        int testStartIdx = (int) (0.9 * N_VIDEOS_TO_GENERATE);  //90% 학습데이터, 10% 테스트데이터
        int nTest = N_VIDEOS_TO_GENERATE - testStartIdx;

        //학습 수행
        System.out.println("Starting training...");
        int nTrainEpochs = 15;
        for (int i = 0; i < nTrainEpochs; i++) {
            DataSetIterator trainData = getDataSetIterator(dataDirectory, 0, testStartIdx - 1, miniBatchSize);
            while(trainData.hasNext())
                net.fit(trainData.next());
            Nd4j.saveBinary(net.params(),new File("videomodel.bin"));
            FileUtils.writeStringToFile(new File("videoconf.json"), conf.toJson());
            System.out.println("Epoch " + i + " complete");

            //분류 성능을 향상시킨다
            evaluatePerformance(net,testStartIdx,nTest,dataDirectory);
        }
    }

    private static void generateData(String path) throws Exception {
        File f = new File(path);
        if (!f.exists()) f.mkdir();

        /** 데이터 생성 코드는 배경 소음과 인식할 수 없는 모양 (타겟 모양을 제외하고 한 프레임에만 등장하는 특수한 모양들)에 대해서 지원하지 않는다.
         * 하지만, 기본적으로 이러한 것들은 예외 처리된다.
         *
         * 학습의 복잡도를 증가 시키면 이러한 것들도 충분히 연산할 수 있다.
         */
        VideoGenerator.generateVideoData(path, "shapes", N_VIDEOS_TO_GENERATE,
                V_NFRAMES, V_WIDTH, V_HEIGHT,
                3,      //비디오당 모양의 개수, 시간이 지남에 따라 한 모양에서 다른 모양으로 무작위로 전환된다.
                false,    //배경 소음. 비디오 파일 사이즈를 크게한다.
                0,      //프레임당 인식할 수 없는 모양의 개수
                12345L);    //데이터 생성의 생산성을 높이기위한 시드
    }

    private static void evaluatePerformance(MultiLayerNetwork net, int testStartIdx, int nExamples, String outputDirectory) throws Exception {
        //여기서 전체 테스트 데이터셋이 메모리에 적합하지 않다면 한번에 10개의 예제만 수행하자
        Map<Integer, String> labelMap = new HashMap<>();
        labelMap.put(0, "circle");
        labelMap.put(1, "square");
        labelMap.put(2, "arc");
        labelMap.put(3, "line");
        Evaluation evaluation = new Evaluation(labelMap);

        DataSetIterator testData = getDataSetIterator(outputDirectory, testStartIdx, nExamples, 10);
        while(testData.hasNext()) {
            DataSet dsTest = testData.next();
            INDArray predicted = net.output(dsTest.getFeatureMatrix(), false);
            INDArray actual = dsTest.getLabels();
            evaluation.evalTimeSeries(actual, predicted);
        }

        System.out.println(evaluation.stats());
    }

    private static DataSetIterator getDataSetIterator(String dataDirectory, int startIdx, int nExamples, int miniBatchSize) throws Exception {=
        // 데이터와 레이블이 여러 파일로 분산되어 있다.
        // 비디오 :  shapes_0.mp4, shapes_1.mp4, etc
        // 레이블 : shapes_0.txt, shapes_1.txt, etc. 혹은 라인당 시간단계
        SequenceRecordReader featuresTrain = getFeaturesReader(dataDirectory, startIdx, nExamples);
        SequenceRecordReader labelsTrain = getLabelsReader(dataDirectory, startIdx, nExamples);

        SequenceRecordReaderDataSetIterator sequenceIter =
                new SequenceRecordReaderDataSetIterator(featuresTrain, labelsTrain, miniBatchSize, 4, false);
        sequenceIter.setPreProcessor(new VideoPreProcessor());

        // AsyncDataSetIterator: 별도의 스레드에서 미리 불러온 데이터를 사용하는데 이용된다.
        return new AsyncDataSetIterator(sequenceIter,1);
    }

    private static SequenceRecordReader getFeaturesReader(String path, int startIdx, int num) throws Exception {
        //nputSplit은 파일 경로의 모양을 정의하는 데 사용된다.
        InputSplit is = new NumberedFileInputSplit(path + "shapes_%d.mp4", startIdx, startIdx + num - 1);

        Configuration conf = new Configuration();
        conf.set(CodecRecordReader.RAVEL, "true");
        conf.set(CodecRecordReader.START_FRAME, "0");
        conf.set(CodecRecordReader.TOTAL_FRAMES, String.valueOf(V_NFRAMES));
        conf.set(CodecRecordReader.ROWS, String.valueOf(V_WIDTH));
        conf.set(CodecRecordReader.COLUMNS, String.valueOf(V_HEIGHT));
        CodecRecordReader crr = new CodecRecordReader();
        crr.initialize(conf, is);
        return crr;
    }

    private static SequenceRecordReader getLabelsReader(String path, int startIdx, int num) throws Exception {
        InputSplit isLabels = new NumberedFileInputSplit(path + "shapes_%d.txt", startIdx, startIdx + num - 1);
        CSVSequenceRecordReader csvSeq = new CSVSequenceRecordReader();
        csvSeq.initialize(isLabels);
        return csvSeq;
    }

    private static class VideoPreProcessor implements DataSetPreProcessor {
        @Override
        public void preProcess(org.nd4j.linalg.dataset.api.DataSet toPreProcess) {
            toPreProcess.getFeatures().divi(255);  //[0,255] -> [0,1] 입력픽셀값
        }
    }
}
