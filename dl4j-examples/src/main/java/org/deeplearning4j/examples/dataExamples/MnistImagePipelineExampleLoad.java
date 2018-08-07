package org.deeplearning4j.examples.dataExamples;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.FilenameUtils;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Random;

/**
 * /**
 *  This code example is featured in this youtube video
 *
 *  https://www.youtube.com/watch?v=zrTSs715Ylo
 *
 * 비디오 예제와 다른 점은
 * 비디오 예제는 이미 데이터가 다운로드되어 있지만
 * 이 에제는 데이터를 다운로드하는 코드도 포함되어 있다는 점이다.
 *
 *  데이터는 아래 명령어로 다운로드할 수 있다.
 *
 *
 *  wget http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
 *  또한 아래 명령어로 압축을 풀 수 있다.
 *  tar xzvf mnist_png.tar.gz
 *
 *

 *  이 예제는 MnistImagePipelineExample 예제를 기반으로 만들어졌으며
 *  이전에 저장한 신경망을 로드하는 부분이 추가됐다.
 */
public class MnistImagePipelineExampleLoad {

    /** 데이터를 다운로드할 URL */
    public static final String DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

    /** 추출한 training/testing를 저장할 경로 */
    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/");

    private static Logger log = LoggerFactory.getLogger(MnistImagePipelineExampleLoad.class);

    public static void main(String[] args) throws Exception {
        // 이미지 정보
        // 28 * 28 그레이스케일
        // 그레이스케일은 단일 채널을 의미함
        int height = 28;
        int width = 28;
        int channels = 1;
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);
        int batchSize = 128;
        int outputNum = 10;
        int numEpochs = 15;

         /*
        이 클래스의 downloadData()언 데이터를 다운로드하고
        자바의 tmpdir에 저장한다.
        15MB 짜리 압축 파일을 다운로드하고
        압축을 풀려면 158MB 공간이 필요하다.
        데이터는 여기에서 수동으로 다운로드할 수 있다.
        http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
         */


        downloadData();

        // 파일 경로 정의
        File trainData = new File(DATA_PATH + "/mnist_png/training");
        File testData = new File(DATA_PATH + "/mnist_png/testing");


        // FileSplit(경로, 허용 확장자, 랜덤값) 정의

        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS,randNumGen);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS,randNumGen);

        // 상위 경로에서 이미지 레이블을 추출

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);

        // 레코드 리더 초기화
        // 이름을 추출하기 위해 리스너 추가

        recordReader.initialize(train);
        //recordReader.setListeners(new LogRecordListener());

        // DataSet 반복자

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);

        // 픽셀 값을 0~1 사이로 변경

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);


        // 신경망 구축


        log.info("******LOAD TRAINED MODEL******");
        // 상세

        // MnistImagePipelineSave를 실행했다면
        // 저장된 모델을 불러올 수 있다
        File locationToSave = new File("trained_mnist_model.zip");

        if(locationToSave.exists()){
            System.out.println("\n######Saved Model Found######\n");
        }else{
            System.out.println("\n\n#######File not found!#######");
            System.out.println("This example depends on running ");
            System.out.println("MnistImagePipelineExampleSave");
            System.out.println("Run that Example First");
            System.out.println("#############################\n\n");


            System.exit(0);
        }







        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);


        model.getLabels();


        // 테스트 데이터로 로드된 모델을 테스트

        recordReader.initialize(test);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);

        // 클래스 10개로 Evaluation 객체 생성
        Evaluation eval = new Evaluation(outputNum);




        while(testIter.hasNext()){
            DataSet next = testIter.next();
            INDArray output = model.output(next.getFeatures());
            eval.eval(next.getLabels(),output);

        }

        log.info(eval.stats());


    }

      /*
    아래 내용은 레코드 리더, DataVec, 신경망과 아무 관련이 없다.
    downloadData, getMnistPNG(), extractTarGz는 데이터를 다운로드하고 추추출하기 위한 메서드다.
     */

    private static void downloadData() throws Exception {
        // 필요 시 디렉토리 생성
        File directory = new File(DATA_PATH);
        if(!directory.exists()) directory.mkdir();

        // 파일 다운로드
        String archizePath = DATA_PATH + "/mnist_png.tar.gz";
        File archiveFile = new File(archizePath);
        String extractedPath = DATA_PATH + "mnist_png";
        File extractedFile = new File(extractedPath);

        if( !archiveFile.exists() ){
            System.out.println("Starting data download (15MB)...");
            getMnistPNG();
            // 출력 디렉토리에 tar.gz 파일 추출
            extractTarGz(archizePath, DATA_PATH);
        } else {
            // 아카이브(.tar.gz)가 있다면 데이터도 이미 추출되었다고 가정
            System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
            if( !extractedFile.exists()){
                // 출력 디렉토리에 tar.gz 파일 추출
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

            /** getNextEntry 메서드를 사용해 tar 항목 읽기 **/
            while ((entry = (TarArchiveEntry) tais.getNextEntry()) != null) {
                //System.out.println("Extracting file: " + entry.getName());

                // 필요 시 디렉토리 생성
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

    public static void getMnistPNG() throws IOException {
        String tmpDirStr = System.getProperty("java.io.tmpdir");
        String archizePath = DATA_PATH + "/mnist_png.tar.gz";

        if (tmpDirStr == null) {
            throw new IOException("System property 'java.io.tmpdir' does specify a tmp dir");
        }
        String url = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";
        File f = new File(archizePath);
        File dir = new File(tmpDirStr);
        if (!f.exists()) {
            HttpClientBuilder builder = HttpClientBuilder.create();
            CloseableHttpClient client = builder.build();
            try (CloseableHttpResponse response = client.execute(new HttpGet(url))) {
                HttpEntity entity = response.getEntity();
                if (entity != null) {
                    try (FileOutputStream outstream = new FileOutputStream(f)) {
                        entity.writeTo(outstream);
                        outstream.flush();
                        outstream.close();
                    }
                }

            }
            System.out.println("Data downloaded to " + f.getAbsolutePath());
        } else {
            System.out.println("Using existing directory at " + f.getAbsolutePath());
        }

    }


}
