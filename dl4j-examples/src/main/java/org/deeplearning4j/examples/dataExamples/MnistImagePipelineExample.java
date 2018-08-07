package org.deeplearning4j.examples.dataExamples;



import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.URL;
import java.util.Random;

/**
 * 11/7/16에 tom hanlon가 생성.
 *
 * 이 예제에 대한 설명은 유튜브에서도 확인할 수 있다.
 * https://www.youtube.com/watch?v=GLC8CIoHDnI
 *
 * 비디오 예제와 다른 점은
 * 비디오 예제는 이미 데이터가 다운로드되어 있지만
 * 이 에제는 데이터를 다운로드하는 코드도 포함되어 있다는 점이다.
 *
 * 설명
 * testing 및 training 폴더를 포함한 디렉토리를 다운로드한다.
 * 각 폴더는 0부터 9까지 디렉토리 10개를 가지고 있다.
 * 각 디렉토리는 손글씨 숫자를 28 x 그레이 스케일의 png 파일들을 가지고 있다.
 *
 * 이 코드는 ParentPathLabelGenerator를 사용해 이미지가 레코드 리더로 읽혀질 때 레이블을 지정하는 방법을 보여준다.
 *
 * ImagePreProcessingScaler 픽셀 값은 0과 1 사이의 값으로 변경된다.
 *
 *  이 예제에서는 루프 스탭마다 이미지 3개를 처리하고 터미널에 DataSet을 출력한다.
 *  출력 내용은 28 x 28 행렬로 표현된 0~1 픽셀 값 리스트와 해당 이미지의 레이블 및 레이블 값 리스트다.
 *
 *  또한 이 예제는 레코드 리더에 읽은 이미지의 경로를 기록하는 리스너를 적용한다.
 *  보통 상용 환경에서는 리스너를 적용하지 않는다.
 *  리스너를 적용하면 (예를 들어) 손글씨 숫자 3은 디렉토리 3에서 읽었고,
 *  픽셀 값 행렬을 확인할 수 있으며,
 *  레이블 값이 3인 것을 확인할 수 있다.
 *
 */

public class MnistImagePipelineExample {

    /** 데이터를 다운로드할 URL */
    public static final String DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

    /** 추출한 training/testing를 저장할 경로 */
    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/");



    private static Logger log = LoggerFactory.getLogger(MnistImagePipelineExample.class);

    public static void main(String[] args) throws Exception {
        /*
        이미지 정보
        28 * 28 그레이스케일
        그레이스케일은 단일 채널을 의미함
        */
        int height = 28;
        int width = 28;
        int channels = 1;
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);
        int batchSize = 1;
        int outputNum = 10;



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


        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        // 상위 경로에서 이미지 레이블을 추출

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        // 레코드 리더 초기화
        // 이름을 추출하기 위해 리스너 추가

        recordReader.initialize(train);

        // LogRecordListener는 각 이미지를 읽을 때 경로를 로그로 남긴다.
        // 이 로그는 정보 확인용으로 사용된다.
        // 전체 데이터셋을 처리하면 60,000개의 로그가 출력된다.
        // 로그는 아래와 같은 형식으로 출력된다.
        // o.d.a.r.l.i.LogRecordListener - Reading /tmp/mnist_png/training/4/36384.png

        recordReader.setListeners(new LogRecordListener());

        // DataSet 반복자

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

        // 픽셀 값을 0~1 사이로 변경

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        // 상용 환경에서는 모든 데이터를 순회하지만
        // 이 예제에서는 데모 목적으로 이미지 3개만 순회한다.
        for (int i = 1; i < 3; i++) {
            DataSet ds = dataIter.next();
            System.out.println(ds);
            System.out.println(dataIter.getLabels());

        }

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


