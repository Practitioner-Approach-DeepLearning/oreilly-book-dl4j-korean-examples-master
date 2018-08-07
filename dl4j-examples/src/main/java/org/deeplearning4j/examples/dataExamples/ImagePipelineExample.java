package org.deeplearning4j.examples.dataExamples;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

/**
 * 6/9/16에 susaneraly가 생성
 */
public class ImagePipelineExample {

    protected static final Logger log = LoggerFactory.getLogger(ImagePipelineExample.class);

    // 이미지는 allowedExtension에 있는 포멧만 사용 가능
    protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    protected static final long seed = 12345;

    public static final Random randNumGen = new Random(seed);

    protected static int height = 50;
    protected static int width = 50;
    protected static int channels = 3;
    protected static int numExamples = 80;
    protected static int outputNum = 4;

    public static void main(String[] args) throws Exception {

        // 디렉토리 구조:
        // 데이터셋의 이미지는 클레스/레이블에 따라 디렉토리가 구성되야 함
        // 이 예제에서는 3가지 클래스로 구성된 이미지 10개를 사용
        // 디렉토리 구조는 다음과 같음
        //                                    부모 디렉토리
        //                                  /    |     \
        //                                 /     |      \
        //                            레이블 A  레이블 B   레이블 C
        //
        // 데이터는 레이블에 따라 각 레이블/클래스 디렉토리에 위치
        // 레이블/클래스 디렉토리는 모두 같은 부모 디렉토리에 위치
        //
        //
        File parentDir = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/DataExamples/ImagePipeline/");
        // 허용된 확장자를 가진
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        // 레이블을 수동으로 지정하지 않아도 된다. 레이블/클래스는 부모 디렉토리 및 하위 디렉토리 이름을 사용한다.
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        // 밸런스 경로 필터를 사용하면 각 클래스에 대해 최소 / 최대 케이스를 미세하게 조절할 수 있다.
        // 자세한 설명은 자바독을 참조
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

        // 이미지 파일을 학습 데이터와 테스트 데이터로 분할. 학습 데이터 : 테스트 데이터 = 8 : 2
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        // 희망 이미지 크기(높이, 너비)를 갖는 새로운 레코드 리더를 생성
        // 이 예제의 이미지들은 각각 다른 크기를 가짐
        // 이미지들은 모두 같은 크기로 조정됨
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);

        // 데이터셋의 크기를 임의로 늘리기 위해 이밎 ㅣ변환을 사용해야 하는 경우도 있음
        // DataVec는 OpenCV의 강력한 기능을 내장
        // 아래와 같이 변환을 연결해 얼굴을 감지하고 자르는 클래스를 작성할 수 있음
        /*ImageTransform transform = new MultiImageTransform(randNumGen,
            new CropImageTransform(10), new FlipImageTransform(),
            new ScaleImageTransform(10), new WarpImageTransform(10));
            */

        // ShowImageTransform를 사용해 이미지를 볼 수 있음
        // 아래 코드는 이전과 이후의 모습을 나란히 보여줌
        ImageTransform transform = new MultiImageTransform(randNumGen,new ShowImageTransform("Display - before "));

        // 학습 데이터와 변환 체인으로 레코드 리더 초기화
        recordReader.initialize(trainData,transform);
        // 레코드 리더를 학습용 반복자로 변환 - 반복자 사용법은 다른 예제를 참고
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum);
        while (dataIter.hasNext()) {
            DataSet ds = dataIter.next();
            System.out.println(ds);
            try {
                Thread.sleep(3000);                 // 1000 밀리초는 1초다
            } catch(InterruptedException ex) {
                Thread.currentThread().interrupt();
            }
        }
        recordReader.reset();

        //transform = new MultiImageTransform(randNumGen,new CropImageTransform(50), new ShowImageTransform("Display - after"));
        //recordReader.initialize(trainData,transform);
        recordReader.initialize(trainData);
        dataIter = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum);
        while (dataIter.hasNext()) {
            DataSet ds = dataIter.next();
        }
        recordReader.reset();

    }
}
