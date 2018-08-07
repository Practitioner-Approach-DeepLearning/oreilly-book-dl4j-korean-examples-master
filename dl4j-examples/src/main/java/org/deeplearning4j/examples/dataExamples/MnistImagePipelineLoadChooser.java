package org.deeplearning4j.examples.dataExamples;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * /**
 * 이 예제에 대한 설명은 유튜브에서도 확인할 수 있다.
 *
 *  http://www.youtube.com/watch?v=DRHIpeJpJDI
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
 * 데이터 디렉토리 mnist_png는 training, testing 하위 디렉토리를 가지고 있다.
 * training, testing 디렉토리는 0~9 하위 디렉토리를 가지고 있다.
 * 각 디렉토리에는 손글씨 이미지 28 * 28 PNG가 포함되어 있다.
 *
 *
 *
 *
 *
 *  이 예제는 MnistImagePipelineExample 예제를 기반으로 만들어졌으며
 *  사용자가 이미지를 골라 신경망을 테스트해볼 수 있도록 파일 선택기가 추가됐다.
 *  사용자는 무엇이든 테스트할 수 있지만, 신경망이 검은 바탕 흰글씨로 작성된 0~9 손글씨로 학습된 상태이므로 설계대로 잘 동작할 것이다.
 *
 */
public class MnistImagePipelineLoadChooser {
    private static Logger log = LoggerFactory.getLogger(MnistImagePipelineLoadChooser.class);


    /*
    훈련된 신경망에 대해 테스트할 이미지 파일을 선택할 수 있는 팝업 창을 만든다.
    선택한 이미지는 자동으로 28 x 28 그레이 스케일로 조정된다.
     */
    public static String fileChose(){
        JFileChooser fc = new JFileChooser();
        int ret = fc.showOpenDialog(null);
        if (ret == JFileChooser.APPROVE_OPTION)
        {
            File file = fc.getSelectedFile();
            String filename = file.getAbsolutePath();
            return filename;
        }
        else {
            return null;
        }
    }

    public static void main(String[] args) throws Exception{
        int height = 28;
        int width = 28;
        int channels = 1;

        // recordReader.getLabels()
        // 이 버전에서는 라벨이 항상 순서대로 되어 있으므로
        // 이 부분은 불필요하다.
        //List<Integer> labelList = Arrays.asList(2,3,7,1,6,4,0,5,8,9);
        List<Integer> labelList = Arrays.asList(0,1,2,3,4,5,6,7,8,9);

        // 파일 선택기를 팝업으로 띄운다.
        String filechose = fileChose().toString();

        // 신경망 불러오기

        // 모델이 저장된 위치
        File locationToSave = new File("trained_mnist_model.zip");
        // 모델 저장 여부 확인
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

        log.info("*********TEST YOUR IMAGE AGAINST SAVED NETWORK********");

        // FileChose는 선택한 파일 경로 문자열이다

        File file = new File(filechose);

        // NativeImageLoader를 사용해 숫자 행렬로 변환

        NativeImageLoader loader = new NativeImageLoader(height, width, channels);

        // 이미지를 INDarray로 가져온다.

        INDArray image = loader.asMatrix(file);

        // 0-255
        // 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);
        // 신경망을 통과

        INDArray output = model.output(image);

        log.info("## The FILE CHOSEN WAS " + filechose);
        log.info("## The Neural Nets Pediction ##");
        log.info("## list of probabilities per label ##");
        //log.info("## List of Labels in Order## ");
        // 이 버전에서는 레이블이 항상 순서대로 지정된다.
        log.info(output.toString());
        log.info(labelList.toString());

    }



}
