package org.deeplearning4j.examples.dataExamples;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.image.*;
import javafx.scene.layout.HBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * JavaFX 애플리케이션은 이미지를 그려서 신경망 학습 과정을 보여준다.
 * (Image 같은) 외부에서 비롯된 데이터를 신경망에 공급하는 방법을 보여준다.
 *
 * 이 예제는 OracleJDK가 필요한 JavaFX를 사용한다. 다른 JDK를 사용하는 경우 이 예제를 주석 처리하라.
 * OpenJDK 및 openjfx에서는 정상적으로 작동하는 것으로 확인됐다.
 *
 * TODO: 샘플이 제대로 종료되지 않음. IDE에서 명시적으로 종료시켜야 함
 *
 * @author 로버트 알테나
 */
public class ImageDrawer extends Application {

    private Image originalImage; // 왼편에 표시될 입력 이미지
    private WritableImage composition; // 신경망에 의해 생성될 출력 이미지
    private MultiLayerNetwork nn; // 신경망
    private DataSet ds; // (초기에 한번) 원본으로부터 생성된 학습 데이터. 학습에 사용됨
    private INDArray xyOut; // 출력 이미지를 계산할 x, y 격자. 한번만 계산해서 계속 사용됨

    /**
     * 현재 그래픽 출력을 업데이트하기 위한 신경망 학습
     */
    private void onCalc(){
        nn.fit(ds);
        drawImage();
        Platform.runLater(this::onCalc);
    }

    @Override
    public void init(){
        originalImage = new Image("/DataExamples/Mona_Lisa.png");

        final int w = (int) originalImage.getWidth();
        final int h = (int) originalImage.getHeight();
        composition = new WritableImage(w, h); // 오른쪽 이미지

        ds = generateDataSet(originalImage);
        nn = createNN();

        // 신경망 입력 x, y 격자는 한번만 계산되면 됨
        int numPoints = h * w;
        xyOut = Nd4j.zeros(numPoints, 2);
        for (int i = 0; i < w; i++) {
            double xp = (double) i / (double) (w - 1);
            for (int j = 0; j < h; j++) {
                int index = i + w * j;
                double yp = (double) j / (double) (h - 1);

                xyOut.put(index, 0, xp); // 입력 2개(x, y)
                xyOut.put(index, 1, yp);
            }
        }
        drawImage();
    }
    /**
     * 기본적인 JavaFX 시작: UI를 구축하고, 표시
     */
    @Override
    public void start(Stage primaryStage) {

        final int w = (int) originalImage.getWidth();
        final int h = (int) originalImage.getHeight();
        final int zoom = 5; // 이미지가 작으므로 보기 편하도록 확대해서 표시

        ImageView iv1 = new ImageView(); // 왼쪽 이미지
        iv1.setImage(originalImage);
        iv1.setFitHeight( zoom* h);
        iv1.setFitWidth(zoom*w);

        ImageView iv2 = new ImageView();
        iv2.setImage(composition);
        iv2.setFitHeight( zoom* h);
        iv2.setFitWidth(zoom*w);

        HBox root = new HBox(); // 화면 구축
        Scene scene = new Scene(root);
        root.getChildren().addAll(iv1, iv2);

        primaryStage.setTitle("Neural Network Drawing Demo.");
        primaryStage.setScene(scene);
        primaryStage.show();

        Platform.setImplicitExit(true);

        // 신경망이 초기화됐을 때 쯤 JavaFX가 시작하도록 허용
        Platform.runLater(this::onCalc);
    }

    public static void main( String[] args )
    {
        launch(args);
    }

    /**
     * 신경망 구축
     */
    private static MultiLayerNetwork createNN() {
        int seed = 2345;
        int iterations = 25; //<-- 반복마다 피팅 호출
        double learningRate = 0.1;
        int numInputs = 2;   // x, y
        int numHiddenNodes = 25;
        int numOutputs = 3 ; // R, G, B 3가지 값

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                .activation(Activation.IDENTITY)
                .build())
            .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                .activation(Activation.RELU)
                .build())
            .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                .activation(Activation.RELU)
                .build())
            .layer(3, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                .activation(Activation.RELU)
                .build())
            .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.L2)
                .activation(Activation.IDENTITY)
                .nIn(numHiddenNodes).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        return net;
    }

    /**
     * DL4J에서 사용할 Javafx 이미지 처리
     *
     * @param img 처리할 Javafx 이미지
     * @return DL4J DataSet
     */
    private static DataSet generateDataSet(Image img) {
        int w = (int) img.getWidth();
        int h = (int) img.getHeight();
        int numPoints = h * w;

        PixelReader reader = img.getPixelReader();

        INDArray xy = Nd4j.zeros(numPoints, 2);
        INDArray out = Nd4j.zeros(numPoints, 3);

        // 가장 간단한 구현
        for (int i = 0; i < w; i++) {
            double xp = (double) i / (double) (w - 1);
            for (int j = 0; j < h; j++) {
                Color c = reader.getColor(i, j);
                int index = i + w * j;
                double yp = (double) j / (double) (h - 1);

                xy.put(index, 0, xp); // 입력 2개(x, y)
                xy.put(index, 1, yp);

                out.put(index, 0, c.getRed());  // RGB 값 출력
                out.put(index, 1, c.getGreen());
                out.put(index, 2, c.getBlue());
            }
        }
        return new DataSet(xy, out);
    }

    /**
     * 신경망으로 이미지를 그림
     */
    private void drawImage() {
        int w = (int) composition.getWidth();
        int h = (int) composition.getHeight();

        INDArray out = nn.output(xyOut);
        PixelWriter writer = composition.getPixelWriter();

        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                int index = i + w * j;
                double red = capNNOutput(out.getDouble(index, 0));
                double green = capNNOutput(out.getDouble(index, 1));
                double blue = capNNOutput(out.getDouble(index, 2));

                Color c = new Color(red, green, blue, 1.0);
                writer.setColor(i, j, c);
            }
        }
    }

    /**
     * 색 수치를 0과 1사이의 값으로 만듬
     */
    private static double capNNOutput(double x) {
        double tmp = (x<0.0) ? 0.0 : x;
        return (tmp > 1.0) ? 1.0 : tmp;
    }
}
