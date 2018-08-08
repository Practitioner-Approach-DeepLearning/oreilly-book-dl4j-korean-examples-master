package org.deeplearning4j.examples.misc.customlayers;

import org.deeplearning4j.examples.misc.customlayers.layer.CustomLayer;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * 커스텀 계층 예제. 이 예제는 커스텀 계층 구현에 대한 사용 및 기본 테스트를 한다.
 * 자세한 내용은 CustomLayerExampleReadme.md 참고
 *
 * @author Alex Black
 */


public class CustomLayerExample {

    static{
        // 기울기 검사를 위한 Double 타입 정밀도 계산. doGradientCheck () 메소드의 주석보기
        // 혹은  http://nd4j.org/userguide.html#miscdatatype를 참고하자
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    public static void main(String[] args) throws IOException {
        runInitialTests();
        doGradientCheck();
    }

    private static void runInitialTests() throws IOException {
        /*
        This method shows the configuration and use of the custom layer.
        It also shows some basic sanity checks and tests for the layer.
        In practice, these tests should be implemented as unit tests; for simplicity, we are just printing the results
         */
        /*
        이 메소드는 커스텀 계층의 설정과 사용에 대해 보여준다.
        또한 계층에 대한 몇 가지 기본 안정성 검사 및 테스트를 한다.
        실제로는 이런 테스트의 경우 단위 테스트로 구현되어야 한다. 이것은 예제로서 단순히 결과를 보여주기위해 출력할 뿐이다.
         */

        System.out.println("----- Starting Initial Tests -----");

        int nIn = 5;
        int nOut = 8;

        // 커스텀 계층을 사용하여 신경망을 만들어보자
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .updater(Updater.RMSPROP).rmsDecay(0.95)
            .weightInit(WeightInit.XAVIER)
            .regularization(true).l2(0.03)
            .list()
            .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(nIn).nOut(6).build())     //기본적인 DenseLayer
            .layer(1, new CustomLayer.Builder()
                .activation(Activation.TANH)                                                    //FeedForwardLayer로 부터 프로퍼티 계승
                .secondActivationFunction(Activation.SIGMOID)                                   //계층을 위한 사용자정의 프로퍼티
                .nIn(6).nOut(7)                                                                 //nIn 과 nOut 역시 FeedForwardLayer로부터 승계됨
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)                //기본적인 OutputLayer
                .activation(Activation.SOFTMAX).nIn(7).nOut(nOut).build())
            .pretrain(false).backprop(true).build();


        // 첫번째 : 구성에 대한 몇 가지 기본 안정성 검사를 실행
        double customLayerL2 = config.getConf(1).getLayer().getL2();
        System.out.println("l2 coefficient for custom layer: " + customLayerL2);                //예상치:  global L2 파라미터 설정을 커스텀 계층이 계승한다.
        Updater customLayerUpdater = config.getConf(1).getLayer().getUpdater();
        System.out.println("Updater for custom layer: " + customLayerUpdater);                  //예상치: global Updater 설정을 커스텀 계층이 계승한다.

        // 두번째: JSON 및 YAML 구성이 커스텀 계층과 함께 작동하는지 확인한다.
        // 직렬화에 문제가 있는 경우 직렬화 해제 중에 예외가 발생한다. ("No suitable constructor found...")

        String configAsJson = config.toJson();
        String configAsYaml = config.toYaml();
        MultiLayerConfiguration fromJson = MultiLayerConfiguration.fromJson(configAsJson);
        MultiLayerConfiguration fromYaml = MultiLayerConfiguration.fromYaml(configAsYaml);

        System.out.println("JSON configuration works: " + config.equals(fromJson));
        System.out.println("YAML configuration works: " + config.equals(fromYaml));

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();


        // 세번째: 기본적인 테스트를 해보자, 먼저 전방향 및 역방향 전달 메소드가 예외를 던지지 않았는지 확인한다.
        // 이를 위해 간단한 테스트 데이터를 생성한다.
        int minibatchSize = 5;
        INDArray testFeatures = Nd4j.rand(minibatchSize, nIn);
        INDArray testLabels = Nd4j.zeros(minibatchSize, nOut);
        Random r = new Random(12345);
        for( int i=0; i<minibatchSize; i++ ){
            testLabels.putScalar(i,r.nextInt(nOut),1);  //무작위 원-핫 라벨 데이터
        }

        List<INDArray> activations = net.feedForward(testFeatures);
        INDArray activationsCustomLayer = activations.get(2);                                    // 활성화 인덱스 2, 입력 0, 첫번쨰 계층 1 등등..
        System.out.println("\nActivations from custom layer:");
        System.out.println(activationsCustomLayer);
        net.fit(new DataSet(testFeatures, testLabels));

        // 최종적으로 ModelSerializer를 이용해서 모델 직렬화 과정을 확인해보자.
        ModelSerializer.writeModel(net, new File("CustomLayerModel.zip"), true);
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(new File("CustomLayerModel.zip"));

        System.out.println();
        System.out.println("Original and restored networks: configs are equal: " + net.getLayerWiseConfigurations().equals(restored.getLayerWiseConfigurations()));
        System.out.println("Original and restored networks: parameters are equal: " + net.params().equals(restored.params()));
    }


    private static void doGradientCheck(){
        /*
        기울기를 체크하는 것은 계층을 구현하는데 중요한 요소중에 하나이다.
        이것이 현재 구현된 계층이 신뢰할만하다는 것을 증명하기 때문이다. 이게 없다면 알 수 없는 에러들이 발생할 수도 있다.

        Deeplearning4j comes with a gradient check utility that you can use to check your layers.
        This utility works for feed-forward layers, CNNs, RNNs etc.
        Deeplearning4j는 계층을 확인해볼 수 있도록 기울기 체크 유틸을 제공한다. 이 유틸은 피드포워드 계층 및 CNN, RNN 등에서 모두 사용 가능하다.
        이것에 대한 좀 더 자세한 내용을 알고싶다면 GradientCheckUtil 클래스에 대한 문서를 확인해 보자.
        https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/gradientcheck/GradientCheckUtil.java

        기울기 체크할 때, 몇가지 주의할 점이 있다.
        1. ND4J에는 double 정밀도를 사용해야 한다. 단 정밀도 (부동 소수점 - 기본값)가 기울기 체크를 안정적으로 수행하기에 충분히 정확하지 않다.
        2. 업데이터를 없음으로 설정하거나 확률 경사 하강법을 사용하는 업데이터와 학습 속도 1.0을 모두 동일하게 사용해야 한다.
            이유 : 학습률, 모멘텀 등으로 수정되기 전에 원시 기울기를 테스트하고 있기 때문이다.
        */

        System.out.println("\n\n\n----- Starting Gradient Check -----");

        Nd4j.getRandom().setSeed(12345);
        int nIn = 3;
        int nOut = 2;

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .updater(Updater.NONE).learningRate(1.0)
            .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))              // 보통보다 큰 가중치 초기화는 기울기 체크에 도움이 된다
            .regularization(true).l2(0.03)
            .list()
            .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(nIn).nOut(3).build())    //기본적인 DenseLayer
            .layer(1, new CustomLayer.Builder()
                .activation(Activation.TANH)                                                    //FeedForwardLayer로 부터 프로퍼티 계승
                .secondActivationFunction(Activation.SIGMOID)                                   //계층을 위한 사용자정의 프로퍼티
                .nIn(3).nOut(3)                                                                 //nIn 과 nOut 역시 FeedForwardLayer로부터 승계됨
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)                //기본적인 OutputLayer
                .activation(Activation.SOFTMAX).nIn(3).nOut(nOut).build())
            .pretrain(false).backprop(true).build();
        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();

        boolean print = true;                                                                   //테스트중에 각 파라미터의 상태를 출력할지 여부
        boolean return_on_first_failure = false;                                                //true인 경우 첫번째 테스트가 실패했을 때 테스트 종료
        double gradient_check_epsilon = 1e-8;                                                   //기울기 체크에 적용된 엡실론 값
        double max_relative_error = 1e-5;                                                       //각 파라미터에 허용되는 최대 상대 오차
        double min_absolute_error = 1e-10;                                                      //최소 에러 숫자

        //기울기 검사에 사용할 임의의 입력 데이터 만들기
        int minibatchSize = 5;
        INDArray features = Nd4j.rand(minibatchSize, nIn);
        INDArray labels = Nd4j.zeros(minibatchSize, nOut);
        Random r = new Random(12345);
        for( int i=0; i<minibatchSize; i++ ){
            labels.putScalar(i,r.nextInt(nOut),1);  //무작위 원-핫 라벨 데이터
        }

        // 각 계층의 파라미터 수를 출력하자. 이렇게 하면 실패한 파라미터가 속한 계층을 식별하는데 도움이 된다.
        for( int i=0; i<3; i++ ){
            System.out.println("# params, layer " + i + ":\t" + net.getLayer(i).numParams());
        }

        GradientCheckUtil.checkGradients(net, gradient_check_epsilon, max_relative_error, min_absolute_error, print,
            return_on_first_failure, features, labels);
    }

}
