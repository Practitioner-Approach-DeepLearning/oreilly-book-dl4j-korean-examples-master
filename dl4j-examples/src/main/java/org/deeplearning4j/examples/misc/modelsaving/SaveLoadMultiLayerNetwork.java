package org.deeplearning4j.examples.misc.modelsaving;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

/**
 * MultiLayerNetwork를 저장하고 로드하는 매우 쉬운 예제
 *
 * @author Alex Black
 */
public class SaveLoadMultiLayerNetwork {

    public static void main(String[] args) throws Exception {
        //간단한 MultiLayerNetwork 정의
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS)
            .learningRate(0.1)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).activation(Activation.TANH).build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(3).nOut(3).build())
            .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();


        //모델 저장
        File locationToSave = new File("MyMultiLayerNetwork.zip");      //신경망을 저장할 위치. 참고 : 파일은 .zip 형식이며 외부에서 열 수 있다..
        boolean saveUpdater = true;                                             //업데이터: 모멘텀, RMSProp, Adagrad 등의 상태, 나중에 신경망을 더 많이 훈련하려면 해당 내용을 저장하자.
        ModelSerializer.writeModel(net, locationToSave, saveUpdater);

        //모델 로드
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(locationToSave);


        System.out.println("Saved and loaded parameters are equal:      " + net.params().equals(restored.params()));
        System.out.println("Saved and loaded configurations are equal:  " + net.getLayerWiseConfigurations().equals(restored.getLayerWiseConfigurations()));
    }

}
