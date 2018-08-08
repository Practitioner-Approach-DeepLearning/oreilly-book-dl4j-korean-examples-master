package org.deeplearning4j.examples.misc.modelsaving;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

/**
 * ComputationGraph를 저장하고 로드할 수 있는 가장 쉬운 예제
 *
 * @author Alex Black
 */

public class SaveLoadComputationGraph {

    public static void main(String[] args) throws Exception {
        //간단한 ComputationGraph 정의
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS)
            .learningRate(0.1)
            .graphBuilder()
            .addInputs("in")
            .addLayer("layer0", new DenseLayer.Builder().nIn(4).nOut(3).activation(Activation.TANH).build(), "in")
            .addLayer("layer1", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(3).nOut(3).build(), "layer0")
            .setOutputs("layer1")
            .backprop(true).pretrain(false).build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        // 모델 저장
        File locationToSave = new File("MyComputationGraph.zip");       //신경망을 저장할 위치. 참고 : 파일은 .zip 형식이며 외부에서 열 수 있다..
        boolean saveUpdater = true;                                             //업데이터: 모멘텀, RMSProp, Adagrad 등의 상태, 나중에 신경망을 더 많이 훈련하려면 해당 내용을 저장하자.
        ModelSerializer.writeModel(net, locationToSave, saveUpdater);

        // 모델 로드
        ComputationGraph restored = ModelSerializer.restoreComputationGraph(locationToSave);


        System.out.println("Saved and loaded parameters are equal:      " + net.params().equals(restored.params()));
        System.out.println("Saved and loaded configurations are equal:  " + net.getConfiguration().equals(restored.getConfiguration()));
    }

}
