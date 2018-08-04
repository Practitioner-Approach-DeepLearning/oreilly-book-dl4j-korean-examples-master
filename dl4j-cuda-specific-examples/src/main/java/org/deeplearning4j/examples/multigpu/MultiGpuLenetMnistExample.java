package org.deeplearning4j.examples.multigpu;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * 이 예제는 LenetMnistExample의 수정 버전으로 다중 GPU 환경과 호환된다.
 *
 * @author  @agibsonccc
 * @author raver119@gmail.com
 */
public class MultiGpuLenetMnistExample {
    private static final Logger log = LoggerFactory.getLogger(MultiGpuLenetMnistExample.class);

    public static void main(String[] args) throws Exception {
        // 필독: CUDA FP 16 정밀도를 지원한다.
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);

        // 백엔드 초기화를 할 수 있는 임시적인 방법
        Nd4j.create(1);

        CudaEnvironment.getInstance().getConfiguration()
            // 키 옵션 사용
            .allowMultiGPU(true)

            // 메모리 캐시 증가
            .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)

            // 장치 간 직접 접근 기능을 사옹하면 PCIe를 거쳐갈 때보다 모델 평균화가 빠르다.
            .allowCrossDeviceAccess(true);

        int nChannels = 1;
        int outputNum = 10;

        // GPU를 사용할 때는 배치 크기를 더 크게 잡으면 좋다.
        int batchSize = 128;
        int nEpochs = 10;
        int iterations = 1;
        int seed = 123;

        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations) // 학습 반복 횟수 설정
            .regularization(true).l2(0.0005)
                /*
                    학습 감쇠 및 편향을 적용하려면 아래 주석을 해제할 것
                 */
            .learningRate(.01)//.biasLearningRate(0.02)
            //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, new ConvolutionLayer.Builder(5, 5)
                // nIn과 nOut으로 깊이 지정. nIn은 채널 수고 nOut은 필터 수다.
                .nIn(nChannels)
                .stride(1, 1)
                .nOut(20)
                .activation(Activation.IDENTITY)
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .build())
            .layer(2, new ConvolutionLayer.Builder(5, 5)
                // 이 계층부터는 nIn을 설정할 필요가 없음
                .stride(1, 1)
                .nOut(50)
                .activation(Activation.IDENTITY)
                .build())
            .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2,2)
                .stride(2,2)
                .build())
            .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                .nOut(500).build())
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutionalFlat(28,28,1)) // 아래 주석 참조
            .backprop(true).pretrain(false).build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // ParallelWrapper는 GPU 간 로드밸런싱을 처리함
        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
            // DataSets 선적재 옵션. 실제 장치 개수에 따라 값을 설정
            .prefetchBuffer(24)

            // 사용 가능한 장치 개수 대비 같거나 많게 설정. x1-x2로 시작하면 좋다.
            .workers(4)

            // 평균화 빈도가 줄어들면 성능은 향상되지만 모델 정확도는 감소할 수 있음
            .averagingFrequency(3)

            // true로 설정하면 모든 평균 모델 점수가 보고됨
            .reportScoreAfterAveraging(true)

            // 선택 옵션, 시스템이 PCIe에서 P2P 메모리 액세스를 지원하는 경우에만 false로 설정(참고: AWS는 P2P를 지원하지 않음)
            .useLegacyAveraging(true)

            .build();

        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(100));
        long timeX = System.currentTimeMillis();

        // 반복자를 수동으로 반복/재설정하는 대신 MultipleEpochsIterator를 사용할 수도 있음
        //MultipleEpochsIterator mnistMultiEpochIterator = new MultipleEpochsIterator(nEpochs, mnistTrain);

        for( int i=0; i<nEpochs; i++ ) {
            long time1 = System.currentTimeMillis();

            // 필독: ParallelWrapper에 모델을 직접 전달하지 않고 iterator를 전달할 수도 있음
//            wrapper.fit(mnistMultiEpochIterator);
            wrapper.fit(mnistTrain);
            long time2 = System.currentTimeMillis();
            log.info("*** Completed epoch {}, time: {} ***", i, (time2 - time1));
        }
        long timeY = System.currentTimeMillis();

        log.info("*** Training complete, time: {} ***", (timeY - timeX));

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        while(mnistTest.hasNext()){
            DataSet ds = mnistTest.next();
            INDArray output = model.output(ds.getFeatureMatrix(), false);
            eval.eval(ds.getLabels(), output);
        }
        log.info(eval.stats());
        mnistTest.reset();

        log.info("****************Example finished********************");
    }
}
