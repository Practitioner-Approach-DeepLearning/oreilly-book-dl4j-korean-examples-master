package org.deeplearning4j.examples.dataExamples;


import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;



/**
 * 이 예제는 동물 분류를 위한 테스트 데이터에서 학습 데이터를 분할하는 간단한 CSV 분류 예제이다.
 * CSV 데이터를 네트워크에 로드할 뿐만 아니라 데이터를 추출하고 분류 결과를 표시하는 방법 및 테스트 데이터에 레이블을 매핑해 결과를 도출하는 간단한 방법을
 * 보여주기 때문에 초보자용 예제로 적합하다.
 *
 * @author 클레이 그레이엄
 */
public class BasicCSVClassifier {

    private static Logger log = LoggerFactory.getLogger(BasicCSVClassifier.class);

    private static Map<Integer,String> eats = readEnumCSV("/DataExamples/animals/eats.csv");
    private static Map<Integer,String> sounds = readEnumCSV("/DataExamples/animals/sounds.csv");
    private static Map<Integer,String> classifiers = readEnumCSV("/DataExamples/animals/classifiers.csv");

    public static void main(String[] args){

        try {

            // 이제 RecordReaderDataSetIterator가 DataSet 객체로의 변환을 처리해 신경망에서 사용할 준비가 되었다.
            int labelIndex = 4;     // 각 행에 값 5개가 있는 iris.txt CSV: 입력 특징 4개 다음에 정수 레이블(클래스) 색인이 위치한다. 레이블은 각 행의 5번째 값(색인 4)이다.
            int numClasses = 3;     // 붓꽃 데이터셋에는 클래스 3개(붓꽃 유형)가 존재한다. 각 클래스 정수 값은 0, 1, 2이다.

            int batchSizeTraining = 30;    // 붓꽃 데이터셋은 총 150개의 입력데이터가 있다. 이를 한 DataSet 객체에 로드할 것이다(데이터셋이 클 때는 추천하지 않는다).
            DataSet trainingData = readCSVDataset(
                    "/DataExamples/animals/animals_train.csv",
                    batchSizeTraining, labelIndex, numClasses);

            // 분류하고자 하는 데이터
            int batchSizeTest = 44;
            DataSet testData = readCSVDataset("/DataExamples/animals/animals.csv",
                    batchSizeTest, labelIndex, numClasses);


            // 정규화는 데이터를 변경하기 때문에 정규화하기 전에 레코드에 대한 데이터 모델을 만든다.
            Map<Integer,Map<String,Object>> animals = makeAnimalsForTesting(testData);


            // 데이터를 정규화하는데 NormalizerStandardize(평균 0, 단위 분산 제공)를 사용한다.
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(trainingData);           // 학습 데이터에서 통계 (평균/표준편차)를 수집. 이 단계에서는 입력 데이터를 수정하지는 않음
            normalizer.transform(trainingData);     // 학습 데이터에 정규화 적용
            normalizer.transform(testData);         // 테스트 데이터에 정규화 적용. 학습 데이터셋에서 계산된 통계를 이용
            final int numInputs = 4;
            int outputNum = 3;
            int iterations = 1000;
            long seed = 6;

            log.info("Build model....");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(iterations)
                    .activation(Activation.TANH)
                    .weightInit(WeightInit.XAVIER)
                    .learningRate(0.1)
                    .regularization(true).l2(1e-4)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3).build())
                    .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build())
                    .backprop(true).pretrain(false)
                    .build();

            // 모델 실행
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(100));

            model.fit(trainingData);

            // 테스트셋으로 모델 평가
            Evaluation eval = new Evaluation(3);
            INDArray output = model.output(testData.getFeatureMatrix());

            eval.eval(testData.getLabels(), output);
            log.info(eval.stats());

            setFittedClassifiers(output, animals);
            logAnimals(animals);

        } catch (Exception e){
            e.printStackTrace();
        }

    }



    public static void logAnimals(Map<Integer,Map<String,Object>> animals){
        for(Map<String,Object> a:animals.values())
            log.info(a.toString());
    }

    public static void setFittedClassifiers(INDArray output, Map<Integer,Map<String,Object>> animals){
        for (int i = 0; i < output.rows() ; i++) {

            // 피팅한 결과로부터 분류를 수행
            animals.get(i).put("classifier",
                    classifiers.get(maxIndex(getFloatArrayFromSlice(output.slice(i)))));

        }

    }


    /**
     * 이 메서드는 INDArray를 부동소수점 배열로 변환하는 방법을 보여준다.
     *
     * @param rowSlice
     * @return
     */
    public static float[] getFloatArrayFromSlice(INDArray rowSlice){
        float[] result = new float[rowSlice.columns()];
        for (int i = 0; i < rowSlice.columns(); i++) {
            result[i] = rowSlice.getFloat(i);
        }
        return result;
    }

    /**
     * 최대 항목 색인을 찾는다. 데이터가 피팅됐을 때 테스트 행에 할당할 클래스를 결정하는데 사용된다.
     *
     * @param vals
     * @return
     */
    public static int maxIndex(float[] vals){
        int maxIndex = 0;
        for (int i = 1; i < vals.length; i++){
            float newnumber = vals[i];
            if ((newnumber > vals[maxIndex])){
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * 로드된 데이터셋으로 레코드 모델을 만들 수 있다.
     * 그러면 피팅된 분류기를 레코드와 연결할 수 있다.
     *
     * @param testData
     * @return
     */
    public static Map<Integer,Map<String,Object>> makeAnimalsForTesting(DataSet testData){
        Map<Integer,Map<String,Object>> animals = new HashMap<>();

        INDArray features = testData.getFeatureMatrix();
        for (int i = 0; i < features.rows() ; i++) {
            INDArray slice = features.slice(i);
            Map<String,Object> animal = new HashMap();

            //속성 설정
            animal.put("yearsLived", slice.getInt(0));
            animal.put("eats", eats.get(slice.getInt(1)));
            animal.put("sounds", sounds.get(slice.getInt(2)));
            animal.put("weight", slice.getFloat(3));

            animals.put(i,animal);
        }
        return animals;

    }


    public static Map<Integer,String> readEnumCSV(String csvFileClasspath) {
        try{
            List<String> lines = IOUtils.readLines(new ClassPathResource(csvFileClasspath).getInputStream());
            Map<Integer,String> enums = new HashMap<>();
            for(String line:lines){
                String[] parts = line.split(",");
                enums.put(Integer.parseInt(parts[0]),parts[1]);
            }
            return enums;
        } catch (Exception e){
            e.printStackTrace();
            return null;
        }

    }

    /**
     * 학습과 테스트에 사용된다.
     *
     * @param csvFileClasspath
     * @param batchSize
     * @param labelIndex
     * @param numClasses
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    private static DataSet readCSVDataset(
            String csvFileClasspath, int batchSize, int labelIndex, int numClasses)
            throws IOException, InterruptedException{

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile()));
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr,batchSize,labelIndex,numClasses);
        return iterator.next();
    }



}
