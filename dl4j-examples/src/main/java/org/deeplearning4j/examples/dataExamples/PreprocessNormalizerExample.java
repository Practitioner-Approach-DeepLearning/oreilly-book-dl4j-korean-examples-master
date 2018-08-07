package org.deeplearning4j.examples.dataExamples;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 이 기본적인 예제는 전처리기를 사용하는 방법을 보여준다.
 * 이 예제는 minmax scaler를 사용하며 3.10 릴리스 이후 버전에서 작동된다.
 * 현재 마스터 버전 및 추후 릴리스 버전은 다른 모든 전처리를 작동 가능하다.
 * 6/8/16에 susaneraly가 생성.
 */
public class PreprocessNormalizerExample {

    private static Logger log = LoggerFactory.getLogger(PreprocessNormalizerExample.class);

    public static void main(String[] args) throws  Exception {


        //========= CSV로 저장된 붓꽃 데이터셋으로부터 데이터셋과 데이터셋 반복자 생성  =============
        //                               자세한 내용은 CSV 예제 참조
        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        RecordReader recordReaderA = new CSVRecordReader(numLinesToSkip,delimiter);
        RecordReader recordReaderB = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        recordReaderA.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        recordReaderB.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        int labelIndex = 4;
        int numClasses = 3;
        DataSetIterator iteratorA = new RecordReaderDataSetIterator(recordReaderA,10,labelIndex,numClasses);
        DataSetIterator iteratorB = new RecordReaderDataSetIterator(recordReaderB,10,labelIndex,numClasses);
        DataSetIterator fulliterator = new RecordReaderDataSetIterator(recordReader,150,labelIndex,numClasses);
        DataSet datasetX = fulliterator.next();
        DataSet datasetY = datasetX.copy();

        // 이제 데이터셋 X, 데이터셋 Y, 반복자 A, 반복자 B에 모든 붓꽃 데이터셋이 로드되었다.
        // 반복자 A와 반복자 B는 배치 크기가 10이다. 따라서 전체 데이터셋은 150 / 10 = 15 배치다.
        //=====================================================================================================================

        log.info("All preprocessors have to be fit to the intended metrics before they can be used to transform");
        log.info("To have a transformation occur when next on an iterator is called use the 'setpreprocessor', example at the very end here\n");
        log.info("This example demonstrates preprocessor use with the min max normalizer.");
        log.info("A standardizing preprocessor is also available.");
        log.info("Usage for all preprocessors are the same - fit then transform a dataset or set as preprocessor to an iterator");

        log.info("Instantiating a preprocessor...\n");
        NormalizerMinMaxScaler preProcessor = new NormalizerMinMaxScaler();
        log.info("During 'fit' the preprocessor calculates the metrics (std dev and mean for the standardizer, min and max for minmaxscaler) from the data given");
        log.info("Fit can take a dataset or a dataset iterator\n");

        // 데이터셋에 전처리기 피팅
        log.info("Fitting with a dataset...............");
        preProcessor.fit(datasetX);
        log.info("Calculated metrics");
        log.info("Min: {}",preProcessor.getMin());
        log.info("Max: {}",preProcessor.getMax());

        log.info("Once fit the preprocessor can be used to transform data wrt to the metrics of the dataset it was fit to");
        log.info("Transform takes a dataset and modifies it in place");

        log.info("Transforming a dataset, printing only the first ten.....");
        preProcessor.transform(datasetX);
        log.info("\n{}\n",datasetX.getRange(0,9));

        log.info("Transformed datasets can be reverted back as well...");
        log.info("Note the reverting happens in place.");
        log.info("Reverting back the dataset, printing only the first ten.....");
        preProcessor.revert(datasetX);
        log.info("\n{}\n",datasetX.getRange(0,9));

        // 반복자에 전처리기 설정
        log.info("Fitting a preprocessor with iteratorB......");
        NormalizerMinMaxScaler preProcessorIter = new NormalizerMinMaxScaler();
        preProcessorIter.fit(iteratorB);
        log.info("A fitted preprocessor can be set to an iterator so each time next is called the transform step happens automatically");
        log.info("Setting a preprocessor for iteratorA");
        iteratorA.setPreProcessor(preProcessorIter);
        while (iteratorA.hasNext()) {
            log.info("Calling next on iterator A that has a preprocessor on it");
            log.info("\n{}",iteratorA.next());
            log.info("Calling next on iterator B that has no preprocessor on it");
            DataSet firstBatch = iteratorB.next();
            log.info("\n{}",firstBatch);
            log.info("Note the data is different - iteratorA is preprocessed, iteratorB is not");
            log.info("Now using transform on the next datset on iteratorB");
            iteratorB.reset();
            firstBatch = iteratorB.next();
            preProcessorIter.transform(firstBatch);
            log.info("\n{}",firstBatch);
            log.info("Note that this now gives the same results");
            break;
        }

        log.info("If you are using batches and an iterator, set the preprocessor on your iterator to transform data automatically when next is called");
        log.info("Use the .transform function only if you are working with a small dataset and no iterator");

        log.info("MinMax scaler also takes a min-max range to scale to.");
        log.info("Instantiating a new preprocessor and setting it's min-max scale to {-1,1}");
        NormalizerMinMaxScaler preProcessorRange = new NormalizerMinMaxScaler(-1,1);
        log.info("Fitting to dataset");
        preProcessorRange.fit(datasetY);
        log.info("First ten before transforming");
        log.info("\n{}",datasetY.getRange(0,9));
        log.info("First ten after transforming");
        preProcessorRange.transform(datasetY);
        log.info("\n{}",datasetY.getRange(0,9));

    }
}
