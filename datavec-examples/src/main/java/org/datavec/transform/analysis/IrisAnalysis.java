package org.datavec.transform.analysis;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.DataAction;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.analysis.columns.DoubleAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.ui.HtmlAnalysis;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.AnalyzeSpark;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.utils.SparkUtils;

import java.io.File;
import java.util.List;

/**
 * .html file. 독립 실행 형 .html 파일로 아이리스 데이터 집합에 대한 기본 분석을 수행하고 내 보낸다. 이 기능은 기본이지만
 * 분석 및 디버깅에 여전히 유용 할 수 있다.
 *
 * @author Alex Black
 */
public class IrisAnalysis {

    public static void main(String[] args) throws Exception {

        Schema schema = new Schema.Builder()
                .addColumnsDouble("Sepal length", "Sepal width", "Petal length", "Petal width")
                .addColumnInteger("Species").build();

        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DataVec Example");

        JavaSparkContext sc = new JavaSparkContext(conf);

        String directory = new ClassPathResource("IrisData/iris.txt").getFile().getParent(); // 일반적으로 파일패스는 "file:/..."
                                                                                             // or "hdfs:/..." 이다.
        JavaRDD<String> stringData = sc.textFile(directory);

        // 첫번째로 CSV포맷을 파싱한다; 이것을 CSVRecordReader라는 것으로 사용한다.
        RecordReader rr = new CSVRecordReader();
        JavaRDD<List<Writable>> parsedInputData = stringData.map(new StringToWritablesFunction(rr));

        int maxHistogramBuckets = 10;
        DataAnalysis dataAnalysis = AnalyzeSpark.analyze(schema, parsedInputData, maxHistogramBuckets);

        System.out.println(dataAnalysis);

        // 열 단위의 값을 얻을 수 있다.
        DoubleAnalysis da = (DoubleAnalysis) dataAnalysis.getColumnAnalysis("Sepal length");
        double minValue = da.getMin();
        double maxValue = da.getMax();
        double mean = da.getMean();

        HtmlAnalysis.createHtmlAnalysisFile(dataAnalysis, new File("DataVecIrisAnalysis.html"));

        // HDFS에 쓰기:
        // String htmlAnalysisFileContents =
        // HtmlAnalysis.createHtmlAnalysisString(dataAnalysis);
        // SparkUtils.writeStringToFile("hdfs://your/hdfs/path/here",htmlAnalysisFileContents,sc);
    }

}
