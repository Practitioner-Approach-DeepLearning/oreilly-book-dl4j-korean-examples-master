package org.datavec.transform.basic;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.condition.ConditionalReplaceValueTransform;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.transform.transform.time.StringToTimeTransform;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.datavec.spark.functions.RecordReaderFunction;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

/**
 * 기본적인 CSV 데이터에 대한 전처리 작업을 다루는 기본적인 DataVec 예제이다. CSV 데이터를 로드해서 학습에 사용하고 싶다면
 * org.deeplearning4j.examples.dataExample.CSVExample을 참고하자
 *
 * 여기서는 트랜잭션과 관련된 일부 데이터를 CSV 형식으로 사용할 수 있다는 것을 전제로하고 있으며 이 데이터에 대해 일부 작업을 수행한다.
 * 
 * 1. 불필요한 열을 제거한다. 2. "USA", "CAN"이 "MerchantCountryCode" 열에 유지되기 위한 필터링 3.
 * "TransactionAmountUSD" 컬럼에서 유효하지 않은 값 대체
 * 
 * 날짜 문자열을 파싱하고 시간을 추출하여 새로운 "HourOfDay"열을 만든다.
 *
 * @author Alex Black
 */
public class BasicDataVecExample {

    public static void main(String[] args) throws Exception {

        // =====================================================================
        // 첫번째 : 입력 데이터 스키마를 정의
        // =====================================================================

        // 가져오려는 데이터의 스키마를 정의해보자
        // 여기에 정의 된 열 순서는 입력 데이터에 나타나는 순서와 일치해야한다.
        Schema inputDataSchema = new Schema.Builder()
                // 단일 열을 정의한다.
                .addColumnString("DateTimeString")
                // 혹은 편의를 위해 같은 유형의 여러 열 정의
                .addColumnsString("CustomerID", "MerchantID")
                // 다른종류의 데이터 타입을 위해 다른 열들을 정의한다.
                .addColumnInteger("NumItemsInTransaction")
                .addColumnCategorical("MerchantCountryCode", Arrays.asList("USA", "CAN", "FR", "MX"))
                // 일부 열에는 허용되는 값에 대한 제한이 있으며 유효하다고 간주된다.
                .addColumnDouble("TransactionAmountUSD", 0.0, null, false, false) // $ 0.0 이상, 최대 제한 없음, NaN 없음, 무한 값 없음
                .addColumnCategorical("FraudLabel", Arrays.asList("Fraud", "Legit")).build();

        // 스키마 출력:
        System.out.println("Input data schema details:");
        System.out.println(inputDataSchema);

        System.out.println("\n\nOther information obtainable from schema:");
        System.out.println("Number of columns: " + inputDataSchema.numColumns());
        System.out.println("Column names: " + inputDataSchema.getColumnNames());
        System.out.println("Column types: " + inputDataSchema.getColumnTypes());

        // =====================================================================
        // 두번쨰 : 원하는 작업 정의하기
        // =====================================================================

        // 데이터에서 실행하는 일부 작업을 정의해보자.
        // TransformProcess를 정의하기 위한 작업
        // At each step, we identify column by the name we gave them in the input data
        // schema, above
        // 각각 단계에서, 위의 입력 데이터 스키마에서 지정한 이름으로 컬럼을 식별한다.

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                // 원치 않는 열을 제거하자
                .removeColumns("CustomerID", "MerchantID")

                // 이제 미국이나 캐나다의 상인과 관련된 거래 만 분석하려고한다고 가정 해보자. 그 나라들을 제외하고는 모든 것을 걸러 내자.
                // 여기서는 조건부 필터를 적용하자. 조건과 일치하는 모든 예제를 제거한다.
                // 조건은 "MerchantCountryCode"가 {"USA", "CAN"}이 아니다 이다.

                .filter(new ConditionFilter(new CategoricalColumnCondition("MerchantCountryCode", ConditionOp.NotInSet,
                        new HashSet<>(Arrays.asList("USA", "CAN")))))

                // 우리의 데이터 소스가 완벽하지 않다고 가정 해 보자. 0.0으로 대체하려는 마이너스 달러와 같은 잘못된 데이터가 있다.
                // 마이너스가 아닌 달러의 양은 그 값을 수정하기를 원치 않는다.
                // TransactionAmountUSD 열에 ConditionalReplaceValueTransform을 사용한다.

                .conditionalReplaceValueTransform("TransactionAmountUSD", // 작업을 할 열
                        new DoubleWritable(0.0), // 조건을 만족했을때 대체할 새로운 값
                        new DoubleColumnCondition("TransactionAmountUSD", ConditionOp.LessThan, 0.0)) // 조건: amount <
                                                                                                      // 0.0

                // Finally, let's suppose we want to parse our date/time column in a format like
                // "2016/01/01 17:50.000"
                // We use JodaTime internally, so formats can be specified as follows:
                // http://www.joda.org/joda-time/apidocs/org/joda/time/format/DateTimeFormat.html

                .stringToTimeTransform("DateTimeString", "YYYY-MM-DD HH:mm:ss.SSS", DateTimeZone.UTC)

                // However, our time column ("DateTimeString") isn't a String anymore. So let's
                // rename it to something better:
                .renameColumn("DateTimeString", "DateTime")

                // At this point, we have our date/time format stored internally as a long value
                // (Unix/Epoch format): milliseconds since 00:00.000 01/01/1970
                // Suppose we only care about the hour of the day. Let's derive a new column for
                // that, from the DateTime column
                .transform(new DeriveColumnsFromTimeTransform.Builder("DateTime")
                        .addIntegerDerivedColumn("HourOfDay", DateTimeFieldType.hourOfDay()).build())

                // We no longer need our "DateTime" column, as we've extracted what we need from
                // it. So let's remove it
                .removeColumns("DateTime")

                // We've finished with the sequence of operations we want to do: let's create
                // the final TransformProcess object
                .build();

        // After executing all of these operations, we have a new and different schema:
        Schema outputSchema = tp.getFinalSchema();

        System.out.println("\n\n\nSchema after transforming data:");
        System.out.println(outputSchema);

        // =====================================================================
        // Step 3: Load our data and execute the operations on Spark
        // =====================================================================

        // We'll use Spark local to handle our data

        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DataVec Example");

        JavaSparkContext sc = new JavaSparkContext(conf);

        String directory = new ClassPathResource("BasicDataVecExample/exampledata.csv").getFile().getParent(); // Normally
                                                                                                               // just
                                                                                                               // define
                                                                                                               // your
                                                                                                               // directory
                                                                                                               // like
                                                                                                               // "file:/..."
                                                                                                               // or
                                                                                                               // "hdfs:/..."
        JavaRDD<String> stringData = sc.textFile(directory);

        // We first need to parse this format. It's comma-delimited (CSV) format, so
        // let's parse it using CSVRecordReader:
        RecordReader rr = new CSVRecordReader();
        JavaRDD<List<Writable>> parsedInputData = stringData.map(new StringToWritablesFunction(rr));

        // Now, let's execute the transforms we defined earlier:
        JavaRDD<List<Writable>> processedData = SparkTransformExecutor.execute(parsedInputData, tp);

        // For the sake of this example, let's collect the data locally and print it:
        JavaRDD<String> processedAsString = processedData.map(new WritablesToStringFunction(","));
        // processedAsString.saveAsTextFile("file://your/local/save/path/here"); //To
        // save locally
        // processedAsString.saveAsTextFile("hdfs://your/hdfs/save/path/here"); //To
        // save to hdfs

        List<String> processedCollected = processedAsString.collect();
        List<String> inputDataCollected = stringData.collect();

        System.out.println("\n\n---- Original Data ----");
        for (String s : inputDataCollected)
            System.out.println(s);

        System.out.println("\n\n---- Processed Data ----");
        for (String s : processedCollected)
            System.out.println(s);

        System.out.println("\n\nDONE");
    }

}
