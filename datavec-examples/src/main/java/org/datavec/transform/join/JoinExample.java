package org.datavec.transform.join;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.join.Join;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.joda.time.DateTimeZone;

import java.util.Arrays;
import java.util.List;

/**
 * 이번 예제는 DataVec에서 조인을 어떻게 실행하는지 보여준다
 * 조인은 database/SQL의 join과 유사하다. 여러 출처의 데이터가 두 출처에 나타나는 공통 키에 따라 함께 결합된다.
 *
 * 이 예제에서는 두개의 CSV 파일을 로드한다. 이것은 무작위로 생성된 고객 데이터이다.
 *
 * @author Alex Black
 */
public class JoinExample {

    public static void main(String[] args) throws Exception {

        String customerInfoPath = new ClassPathResource("JoinExample/CustomerInfo.csv").getFile().getPath();
        String purchaseInfoPath = new ClassPathResource("JoinExample/CustomerPurchases.csv").getFile().getPath();

        // 첫번쨰 : 두개의 데이터 셋을 스키마 형태로 정의하자.
        Schema customerInfoSchema = new Schema.Builder()
            .addColumnLong("customerID")
            .addColumnString("customerName")
            .addColumnCategorical("customerCountry", Arrays.asList("USA","France","Japan","UK"))
            .build();

        Schema customerPurchasesSchema = new Schema.Builder()
            .addColumnLong("customerID")
            .addColumnTime("purchaseTimestamp", DateTimeZone.UTC)
            .addColumnLong("productID")
            .addColumnInteger("purchaseQty")
            .addColumnDouble("unitPriceUSD")
            .build();



        // 스파크 준비
        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DataVec Join Example");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // 데이터 로드
        RecordReader rr = new CSVRecordReader();
        JavaRDD<List<Writable>> customerInfo = sc.textFile(customerInfoPath).map(new StringToWritablesFunction(rr));
        JavaRDD<List<Writable>> purchaseInfo = sc.textFile(purchaseInfoPath).map(new StringToWritablesFunction(rr));
         //출력 이후 데이터 수집
        List<List<Writable>> customerInfoList = customerInfo.collect();
        List<List<Writable>> purchaseInfoList = purchaseInfo.collect();

        //고객 ID를 기반으로 두개의 데이터 셋을 조인해보자.
        Join join = new Join.Builder(Join.JoinType.Inner)
            .setJoinColumns("customerID")
            .setSchemas(customerInfoSchema, customerPurchasesSchema)
            .build();

        JavaRDD<List<Writable>> joinedData = SparkTransformExecutor.executeJoin(join, customerInfo, purchaseInfo);
        List<List<Writable>> joinedDataList = joinedData.collect();

        //스파크 중단, 콘솔에 로깅될때까지 몇 초간 여유를 준다.
        sc.stop();
        Thread.sleep(2000);

        //기본 데이터 출력
        System.out.println("\n\n----- Customer Information -----");
        System.out.println("Source file: " + customerInfoPath);
        System.out.println(customerInfoSchema);
        System.out.println("Customer Information Data:");
        for(List<Writable> line : customerInfoList){
            System.out.println(line);
        }


        System.out.println("\n\n----- Purchase Information -----");
        System.out.println("Source file: " + purchaseInfoPath);
        System.out.println(customerPurchasesSchema);
        System.out.println("Purchase Information Data:");
        for(List<Writable> line : purchaseInfoList){
            System.out.println(line);
        }

        // 조인된 데이터 출력
        System.out.println("\n\n----- Joined Data -----");
        System.out.println(join.getOutputSchema());
        System.out.println("Joined Data:");
        for(List<Writable> line : joinedDataList){
            System.out.println(line);
        }



    }

}
