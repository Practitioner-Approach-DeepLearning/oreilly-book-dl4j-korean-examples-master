package org.datavec.transform.logdata;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.regex.RegexLineRecordReader;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.LongColumnCondition;
import org.datavec.api.transform.condition.string.StringRegexColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.quality.DataQualityAnalysis;
import org.datavec.api.transform.reduce.Reducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.AnalyzeSpark;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.joda.time.DateTimeZone;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * DataVec을 사용하여 일부 웹 로그 데이터에서 일부 사전 처리 / 집계 작업을 수행하는 간단한 예제.
 * 상세 구현 단계:
 * - 데이터 로드
 * - 데이터 품질 분석 수행
 * - 기본 데이터 정리 및 전처리 수행
 * - 호스트에 의한 레코드 그룹화 그리고 각 집계값 계산 (요청 숫자, 전체 바이트 숫자)
 * - 결과 데이터 분석, 결과 출력
 *
 * 데이터는 해당 URL로 다운받을 수 있다 : http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html
* 로그 라인 예제
 * 199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245
 * unicomp6.unicomp.net - - [01/Jul/1995:00:00:06 -0400] "GET /shuttle/countdown/ HTTP/1.0" 200 3985
 *
 * @author Alex Black
 */
public class LogDataExample {

    /** 다운받기 위한 URL  */
    public static final String DATA_URL = "ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz";
    /** 학습/테스트 데이터를 저장하고 추출하기 위한 위치  */
    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "datavec_log_example/");
    public static final String EXTRACTED_PATH = FilenameUtils.concat(DATA_PATH, "data");

    public static void main(String[] args) throws Exception {
        // 준비
        downloadData();
        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DataVec Log Data Example");
        JavaSparkContext sc = new JavaSparkContext(conf);


        //=====================================================================
        //                  첫번째 : 입력 데이터 스키마 정의
        //=====================================================================

        //첫번째로 데이터를 위한 스키마를 지정해보자. 이 정보는  http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html에 기반한다.
        Schema schema = new Schema.Builder()
            .addColumnString("host")
            .addColumnString("timestamp")
            .addColumnString("request")
            .addColumnInteger("httpReplyCode")
            .addColumnInteger("replyBytes")
            .build();

        //=====================================================================
        //                      두번째: 유효하지 않은 라인 삭제
        //=====================================================================

        // 두번째로 위에 정의한 다운로드 URL로 데이터를 로드하자.
        JavaRDD<String> logLines = sc.textFile(EXTRACTED_PATH);

        //이 데이터에는 잘못된 행 수가 적다. 표준 스파크 기능을 사용하여 제거하자.
        logLines = logLines.filter(new Function<String,Boolean>() {
            @Override
            public Boolean call(String s) throws Exception {
                return s.matches("(\\S+) - - \\[(\\S+ -\\d{4})\\] \"(.+)\" (\\d+) (\\d+|-)");   //Regex for the format we expect
            }
        });

        //=====================================================================
        //          세번쨰 : 데이터 파싱 및 기본적인 분석 수행
        //=====================================================================

        //파싱하기 위해 RegexLineRecordReader를 사용한다. regex를 먼저 정의할 필요가 있다.
        String regex = "(\\S+) - - \\[(\\S+ -\\d{4})\\] \"(.+)\" (\\d+) (\\d+|-)";
        RecordReader rr = new RegexLineRecordReader(regex,0);
        JavaRDD<List<Writable>> parsed = logLines.map(new StringToWritablesFunction(rr));

        //품질을 체크해 봅시다, 먼저 정리해야할 것이 있다면 먼저 정리해야 할 것이다.
        DataQualityAnalysis dqa = AnalyzeSpark.analyzeQuality(schema, parsed);
        System.out.println("----- Data Quality -----");
        System.out.println(dqa);    //"replyBytes" column에 정수가 아니니 값이 들어가 있을 것이다.


        //=====================================================================
        //          네번쨰: 정리 및 파싱 및 수집
        //=====================================================================

        //Let's specify the transforms we want to do
        //원하는데로 변환을 지정해보자
        TransformProcess tp = new TransformProcess.Builder(schema)
            // 첫번째: 정수가 아닌 항목을 값 0으로 대체하여 "replyBytes"열을 정리한다.
            .conditionalReplaceValueTransform("replyBytes",new IntWritable(0), new StringRegexColumnCondition("replyBytes","\\D+"))
            //두번쨰: date/time 문자열을 파싱하자
            .stringToTimeTransform("timestamp","dd/MMM/YYYY:HH:mm:ss Z", DateTimeZone.forOffsetHours(-4))

            //호스트별 그룹화 및 요약된 행렬 정리
            .reduce(new Reducer.Builder(ReduceOp.CountUnique)
                .keyColumns("host")                 //keyColumns == columns 이면 그룹화
                .countColumns("timestamp")          //timestamp열의 수를 센다
                .countUniqueColumns("request", "httpReplyCode")     //httpReplyCode에 대한 값을 중복값을 제거하고 센다
                .sumColumns("replyBytes")           //replyBytes 열의 수를 합한다.
                .build())

            .renameColumn("count", "numRequests")

            //마지막으로 총 100 만 바이트 미만을 요청한 모든 호스트를 걸러낸다.
            .filter(new ConditionFilter(new LongColumnCondition("sum(replyBytes)", ConditionOp.LessThan, 1000000)))
            .build();

        JavaRDD<List<Writable>> processed = SparkTransformExecutor.execute(parsed, tp);
        processed.cache();


        //=====================================================================
        //       다섯번째 : 최종 데이터에 대한 분석 실행, 결과 출력
        //=====================================================================

        Schema finalDataSchema = tp.getFinalSchema();
        long finalDataCount = processed.count();
        List<List<Writable>> sample = processed.take(10);

        DataAnalysis analysis = AnalyzeSpark.analyze(finalDataSchema, processed);

        sc.stop();
        Thread.sleep(4000); //스파크를 종료하고 (스팸 콘솔을 중지하는) 시간을 준다.


        System.out.println("----- Final Data Schema -----");
        System.out.println(finalDataSchema);

        System.out.println("\n\nFinal data count: " + finalDataCount);

        System.out.println("\n\n----- Samples of final data -----");
        for(List<Writable> l : sample){
            System.out.println(l);
        }

        System.out.println("\n\n----- Analysis -----");
        System.out.println(analysis);
    }


    private static void downloadData() throws Exception {
        //디렉토리 생성이 필요하면 디렉토를 만들도록 한다.
        File directory = new File(DATA_PATH);
        if(!directory.exists()) directory.mkdir();

        // 다운로드 파일
        String archivePath = DATA_PATH + "NASA_access_log_Jul95.gz";
        File archiveFile = new File(archivePath);
        File extractedFile = new File(EXTRACTED_PATH,"access_log_July95.txt");
        new File(extractedFile.getParent()).mkdirs();

        if( !archiveFile.exists() ){
            System.out.println("Starting data download (20MB)...");
            FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile);
            System.out.println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath());
            //tar.gz 파일을 output 디렉토리에 압축해제
            extractGzip(archivePath, extractedFile.getAbsolutePath());
        } else {
            //아카이브 (.tar.gz)가 있고 데이터가 이미 추출되었다고 가정하자.
            System.out.println("Data (.gz file) already exists at " + archiveFile.getAbsolutePath());
            if( !extractedFile.exists()){
                //tar.gz 파일을 output 디렉토리에 압축해제
                extractGzip(archivePath, extractedFile.getAbsolutePath());
            } else {
                System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
            }
        }
    }

    private static final int BUFFER_SIZE = 4096;
    private static void extractGzip(String filePath, String outputPath) throws IOException {
        System.out.println("Extracting files...");
        byte[] buffer = new byte[BUFFER_SIZE];

        try{
            GZIPInputStream gzis = new GZIPInputStream(new FileInputStream(new File(filePath)));

            FileOutputStream out = new FileOutputStream(new File(outputPath));

            int len;
            while ((len = gzis.read(buffer)) > 0) {
                out.write(buffer, 0, len);
            }

            gzis.close();
            out.close();

            System.out.println("Done");
        }catch(IOException ex){
            ex.printStackTrace();
        }
    }

}
