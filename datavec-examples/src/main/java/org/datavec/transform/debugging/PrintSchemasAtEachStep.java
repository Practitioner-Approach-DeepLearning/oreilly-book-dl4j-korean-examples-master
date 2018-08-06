package org.datavec.transform.debugging;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.writable.DoubleWritable;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;

import java.util.Arrays;
import java.util.HashSet;

/**
 * 이 예제는 DataVec 변환 기능에 대한 기본적인 예제이다. (BasicDataVecExample 기반)
 * 이것은 변환 프로세스의 각 단계 후에 스키마를 얻을 수 있음을 간단히 보여주기 위해 설계되었다.
 * 이것은 TransformProcess 스크립트를 디버깅할 때 유용할 것이다.
 * @author Alex Black
 */
public class PrintSchemasAtEachStep {

    public static void main(String[] args){

        //BasicDataVecExample에 따라 스키마 및 TransformProcess 정의
        Schema inputDataSchema = new Schema.Builder()
            .addColumnsString("DateTimeString", "CustomerID", "MerchantID")
            .addColumnInteger("NumItemsInTransaction")
            .addColumnCategorical("MerchantCountryCode", Arrays.asList("USA","CAN","FR","MX"))
            .addColumnDouble("TransactionAmountUSD",0.0,null,false,false)   //$ 0.0 이상, 최대 제한 없음, NaN 없음, 무한 값 없음
            .addColumnCategorical("FraudLabel", Arrays.asList("Fraud","Legit"))
            .build();

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
            .removeColumns("CustomerID","MerchantID")
            .filter(new ConditionFilter(new CategoricalColumnCondition("MerchantCountryCode", ConditionOp.NotInSet, new HashSet<>(Arrays.asList("USA","CAN")))))
            .conditionalReplaceValueTransform(
                "TransactionAmountUSD",     //조건이 만족 될 때
                new DoubleWritable(0.0),    //사용할 열의 새로운 값
                new DoubleColumnCondition("TransactionAmountUSD",ConditionOp.LessThan, 0.0)) //조건: amount < 0.0
            .stringToTimeTransform("DateTimeString","YYYY-MM-DD HH:mm:ss.SSS", DateTimeZone.UTC)
            .renameColumn("DateTimeString", "DateTime")
            .transform(new DeriveColumnsFromTimeTransform.Builder("DateTime").addIntegerDerivedColumn("HourOfDay", DateTimeFieldType.hourOfDay()).build())
            .removeColumns("DateTime")
            .build();


        // 각 단계별 진행 이후 스키마를 출력
        int numActions = tp.getActionList().size();

        for(int i=0; i<numActions; i++ ){
            System.out.println("\n\n==================================================");
            System.out.println("-- Schema after step " + i + " (" + tp.getActionList().get(i) + ") --");

            System.out.println(tp.getSchemaAfterStep(i));
        }


        System.out.println("DONE.");
    }

}
