package org.deeplearning4j.examples.feedforward.classification.detectgender;

/**
 * 11/7/2016에 KIT Solutions (www.kitsol.com)가 생성.
 */

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.Charset;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.datavec.api.conf.Configuration;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.berkeley.Pair;


/**
 * GenderRecordReader 클래스는 다음 작업을 수행한다.
 * - 초기화 메서드에서 생성자의 레이블에 지정된 CSV 파일을 읽는다.
 * - 사람들의 이름 및 성별 데이터를 이진 변환 데이터로 로드한다.
 * - RecordReaderDataSetIterator에서 사용할 수 있는 이진 문자열 반복기 생성
 */

public class GenderRecordReader extends LineRecordReader
{
    // 생성자에 전달된 레이블을 보고나할 리스트
    private List<String> labels;

    // 사람 이름으로 생성된 실제 이진 데이터를 포함하는 최종 리스트로, 끝에 레이블(1 또는 0)이 포함된다.
    private List<String> names = new ArrayList<String>();

    // 원본 데이터의 모든 사람 이름에서 뽑은 가능한 모든 알파벳을 포함하는 문자열
    // 이 문자열은 사람 이름을 쉼표로 구분된 이진 문자열로 변환하는데 사용된다.
    private String possibleCharacters = "";

    // 모든 사람의 이름 중 가장 긴 이름의 길이를 저장
    public int maxLengthName = 0;

    // 남성 이름과 여성 이름을 모두 포함한 총 이름 수를 저장
    // 이 변수는 PredictGenderTrain.java에서는 사용되지 않는다.
    private int totalRecords = 0;

    // next() 메서드에서 사용할 이름 리스트에 대한 반복자
    private Iterator<String> iter;

    /**
     * 클라이언트 애플리케이션이 레이블 목록을 전달할 수 있도록 허용하는 생성자
     * @param labels - 클라이언트 애플리케이션이 모든 레이블을 전달하는 문자열 목록 (예: "M"과 "F")
     */
    public GenderRecordReader(List<String> labels)
    {
        this.labels = labels;
        //this.labels = this.labels.stream().map(element -> element + ".csv").collect(Collectors.toList());
        //System.out.println("labels : " + this.labels);
    }

    /**
     * 이름 리스트의 레코드 개수를 반환
     * @return - 레코드 개수
     */
    private int totalRecords()
    {
        return totalRecords;
    }


    /**
     * 이 메서드는 다음 단계에 따라 동작한다.
     * - 생성자에 설정된 레이블셋에서 유추한 이름(지정된 폴더)의 파일을 찾는다.
     * - 파일에는 사람 이름과 성별(M 또는 F)가 있어야 한다.
     *   예. Deepan,M
     *        Trupesh,M
     *        Vinay,M
     *        Ghanshyam,M
     *
     *        Meera,F
     *        Jignasha,F
     *        Chaku,F
     *
     * - M.csv, F.csv 등과 같이 남성과 여성의 이름 파일은 서로 달라야 한다.
     * - 모든 이름을 임시 리스트에 채운다.
     * - 모든 사람 이름에 대해 각 알파벳에 대한 이진 문자열을 생선한다.
     * - 각 이름의 모든 알파벳에 대한 이진 문자열을 결합한다.
     * - 위 단계에서 언급한 이진 문자열을 생성하기 위해 모든 고유한 알파벳을 찾는다.
     * - 모든 파일에서 동일한 수의 레코드를 가져온다. 이렇게 하려면 가장 적은 레코드 개수를 가진 파일을 찾은 다음 모든 파일에서 해당 레코드 수를 가져와
     *   서로 다른 레이블의 데이터 간에 균형을 유지한다.
     * - 첨고: 자바 8의 스트림 기능을 사용하면 처리 속도가 빨라진다. 파일을 처리하는 기본적인 방법은 5~7분 이상 걸린다. 스트림 사용 시 800~900밀리초가 걸린다.
     * - 최종 변환 이진 데이터는 List<String> 타입의 names 변수에 저장된다.
     * - next() 메서드에 사용할 이름 리스트에서 반복자를 설정한다.
     * @param split - 사용자는 남성 또는 여성 이름이 포함된 CSV 파일을 포함하는 디렉토리를 전달할 수 있다.
     * @throws IOException
     * @throws InterruptedException
     */
    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException
    {
        if(split instanceof FileSplit)
        {
            URI[] locations = split.locations();
            if(locations != null && locations.length > 1)
            {
                String longestName = "";
                String uniqueCharactersTempString = "";
                List<Pair<String, List<String>>> tempNames = new ArrayList<Pair<String, List<String>>>();
                for(URI location : locations)
                {
                    File file = new File(location);
                    List<String> temp  = this.labels.stream().filter(line -> file.getName().equals(line + ".csv")).collect(Collectors.toList());
                    if(temp.size() > 0)
                    {
                        java.nio.file.Path path = Paths.get(file.getAbsolutePath());
                        List<String> tempList = java.nio.file.Files.readAllLines(path, Charset.defaultCharset()).stream().map(element -> element.split(",")[0]).collect(Collectors.toList());

                        Optional<String> optional = tempList.stream().reduce((name1, name2)->name1.length() >= name2.length() ? name1 : name2);
                        if (optional.isPresent() && optional.get().length() > longestName.length())
                            longestName = optional.get();

                        uniqueCharactersTempString = uniqueCharactersTempString + tempList.toString();
                        Pair<String,List<String>> tempPair = new Pair<String,List<String>>(temp.get(0),tempList);
                        tempNames.add(tempPair);
                    }
                    else
                        throw new InterruptedException("File missing for any of the specified labels");
                }

                this.maxLengthName = longestName.length();

                String unique = Stream.of(uniqueCharactersTempString).map(w -> w.split("")).flatMap(Arrays::stream).distinct().collect(Collectors.toList()).toString();

                char[] chars = unique.toCharArray();
                Arrays.sort(chars);

                unique = new String(chars);
                unique = unique.replaceAll("\\[", "").replaceAll("\\]","").replaceAll(",","");
                if(unique.startsWith(" "))
                    unique = " " + unique.trim();

                this.possibleCharacters = unique;

                Pair<String, List<String>> tempPair = tempNames.get(0);
                int minSize = tempPair.getSecond().size();
                for(int i=1;i<tempNames.size();i++)
                {
                    if (minSize > tempNames.get(i).getSecond().size())
                        minSize = tempNames.get(i).getSecond().size();
                }

                List<Pair<String, List<String>>> oneMoreTempNames = new ArrayList<Pair<String, List<String>>>();
                for(int i=0;i<tempNames.size();i++)
                {
                    int diff = Math.abs(minSize - tempNames.get(i).getSecond().size());
                    List<String> tempList = new ArrayList<String>();

                    if (tempNames.get(i).getSecond().size() > minSize) {
                        tempList = tempNames.get(i).getSecond();
                        tempList = tempList.subList(0, tempList.size() - diff);
                    }
                    else
                        tempList = tempNames.get(i).getSecond();
                    Pair<String, List<String>> tempNewPair = new Pair<String, List<String>>(tempNames.get(i).getFirst(),tempList);
                    oneMoreTempNames.add(tempNewPair);
                }
                tempNames.clear();

                List<Pair<String, List<String>>> secondMoreTempNames = new ArrayList<Pair<String, List<String>>>();

                for(int i=0;i<oneMoreTempNames.size();i++)
                {
                    int gender = oneMoreTempNames.get(i).getFirst().equals("M") ? 1 : 0;
                    List<String> secondList = oneMoreTempNames.get(i).getSecond().stream().map(element -> getBinaryString(element.split(",")[0],gender)).collect(Collectors.toList());
                    Pair<String,List<String>> secondTempPair = new Pair<String, List<String>>(oneMoreTempNames.get(i).getFirst(),secondList);
                    secondMoreTempNames.add(secondTempPair);
                }
                oneMoreTempNames.clear();

                for(int i=0;i<secondMoreTempNames.size();i++)
                {
                    names.addAll(secondMoreTempNames.get(i).getSecond());
                }
                secondMoreTempNames.clear();
                this.totalRecords = names.size();
                Collections.shuffle(names);
                this.iter = names.iterator();
            }
            else
                throw new InterruptedException("File missing for any of the specified labels");
        }
        else if (split instanceof InputStreamInputSplit)
        {
            System.out.println("InputStream Split found...Currently not supported");
            throw new InterruptedException("File missing for any of the specified labels");
        }
    }


    /**
     * - iter 반복자를 사용해 이름 리스트에서 한번에 하나의 레코드를 가져온다.
     * - Writable 리스트에 저장하고 반환한다.
     *
     * @return
     */
    @Override
    public List<Writable> next()
    {
        if (iter.hasNext())
        {
            List<Writable> ret = new ArrayList<>();
            String currentRecord = iter.next();
            String[] temp = currentRecord.split(",");
            for(int i=0;i<temp.length;i++)
            {
                ret.add(new DoubleWritable(Double.parseDouble(temp[i])));
            }
            return ret;
        }
        else
            throw new IllegalStateException("no more elements");
    }

    @Override
    public boolean hasNext()
    {
        if(iter != null) {
            return iter.hasNext();
        }
        throw new IllegalStateException("Indeterminant state: record must not be null, or a file iterator must exist");
    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    @Override
    public Configuration getConf() {
        return conf;
    }

    @Override
    public void reset()
    {
        this.iter = names.iterator();
    }

    /**
     * 이 메서드는 전체 이름 문자열에 대한 이진 문자열을 제공한다.
     * - "PossibleCharacters" 문자열을 사용해 문자열에서 알파벳과 동등한 10진수 값을 찾는다.
     * - 각 알파벳에 대한 이진 문자열을 생성한다.
     * - 왼쪽 패딩을 적용해 이진 문자열읠 길이를 5로 만든다.
     * - 이름의 모든 알파벳에 대한 이진문자열을 결합한다.
     * - 오른쪽 패딩으로 모든 이름 길이를 가장 긴 이름 길이와 동일하게 만들어 이진 문자열을 완성한다.
     * - 레이블을 끝에 축자한다 (남성은 1, 여성은 0)
     * @param name - 이진 문자열로 변환될 사람 이름
     * @param gender - 이름의 이진 문자열 끝에 추가할 레이블 값을 결정하는 변수
     * @return
     */
    private String getBinaryString(String name, int gender)
    {
        String binaryString = "";
        for (int j = 0; j < name.length(); j++)
        {
            String fs = org.apache.commons.lang3.StringUtils.leftPad(Integer.toBinaryString(this.possibleCharacters.indexOf(name.charAt(j))),5,"0");
            binaryString = binaryString + fs;
        }
        //binaryString = String.format("%-" + this.maxLengthName*5 + "s",binaryString).replace(' ','0'); // 이 방법은 StringUtils보다 오래 걸리므로 주석 처리했다.

        binaryString  = org.apache.commons.lang3.StringUtils.rightPad(binaryString,this.maxLengthName*5,"0");
        binaryString = binaryString.replaceAll(".(?!$)", "$0,");

        //System.out.println("binary String : " + binaryString);
        return binaryString + "," + String.valueOf(gender);
    }
}
