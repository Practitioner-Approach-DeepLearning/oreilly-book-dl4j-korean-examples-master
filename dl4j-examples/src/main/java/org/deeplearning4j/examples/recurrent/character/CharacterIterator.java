package org.deeplearning4j.examples.recurrent.character;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.*;

/** GravesLSTMCharModellingExample에서 사용하기위한 간단한 DataSetIterator이다.
 * 텍스트 파일과 몇 가지 옵션이 주어지면 학습을 위한 특성 벡터와 라벨을 생성한다. 여기서 시퀀스의 다음 문자를 예측한다.
 * 
 * 이것은 텍스트 파일의 위치를 0, exampleLength, 2 * exampleLength 등의 오프셋에서 무작위로 선택하여 각 시퀀스를 시작하여 수행된다. 
 * 그런 다음 각 문자를 색인, 즉 1 핫 벡터로 변환한다.
 * 그러면 문자 'a'는 [1,0,0,0, ...]이되고 'b'는 [0,1,0,0, ...]이 된다.
 * 특징 벡터 및 레이블은 모두 동일한 길이의 단일 핫 벡터이다.
 * @author Alex Black
 */

public class CharacterIterator implements DataSetIterator {
    //유효한 문자열
	private char[] validCharacters;
    //각 문자를 입력 / 출력의 인덱스에 매핑한다.
	private Map<Character,Integer> charToIdxMap;
    //입력 파일의 모든 문자 (필터링 된 후 유효한 문자로만 필터링 됨)
	private char[] fileCharacters;
    //각 예제의 길이 / 미니배치 (문자 수)
	private int exampleLength;
    //각 미니배치의 크기 (예제의 수)
	private int miniBatchSize;
	private Random rng;
    //각 예제의 시작에 대한 오프셋
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

	/**
	 * @param textFilePath 샘플의 생성에 사용하는 텍스트 파일의 패스.
	 * @param textFileEncoding 텍스트 파일의 인코딩. Charset.defaultCharset()을 시도 할 수 있다.
	 * @param miniBatchSize 미니 배치 당 예제의 수
	 * @param exampleLength 각 입출력 벡터의 문자 수
	 * @param validCharacters 유효한 문자의 문자 배열. 이 배열에없는 문자는 제거된다.
	 * @param rng 난수 생성기 (필요에 따라서 반복 가능)
	 * @throws IOException - 텍스트 파일을 로드 할 수 없는 경우
	 */
	public CharacterIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
                             char[] validCharacters, Random rng) throws IOException {
		if( !new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);
		if( miniBatchSize <= 0 ) throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
		this.validCharacters = validCharacters;
		this.exampleLength = exampleLength;
		this.miniBatchSize = miniBatchSize;
		this.rng = rng;

		//유효한 문자를 저장하면 나중에 벡터화에 사용할 수 있다.
		charToIdxMap = new HashMap<>();
		for( int i=0; i<validCharacters.length; i++ ) charToIdxMap.put(validCharacters[i], i);

		//파일을로드하고 내용을 char[]로 변환.
		boolean newLineValid = charToIdxMap.containsKey('\n');
		List<String> lines = Files.readAllLines(new File(textFilePath).toPath(),textFileEncoding);
		int maxSize = lines.size();	//각 행의 끝에서 줄 바꿈 문자를 설명하기 위해 lines.size ()를 추가하자.
		for( String s : lines ) maxSize += s.length();
		char[] characters = new char[maxSize];
		int currIdx = 0;
		for( String s : lines ){
			char[] thisLine = s.toCharArray();
			for (char aThisLine : thisLine) {
				if (!charToIdxMap.containsKey(aThisLine)) continue;
				characters[currIdx++] = aThisLine;
			}
			if(newLineValid) characters[currIdx++] = '\n';
		}

		if( currIdx == characters.length ){
			fileCharacters = characters;
		} else {
			fileCharacters = Arrays.copyOfRange(characters, 0, currIdx);
		}
		if( exampleLength >= fileCharacters.length ) throw new IllegalArgumentException("exampleLength="+exampleLength
				+" cannot exceed number of valid characters in file ("+fileCharacters.length+")");

		int nRemoved = maxSize - fileCharacters.length;
		System.out.println("Loaded and converted file: " + fileCharacters.length + " valid characters of "
		+ maxSize + " total characters (" + nRemoved + " removed)");

        initializeOffsets();
	}
	
    /** a-z, A-Z, 0-9 및 공통 구두점 등을 포함한 최소 문자 세트  */
	public static char[] getMinimalCharacterSet(){
		List<Character> validChars = new LinkedList<>();
		for(char c='a'; c<='z'; c++) validChars.add(c);
		for(char c='A'; c<='Z'; c++) validChars.add(c);
		for(char c='0'; c<='9'; c++) validChars.add(c);
		char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
		for( char c : temp ) validChars.add(c);
		char[] out = new char[validChars.size()];
		int i=0;
		for( Character c : validChars ) out[i++] = c;
		return out;
	}

	/** getMinimalCharacterSet ()에 따라 약간의 추가 문자가 포함된다. */
	public static char[] getDefaultCharacterSet(){
		List<Character> validChars = new LinkedList<>();
		for(char c : getMinimalCharacterSet() ) validChars.add(c);
		char[] additionalChars = {'@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_',
				'\\', '|', '<', '>'};
		for( char c : additionalChars ) validChars.add(c);
		char[] out = new char[validChars.size()];
		int i=0;
		for( Character c : validChars ) out[i++] = c;
		return out;
	}

	public char convertIndexToCharacter( int idx ){
		return validCharacters[idx];
	}

	public int convertCharacterToIndex( char c ){
		return charToIdxMap.get(c);
	}

	public char getRandomCharacter(){
		return validCharacters[(int) (rng.nextDouble()*validCharacters.length)];
	}

	public boolean hasNext() {
		return exampleStartOffsets.size() > 0;
	}

	public DataSet next() {
		return next(miniBatchSize);
	}

	public DataSet next(int num) {
		if( exampleStartOffsets.size() == 0 ) throw new NoSuchElementException();

        int currMinibatchSize = Math.min(num, exampleStartOffsets.size());
		//Allocate space:
        //Note the order here:
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
		//Why 'f' order here? See http://deeplearning4j.org/usingrnns.html#data section "Alternative: Implementing a custom DataSetIterator"
		// 공간 할당 :
        // 아래내용 참고 :
        // dimension 0 = minibatch에있는 예제의 수
        // dimension 1 = 각 벡터의 크기 (즉, 문자 수)
        // dimension 2 = 각 시계열의 길이 / 예제
		//  http://deeplearning4j.org/usingrnns.html#data 섹션을 참고하자.
		INDArray input = Nd4j.create(new int[]{currMinibatchSize,validCharacters.length,exampleLength}, 'f');
		INDArray labels = Nd4j.create(new int[]{currMinibatchSize,validCharacters.length,exampleLength}, 'f');

        for( int i=0; i<currMinibatchSize; i++ ){
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            int currCharIdx = charToIdxMap.get(fileCharacters[startIdx]);	//현재 입력
            int c=0;
            for( int j=startIdx+1; j<endIdx; j++, c++ ){
                int nextCharIdx = charToIdxMap.get(fileCharacters[j]);		//예측을 위한 다음 문자
                input.putScalar(new int[]{i,currCharIdx,c}, 1.0);
                labels.putScalar(new int[]{i,nextCharIdx,c}, 1.0);
                currCharIdx = nextCharIdx;
            }
        }

		return new DataSet(input,labels);
	}

	public int totalExamples() {
		return (fileCharacters.length-1) / miniBatchSize - 2;
	}

	public int inputColumns() {
		return validCharacters.length;
	}

	public int totalOutcomes() {
		return validCharacters.length;
	}

	public void reset() {
        exampleStartOffsets.clear();
		initializeOffsets();
	}

    private void initializeOffsets() {
        //이것은 파일의 일부분을 가져 오는 순서를 정의한다.
        int nMinibatchesPerEpoch = (fileCharacters.length - 1) / exampleLength - 2;   //-2: 종료 인덱스 및 부분적인 예
        for (int i = 0; i < nMinibatchesPerEpoch; i++) {
            exampleStartOffsets.add(i * exampleLength);
        }
        Collections.shuffle(exampleStartOffsets, rng);
    }

	public boolean resetSupported() {
		return true;
	}

    @Override
    public boolean asyncSupported() {
        return true;
    }

    public int batch() {
		return miniBatchSize;
	}

	public int cursor() {
		return totalExamples() - exampleStartOffsets.size();
	}

	public int numExamples() {
		return totalExamples();
	}

	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public List<String> getLabels() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}

}
