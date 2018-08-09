package org.deeplearning4j.examples.recurrent.word2vecsentiment;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

/** 이것은 Word2VecSentimentRNN 예제에서 사용되는 IMDB 검토 데이터셋에 특화된 DataSetIterator이다.
 * 이 데이터셋의 학습 또는 테스트셋 데이터와 WordVectors 객체 (일반적으로 https://code.google.com/p/word2vec/의 Google 뉴스 300 사전 학습 벡터)를 받아 학습 데이터셋을 생성한다
 * 입력 / 특징 : 각 단어 (알 수없는 단어가 제거 된)가 Word2Vec 벡터 표현으로 표현되는 가변 길이 시계열
 * 라벨 / 타겟 : 각 리뷰의 최종 타임 스텝 (단어)에서 예측 된 단일 클래스 (음수 또는 양수)
 *
 * @author Alex Black
 */

public class SentimentExampleIterator implements DataSetIterator {
    private final WordVectors wordVectors;
    private final int batchSize;
    private final int vectorSize;
    private final int truncateLength;

    private int cursor = 0;
    private final File[] positiveFiles;
    private final File[] negativeFiles;
    private final TokenizerFactory tokenizerFactory;

    /**
     * @param dataDirectory IMDB검토 데이터셋의 위치
     * @param wordVectors WordVectors 객체
     * @param batchSize 학습에 사용되는 미니배치 크기
     * @param truncateLength 리뷰가 초과하는 경우
     * @param train  true라면 학습데이터를 반환한다. false라면 테스트 데이터를 반환한다.
     */
    public SentimentExampleIterator(String dataDirectory, WordVectors wordVectors, int batchSize, int truncateLength, boolean train) throws IOException {
        this.batchSize = batchSize;
        this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;


        File p = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (train ? "train" : "test") + "/pos/") + "/");
        File n = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (train ? "train" : "test") + "/neg/") + "/");
        positiveFiles = p.listFiles();
        negativeFiles = n.listFiles();

        this.wordVectors = wordVectors;
        this.truncateLength = truncateLength;

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }


    @Override
    public DataSet next(int num) {
        if (cursor >= positiveFiles.length + negativeFiles.length) throw new NoSuchElementException();
        try{
            return nextDataSet(num);
        }catch(IOException e){
            throw new RuntimeException(e);
        }
    }

    private DataSet nextDataSet(int num) throws IOException {
        // 첫번째: 리뷰를 String으로 불러온다. positive, negative 리뷰로 분류한다.
        List<String> reviews = new ArrayList<>(num);
        boolean[] positive = new boolean[num];
        for( int i=0; i<num && cursor<totalExamples(); i++ ){
            if(cursor % 2 == 0){
                //긍정적인 리뷰
                int posReviewNumber = cursor / 2;
                String review = FileUtils.readFileToString(positiveFiles[posReviewNumber]);
                reviews.add(review);
                positive[i] = true;
            } else {
                //부정적인 리뷰
                int negReviewNumber = cursor / 2;
                String review = FileUtils.readFileToString(negativeFiles[negReviewNumber]);
                reviews.add(review);
                positive[i] = false;
            }
            cursor++;
        }

        //두번째: 토큰화는 미지의 단어를 검토하고 걸러낸다.
        List<List<String>> allTokens = new ArrayList<>(reviews.size());
        int maxLength = 0;
        for(String s : reviews){
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            List<String> tokensFiltered = new ArrayList<>();
            for(String t : tokens ){
                if(wordVectors.hasWord(t)) tokensFiltered.add(t);
            }
            allTokens.add(tokensFiltered);
            maxLength = Math.max(maxLength,tokensFiltered.size());
        }

        //가장 긴 리뷰가 'truncateLength'를 초과하는 경우 : 첫 번째 'truncateLength'단어 만 가져온다.
        if(maxLength > truncateLength) maxLength = truncateLength;

        //학습을 위해 데이터를 생성한다
        //다양한 길이의 reviews.size()가 있다
        INDArray features = Nd4j.create(reviews.size(), vectorSize, maxLength);
        INDArray labels = Nd4j.create(reviews.size(), 2, maxLength);    //두개의 레이블: 긍정적 혹은 부정적
        //서로 다른 길이의 리뷰와 마지막 타임 스텝에서 하나의 출력 만 처리하기 때문에 패딩 배열을 사용한다.
        //마스크 배열은 해당 예제의 해당 시간 단계에 데이터가 있으면 1을, 데이터가 패딩이면 0을 포함한다.
        INDArray featuresMask = Nd4j.zeros(reviews.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(reviews.size(), maxLength);

        int[] temp = new int[2];
        for( int i=0; i<reviews.size(); i++ ){
            List<String> tokens = allTokens.get(i);
            temp[0] = i;
            //리뷰에서 각 단어에 대한 단어 벡터를 가져 와서 교육 데이터에 넣는다.
            for( int j=0; j<tokens.size() && j<maxLength; j++ ){
                String token = tokens.get(j);
                INDArray vector = wordVectors.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);

                temp[1] = j;
                featuresMask.putScalar(temp, 1.0);  //이 예제에서는 Word가 존재하지만 (패딩이 아님) 피처 마스크의 시간 단계 -> 1.0
            }

            int idx = (positive[i] ? 0 : 1);
            int lastIdx = Math.min(tokens.size(),maxLength);
            labels.putScalar(new int[]{i,idx,lastIdx-1},1.0);   //레이블 셋팅: [0,1] 은 부정, [1,0]은 긍정
            labelsMask.putScalar(new int[]{i,lastIdx-1},1.0);   //이 예제의 마지막 타임 스탭에 자세한 출력값이 있다.
        }

        return new DataSet(features,labels,featuresMask,labelsMask);
    }

    @Override
    public int totalExamples() {
        return positiveFiles.length + negativeFiles.length;
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return 2;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return Arrays.asList("positive","negative");
    }

    @Override
    public boolean hasNext() {
        return cursor < numExamples();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {

    }
    @Override
    public  DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** Convenience method for loading review to String */
    public String loadReviewToString(int index) throws IOException{
        File f;
        if(index%2 == 0) f = positiveFiles[index/2];
        else f = negativeFiles[index/2];
        return FileUtils.readFileToString(f);
    }

    /** Convenience method to get label for review */
    public boolean isPositiveReview(int index){
        return index%2 == 0;
    }

    /**
     * 파일에서 신경망 출력 방법으로 전달할 수있는 기능 INDArray로 리뷰를 로드하는 데 사용되는 사후 교육
     *
     * @param file      리뷰를 가져올 파일
     * @param maxLength 최대 길이 (검토가 this보다 길면 : truncate to maxLength). nruncate하지 않으려면 Integer.MAX_VALUE를 사용하자.
     * @return          특징 배열
     * @throws IOException 파일을 읽을 수 없는 경우 예외처리
     */
    public INDArray loadFeaturesFromFile(File file, int maxLength) throws IOException {
        String review = FileUtils.readFileToString(file);
        return loadFeaturesFromString(review, maxLength);
    }

    /**
     * 문자열을 신경망 출력 방법으로 전달할 수있는 기능 INDArray로 변환하는 사후 사후 교육
     *
     * @param reviewContents 벡터화할 검토내용
     * @param maxLength 최대 길이 (검토가 this보다 길면 : truncate to maxLength). nruncate하지 않으려면 Integer.MAX_VALUE를 사용하자.
     * @return 입력 String에 대한 특징 배열
     */
    public INDArray loadFeaturesFromString(String reviewContents, int maxLength){
        List<String> tokens = tokenizerFactory.create(reviewContents).getTokens();
        List<String> tokensFiltered = new ArrayList<>();
        for(String t : tokens ){
            if(wordVectors.hasWord(t)) tokensFiltered.add(t);
        }
        int outputLength = Math.max(maxLength,tokensFiltered.size());

        INDArray features = Nd4j.create(1, vectorSize, outputLength);

        for( int j=0; j<tokens.size() && j<maxLength; j++ ){
            String token = tokens.get(j);
            INDArray vector = wordVectors.getWordVectorMatrix(token);
            features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
        }

        return features;
    }
}
