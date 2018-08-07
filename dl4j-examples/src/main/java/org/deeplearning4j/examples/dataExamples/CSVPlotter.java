package org.deeplearning4j.examples.dataExamples;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * csv 파일 읽기. DL4J를 사용해 피팅 후 그리기
 *
 * @author 로버트 알테나
 */
public class CSVPlotter {

    public static void main( String[] args ) throws IOException, InterruptedException
    {
        String filename = new ClassPathResource("/DataExamples/CSVPlotData.csv").getFile().getPath();
    	DataSet ds = readCSVDataset(filename);

    	ArrayList<DataSet> DataSetList = new ArrayList<>();
    	DataSetList.add(ds);

    	plotDataset(DataSetList); // 데이터 그리기, 정확한 데이터인지 확인

    	MultiLayerNetwork net =fitStraightline(ds);

    	// ND4J를 사용해 x의 최솟값, 최댓값 가져오기
    	NormalizerMinMaxScaler preProcessor = new NormalizerMinMaxScaler();
    	preProcessor.fit(ds);
        int nSamples = 50;
        INDArray x = Nd4j.linspace(preProcessor.getMin().getInt(0),preProcessor.getMax().getInt(0),nSamples).reshape(nSamples, 1);
        INDArray y = net.output(x);
        DataSet modeloutput = new DataSet(x,y);
        DataSetList.add(modeloutput);

    	plotDataset(DataSetList);    //모델 피팅 및 데이터 그리기
    }

	/**
	 * 신경망을 사용해 직선에 피팅하기
	 * @param ds 피팅할 데이터셋
	 * @return 데이터에 피팅된 신경망
	 */
	private static MultiLayerNetwork fitStraightline(DataSet ds){
		int seed = 12345;
		int iterations = 1;
		int nEpochs = 200;
		double learningRate = 0.00001;
		int numInputs = 1;
	    int numOutputs = 1;

	    //
	    // 입력 하나를 출력 하나와 연결하기
	    // 모델 결과는 직선
	    //
		MultiLayerConfiguration conf = new  NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numOutputs)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numOutputs).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
	    net.setListeners(new ScoreIterationListener(1));

	    for( int i=0; i<nEpochs; i++ ){
	    	net.fit(ds);
	    }

	    return net;
	}

    /**
	 * CSV 파일을 읽어 데이터셋 만들기
     *
	 * 올바른 생성자를 사용할 것:
     * DataSet ds = new RecordReaderDataSetIterator(rr,batchSize);
     * 반환되는 데이터는 다음과 같음
     * ===========INPUT===================
     *[[12.89, 22.70],
     * [19.34, 20.47],
     * [16.94,  6.08],
     *  [15.87,  8.42],
     *  [10.71, 26.18]]
     *
	 *  프레임워크가 데이터를 다루는 방식과는 다름
     *
     *  예를 들면:
     *   RecordReaderDataSetIterator(rr,batchSize, 1, 1, true);
     *  반환값
     *   ===========INPUT===================
     * [12.89, 19.34, 16.94, 15.87, 10.71]
     * =================OUTPUT==================
     * [22.70, 20.47,  6.08,  8.42, 26.18]
     *
	 * 이 출력은 그대로 회귀 분석에 사용될 수 있음
     */
	private static DataSet readCSVDataset(String filename) throws IOException, InterruptedException{
		int batchSize = 1000;
		RecordReader rr = new CSVRecordReader();
		rr.initialize(new FileSplit(new File(filename)));

		DataSetIterator iter =  new RecordReaderDataSetIterator(rr,batchSize, 1, 1, true);
		return iter.next();
	}

	/**
	 * 제공된 데이터셋의 xy 플롯을 생성
	 * Generate an xy plot of the datasets provided.
	 */
	private static void plotDataset(ArrayList<DataSet> DataSetList){

		XYSeriesCollection c = new XYSeriesCollection();

		int dscounter = 1; //use to name the dataseries
		for (DataSet ds : DataSetList)
		{
			INDArray features = ds.getFeatures();
			INDArray outputs= ds.getLabels();

			int nRows = features.rows();
			XYSeries series = new XYSeries("S" + dscounter);
			for( int i=0; i<nRows; i++ ){
				series.add(features.getDouble(i), outputs.getDouble(i));
			}

			c.addSeries(series);
		}

        String title = "title";
		String xAxisLabel = "xAxisLabel";
		String yAxisLabel = "yAxisLabel";
		PlotOrientation orientation = PlotOrientation.VERTICAL;
		boolean legend = false;
		boolean tooltips = false;
		boolean urls = false;
		JFreeChart chart = ChartFactory.createScatterPlot(title , xAxisLabel, yAxisLabel, c, orientation , legend , tooltips , urls);
    	JPanel panel = new ChartPanel(chart);

    	 JFrame f = new JFrame();
    	 f.add(panel);
    	 f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
         f.pack();
         f.setTitle("Training Data");

         f.setVisible(true);
	}
}
