package org.deeplearning4j.examples.unsupervised.deepbelief;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.Collections;

/**
 *
 * @author Adam Gibson
 */
public class DeepAutoEncoderExample_Pair {

    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoderExample_Web.class);

    @SuppressWarnings("unused")
	public static void main(String[] args) throws Exception {
    	
        final int numRows = 12;
        final int numColumns = 12;       
        int dim = 144;
        int batchSize = 35;
        
        int seed = 123;
        int numSamples = MnistDataFetcher.NUM_EXAMPLES;
        int nEpochs = 10;
        int iterations = 1;
        int listenerFreq = iterations/10;
        
        int labelIndexInDataset = 0;
            
        int numLinesToSkip = 0;
        String delimiter = ",";   
        
        log.info("Load data....");
        //First: get the dataset usingthe record reader         
        RecordReader rr = new CSVRecordReader();
        rr.initialize(
        	new FileSplit(
        	new File("C:/Workspace/Projects/dl4j-example/dl4j-examples/src/main/resources/classification/DLPair.csv")));
        
        //Second: the RecordReaderDateSetIterator handles conversion to Dataset objects, ready for use in neural network    
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, 
        		labelIndexInDataset, dim);
        
        DataSet allData = iterator.next();
                
        
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.6);
        
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();
        
//        DataNormalization normalizer = new NormalizerStandardize();
//        normalizer.fit(trainingData);
//        normalizer.transform(trainingData);
//        normalizer.transform(testData);
        
        
        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list()
                .layer(0, new RBM.Builder().nIn(numRows * numColumns).nOut(15).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//                .layer(1, new RBM.Builder().nIn(200).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//                .layer(2, new RBM.Builder().nIn(100).nOut(50).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//                .layer(3, new RBM.Builder().nIn(50).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//                .layer(4, new RBM.Builder().nIn(100).nOut(30).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) //encoding stops
//                .layer(5, new RBM.Builder().nIn(30).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) //decoding starts
//                .layer(6, new RBM.Builder().nIn(100).nOut(250).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//                .layer(7, new RBM.Builder().nIn(250).nOut(500).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//                .layer(8, new RBM.Builder().nIn(500).nOut(1000).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT).nIn(15).nOut(numRows*numColumns).build())
                .pretrain(true).backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model....");
//        while(iter.hasNext()) {
//            DataSet next = iter.next();
//            model.fit(new DataSet(next.getFeatureMatrix(),next.getFeatureMatrix()));
//        }
        
        for(int i = 0; i < nEpochs; i++){
        	model.fit(allData);
        }
        
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatureMatrix());
        eval.eval(testData.getLabels(), output);
//        eval.eval(allData.getLabels(), output);
        log.info(eval.stats());


    }


}
