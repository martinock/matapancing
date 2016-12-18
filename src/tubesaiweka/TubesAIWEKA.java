/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubesaiweka;


import java.io.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

/**
 *
 * Kelompok Mata Pancing AI - 13514001, 13514048, 13514055, 13514084
 */
public class TubesAIWEKA {
    
    static int classIndex = 0;
    
    public static Instances filter(Instances data) throws Exception{
        Discretize filter = new Discretize();
        filter.setInputFormat(data);
        Instances outputDataset;
        outputDataset = Filter.useFilter(data, filter);
        return outputDataset;
    }
    
    public static void main(String[] args) throws IOException, Exception {
         ConverterUtils.DataSource reader = new ConverterUtils.DataSource("C:\\Users\\Joshua\\Desktop\\Tubes2AI\\mush.arff");
         Instances instances = reader.getDataSet();
         //mengeset class ada di index brp indeks di awal 
         instances.setClassIndex(classIndex);
         Instances dataset = filter(instances);
         Classifier classifier = new NaiveBayesAI();
         classifier.buildClassifier(dataset);
         Evaluation evaluation = new Evaluation(dataset);
         evaluation.evaluateModel(classifier, dataset);
         System.out.println(evaluation.toSummaryString());
         System.out.println(evaluation.toClassDetailsString());
         System.out.println(evaluation.toMatrixString());
         weka.core.SerializationHelper.write("C:\\Users\\Joshua\\Desktop\\Tubes2AI\\output.model",classifier);
         System.out.println("File Eksternal output.model telah berhasil disimpan");
         System.out.println(" ");
         Classifier classifier2 = new NaiveBayesAI();
         classifier2 = (Classifier) weka.core.SerializationHelper.read( new FileInputStream("C:\\Users\\Joshua\\Desktop\\Tubes2AI\\output.model"));
         System.out.println("Model telah berhasil dibaca!");
         System.out.println(" ");
         
         Evaluation evaluation2 = new Evaluation(dataset);
         evaluation2.evaluateModel(classifier2, dataset);
         System.out.println(evaluation2.toSummaryString());
         System.out.println(evaluation2.toClassDetailsString());
         System.out.println(evaluation2.toMatrixString());
         
         
         System.out.println("Terima Kasih Telah Menggunakan Program Kami!");
         System.exit(0);
    }    
    
}
