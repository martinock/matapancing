/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package secondweka;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import java.util.Scanner;
import tubesaiweka.NaiveBayesAI;
import static tubesaiweka.TubesAIWEKA.filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;

/**
 *
 * @author asus
 */
public class SecondWeka {

    static int classIndex;
    
    public static Instances filter(Instances data) throws Exception{
        Discretize filter = new Discretize();
        filter.setInputFormat(data);
        Instances outputDataset;
        outputDataset = Filter.useFilter(data, filter);
        return outputDataset;
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        System.out.println("====================================================");
        System.out.println("                       WELCOME                      ");
        System.out.println("====================================================");
        System.out.print("Masukkan nama dataset di Desktop : ");
        Scanner in = new Scanner(System.in);
        String filename = in.next();
        ConverterUtils.DataSource reader = new ConverterUtils.DataSource("/home/martinock/Desktop/" + filename + ".arff");
        //ConverterUtils.DataSource reader = new ConverterUtils.DataSource("C:\\Users\\asus\\Desktop\\" + filename + ".arff");
        Instances instances = reader.getDataSet();
        //mengeset class ada di index brp indeks di awal
        System.out.println("Pilih metode pembelejaran : ");
        System.out.println("1. FFNN");
        System.out.println("2. Naive Bayes");
        System.out.print("Pilihan Anda : ");
        int pil = in.nextInt();
        FFNN ffnn;
        NaiveBayesAI naiveBayes;
        do {
            if (pil == 1) {
                ffnn = new FFNN();
                System.out.print("Masukkan index kelas : ");
                classIndex = in.nextInt();
                instances.setClassIndex(classIndex);
                ffnn.setClassIndex(classIndex);        
                System.out.print("Masukkan learning rate : ");
                Double lr = in.nextDouble();
                ffnn.setLearningRate(lr);
                System.out.print("Masukkan jumlah iterasi : ");
                int iterasi = in.nextInt();
                ffnn.setNIterasi(iterasi);
                System.out.println("Apakah ingin menggunakan hidden layer?(y/t)");
                String ans = in.next();
                do {
                    if (ans.equalsIgnoreCase("y")) {
                        ffnn.setUseHiddenLayer(true);
                    } else if (ans.equalsIgnoreCase("t")) {
                        ffnn.setUseHiddenLayer(false);
                    } else {
                        System.out.println("Masukan salah! Ulangi");
                    }
                } while (!ans.equalsIgnoreCase("y") && !ans.equalsIgnoreCase("t"));
                System.out.println("Apakah ingin menggunakan Normalize? (y/t)");
                ans = in.next();
                do {
                    if (ans.equalsIgnoreCase("y")) {
                        ffnn.setUseNormalize(true);
                    } else if (ans.equalsIgnoreCase("t")) {
                        ffnn.setUseNormalize(false);
                    } else {
                        System.out.println("Masukan salah! Ulangi");
                    }
                } while (!ans.equalsIgnoreCase("y") && !ans.equalsIgnoreCase("t"));
                ffnn.buildClassifier(instances);
                //Split test
                /*int percent = 60;
                int trainsize = (int) Math.round(ffnn.getInstances().numInstances() * percent / 100);
                int testsize = ffnn.getInstances().numInstances() - trainsize;
                Instances train = new Instances(ffnn.getInstances(), 0, trainsize);
                Instances test = new Instances(ffnn.getInstances(),trainsize,testsize);
                Evaluation evaluation = new Evaluation(train);
                evaluation.evaluateModel(ffnn,test);*/
                
                
                Evaluation evaluation = new Evaluation(ffnn.getInstances());
                
                //10fold
                evaluation.crossValidateModel(ffnn, ffnn.getInstances(), 10, new Random(1));
                
                //Full training
                //evaluation.evaluateModel(ffnn, ffnn.getInstances());
                System.out.println(evaluation.toSummaryString());
                System.out.println(evaluation.toClassDetailsString());
                System.out.println(evaluation.toMatrixString());
                weka.core.SerializationHelper.write("/home/martinock/Desktop/ffnn.model" ,ffnn);
                //weka.core.SerializationHelper.write("C:\\Users\\asus\\Desktop\\ffnn.model" ,ffnn);
                System.out.println("File Eksternal output.model telah berhasil disimpan");
                System.out.println(" ");
                Classifier classifier2 = new FFNN();
                classifier2 = (Classifier) weka.core.SerializationHelper.read( new FileInputStream("/home/martinock/Desktop/ffnn.model"));
                //classifier2 = (Classifier) weka.core.SerializationHelper.read( new FileInputStream("C:\\Users\\asus\\Desktop\\ffnn.model"));
                System.out.println("Model telah berhasil dibaca!");
                System.out.println(" ");
                
                System.out.print("Masukan nama file test : ");
                filename = in.next();
                System.out.println(filename);
                ConverterUtils.DataSource reader2 = new ConverterUtils.DataSource("/home/martinock/Desktop/" + filename + ".arff");
                //ConverterUtils.DataSource reader2 = new ConverterUtils.DataSource("C:\\Users\\asus\\Desktop\\" + filename + ".arff");
                Instances instances2 = reader2.getDataSet();
                instances2.setClassIndex(classIndex);
                Normalize filter1 = new Normalize();
                filter1.setInputFormat(instances2);
                Instances newInstances = Filter.useFilter(instances2, filter1);
                Standardize filter2 = new Standardize();
                filter2.setInputFormat(newInstances);
                Instances newInstances2 = Filter.useFilter(newInstances, filter2);
                Evaluation evaluation2 = new Evaluation(newInstances2);
                evaluation2.evaluateModel(classifier2, newInstances2);
                System.out.println(evaluation2.toSummaryString());
                System.out.println(evaluation2.toClassDetailsString());
                System.out.println(evaluation2.toMatrixString());
            } else if (pil == 2) {
                System.out.print("Masukkan index kelas : ");
                classIndex = in.nextInt();
                instances.setClassIndex(classIndex);
                if (classIndex == 26) {
                    instances.remove(27);
                }
                if (classIndex == 27) {
                    instances.remove(26);
                }
                Instances dataset = filter(instances);         
                naiveBayes = new NaiveBayesAI();                
                naiveBayes.buildClassifier(dataset);
                //Evaluation evaluation = new Evaluation(dataset);
                int percent = 60;
                int trainsize = (int) Math.round(dataset.numInstances() * percent / 100);
                int testsize = dataset.numInstances() - trainsize;
                Instances train = new Instances(dataset, 0, trainsize);
                Instances test = new Instances(dataset,trainsize,testsize);
                Evaluation evaluation = new Evaluation(train);
                evaluation.evaluateModel(naiveBayes,test);
                //evaluation.evaluateModel(naiveBayes, dataset);
                //evaluation.crossValidateModel(naiveBayes, dataset, 10, new Random(1));
                System.out.println(evaluation.toSummaryString());
                System.out.println(evaluation.toClassDetailsString());
                System.out.println(evaluation.toMatrixString());
                weka.core.SerializationHelper.write("/home/martinock/Desktop/bayes.model",naiveBayes);
                //weka.core.SerializationHelper.write("C:\\Users\\asus\\Desktop\\bayes.model",naiveBayes);
                System.out.println("File Eksternal output.model telah berhasil disimpan");
                System.out.println(" ");
                Classifier classifier2 = new NaiveBayesAI();
                classifier2 = (Classifier) weka.core.SerializationHelper.read( new FileInputStream("/home/martinock/Desktop/bayes.model"));
                //classifier2 = (Classifier) weka.core.SerializationHelper.read( new FileInputStream("C:\\Users\\asus\\Desktop\\bayes.model"));
                System.out.println("Model telah berhasil dibaca!");
                System.out.println(" ");
                
                System.out.print("Masukan nama file test : ");
                filename = in.next();
                System.out.println(filename);
                ConverterUtils.DataSource reader2 = new ConverterUtils.DataSource("/home/martinock/Desktop/" + filename + ".arff");
                //ConverterUtils.DataSource reader2 = new ConverterUtils.DataSource("C:\\Users\\asus\\Desktop\\" + filename + ".arff");
                Instances instances2 = reader2.getDataSet();
                instances2.setClassIndex(classIndex);
                Instances newInstances = filter(instances2);
                Evaluation evaluation2 = new Evaluation(newInstances);
                evaluation2.evaluateModel(classifier2, newInstances);
                System.out.println(evaluation2.toSummaryString());
                System.out.println(evaluation2.toClassDetailsString());
                System.out.println(evaluation2.toMatrixString());
            } else {
                System.out.println("Pilihan salah, ulangi");
            }
        } while (pil != 1 && pil != 2);
        System.out.println("Terima Kasih Telah Menggunakan Program Kami!");
        System.exit(0);
    }            
}
