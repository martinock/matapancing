/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package secondweka;

import java.io.Serializable;
import java.util.ArrayList;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;

/**
 *
 * @author asus
 */
public class FFNN extends AbstractClassifier {
    private boolean useNormalize;
    private boolean useHiddenLayer;
    private int nHiddenLayer;
    private int nHiddenNeuron;
    private int nOutputNeuron;
    private int numClass;
    private Double learningRate = 0.2;
    private MLP mlp;
    private Instances instances;
    private int nIterasi;
    private int classIndex;
    private double threshold = 0.0001;
    
    
    public Instances getInstances() {
        return this.instances;
    }
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.DATE_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
        
    }
    
    @Override
    public void buildClassifier(Instances inst) throws Exception {
        getCapabilities().testWithFail(inst);
        inst = new Instances(inst);
        inst.deleteWithMissingClass();
        
        numClass = inst.numClasses();
        if (useHiddenLayer) {
            System.out.println("Use Hidden Layer");
            nHiddenLayer = 1;
            nHiddenNeuron = 25;
        } else {
            System.out.println("Not use hidden Layer");
            nHiddenLayer = 0;
            nHiddenNeuron = 0;
        }
        if (numClass == 2) {
            nOutputNeuron = 1;
        } else {
            nOutputNeuron = inst.numClasses();
        }
        
        /* Filter */
        if (useNormalize) {
            System.out.println("Normalized.");
            Normalize filter1 = new Normalize();
            filter1.setInputFormat(inst);
            this.instances = Filter.useFilter(inst, filter1);
            Standardize filter2 = new Standardize();
            filter2.setInputFormat(this.instances);
            this.instances = Filter.useFilter(inst, filter2);
        } else {
            System.out.println("Unnormalized!");
            this.instances = new Instances(inst);    
        }        
        mlp = new MLP(this.instances.numAttributes() - 2, nHiddenLayer, nHiddenNeuron, 
                nOutputNeuron);
        for (int loop = 0; loop < nIterasi; loop++) {
            boolean isBreak = false;
            for (int i = 0; i < this.instances.numInstances(); i++) {               
                Instance current_instance = this.instances.instance(i);               
//                System.out.println(current_instance);
                ArrayList<Double> input = new ArrayList<>();
                double[] tes = current_instance.toDoubleArray();
//                for(double d : tes) System.out.print(d + " ");
//                System.out.println("");
                for (double d : tes) {
                    input.add(d);
                }
                input.remove(classIndex);
               //input.remove(classIndex);
//                System.out.println(input);
                double[] targetArr;
                if (numClass == 2) {
                    targetArr = new double[1];
                    for (int j = 0; j < 1; j++) {
                        if (current_instance.classValue() == (double) j) {
                            targetArr[j] = 1.0;
                        } else {
                            targetArr[j] = 0.0;
                        }
                    }
                } else {
                    targetArr = new double[numClass];
                    for (int j = 0; j < numClass; j++) {
                        if (current_instance.classValue() == (double) j) {
                            targetArr[j] = 1.0;
                        } else {
                            targetArr[j] = 0.0;
                        }
                    }
                }
                ArrayList<Double> target = new ArrayList<>();
                for(double d : targetArr) target.add(d);
                mlp.inputVal(input);    
                mlp.BackProp(target);
                for (int z = 0; z < mlp.ResultNeurons.size(); z++) {
                    if(mlp.ResultNeurons.get(z).GetError() > threshold || mlp.ResultNeurons.get(z).GetError() < -threshold ) {
                        isBreak = false;
                    }
                }
                mlp.UpdateWeight(learningRate);                
                if (isBreak) {
                    break;
                }
            }
            if (isBreak) {
                break;
            }
        }        
    }       
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {        
        ArrayList<Double> input = new ArrayList<>();
        double[] tes = instance.toDoubleArray();        
        for (double d : tes) {
            input.add(d);
        }
        input.remove(classIndex);
        //input.remove(classIndex);
//        System.out.println(input);
        mlp.inputVal(input);
        ArrayList<Double> result = mlp.getResArr();
//        System.out.println(result);
        if (result.size() == 1) {            
            if (result.get(0)*2 >= 0.5) {
                return 0.0;
            } else {
                return 1.0;
            }
        } else {
            int max = 0;
            for (int i = 1; i < result.size(); i++) {
                if (result.get(max) < result.get(i)) {
                    max = i;                    
                }                
            }
            return (double) max;
        }
    }
    
    public void setUseHiddenLayer(boolean x) {
        useHiddenLayer = x;
    }
    
    public void setLearningRate(Double x) {
        learningRate = x;
    }
    
    public void setNIterasi(int x) {
        nIterasi = x;
    }
    
    public void setUseNormalize(boolean x) {
        useNormalize = x;
    }
    
    public void setClassIndex(int x) {
        classIndex = x;
    }       
}
