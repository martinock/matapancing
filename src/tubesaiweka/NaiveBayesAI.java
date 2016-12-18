    /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tubesaiweka;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * Kelompok Mata Pancing AI - 13514001, 13514048, 13514055, 13514084
 */
public class NaiveBayesAI extends AbstractClassifier{

    private double [][][] probability;
    private int [][][] frequency;
    private int[] nValueClass;
    private double[] nProbClass;
    static int classIndex;
    int nRow = 0;
    int numClass;
    int numAttribute;
    
    @Override
    public Capabilities getCapabilities() {
      Capabilities result = super.getCapabilities();
      result.disableAll();
      //attributes
      result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
      result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
      result.enable( Capabilities.Capability.MISSING_VALUES );
      //class
      result.enable(Capabilities.Capability.NOMINAL_CLASS);
      result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
      //instances
      result.setMinimumNumberInstances(0);
      return result;
    }
    
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        //Mengecek apakah data bisa di handle (tidak ada error)
        getCapabilities().testWithFail(instances);
        //Menghapus instance yang tidak ada kelasnya 
        instances.deleteWithMissingClass();
        //inisialisasi variable
        classIndex = instances.classIndex();
        int numInstances = instances.numInstances();
        numAttribute = instances.numAttributes();
        numClass = instances.numClasses();
        frequency = new int[numAttribute][][];
        probability = new double[numAttribute][][];
        //init table assign 0 ke semua
        for(int i=0 ; i < numAttribute ; i++){
            frequency[i] = new int[instances.attribute(i).numValues()][];
            probability[i] = new double[instances.attribute(i).numValues()][];
            for(int j=0 ; j < instances.attribute(i).numValues() ; j++){
                frequency[i][j] = new int[numClass];
                probability[i][j] = new double[numClass];
                for(int k=0 ; k < numClass ; k++){
                    frequency[i][j][k] = 0;
                    probability[i][j][k] = 0;
                }
            }
        }
        //memasukan nilai frekuensi ke dalam tabel 
        for(int i = 0; i < numInstances ;i++){
            Instance currentInstance = instances.instance(i);
            for(int j = 0; j < numAttribute ; j++){
                frequency[j][(int)currentInstance.value(j)][(int)currentInstance.value(classIndex)]++;
            }
        }
        //init array value class  
        nValueClass = new int[numClass];
        for(int i=0; i < numClass ; ++i){
           nValueClass[i] = frequency[classIndex][i][i]; 
        }        
        //init array prob class  
        nProbClass = new double[numClass];
        for(int i=0; i < numClass ; ++i){
           nProbClass[i] = ((double)nValueClass[i])/numInstances; 
        }
        //mengisi tabel probabilitas naive bayes
        for(int i = 0; i < numAttribute ;i++){            
            for(int j=0 ; j < instances.attribute(i).numValues() ; j++){
                for(int k=0 ; k < numClass ; k++){
                    if(i!=classIndex){
                        double temp = (double)frequency[i][j][k];
                        double valueclass = (double) nValueClass[k];
                        probability[i][j][k] = temp/valueclass;  
                    }
                    else{
                        probability[i][j][k] = (double)frequency[i][j][k]/numInstances;
                    }                    
                }              
            }
        }
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] pResult = new double[numClass]; 
        //assign array dengan nilai 0 terlebih dahulu
        for(int i = 0; i < numClass; ++i){
            pResult[i] = 1;
        }
        for(int i = 0; i < numAttribute; ++i){
            if (i == classIndex){
                continue;
            }
            for(int j = 0; j < numClass; ++j){
                int attribute = (int) instance.value(i);
                    pResult[j] = pResult[j] * probability[i][attribute][j];    
            }
        }
        for(int i = 0; i < numClass; ++i){
            pResult[i] = pResult[i] * nProbClass[i];
        }
        return pResult;
    }
    
       @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] pResult = new double[numClass]; 
        //assign array dengan nilai 0 terlebih dahulu
        for(int i = 0; i < numClass; ++i){
            pResult[i] = 1;
        }
        for(int i = 0; i < numAttribute; ++i){
            if (i == classIndex){
                continue;
            }
            for(int j = 0; j < numClass; ++j){
                int attribute = (int) instance.value(i);
                    pResult[j] = pResult[j] * probability[i][attribute][j];
            }
        }
        for(int i = 0; i < numClass; ++i){
            pResult[i] = pResult[i] * nProbClass[i];
        }
        double max = pResult[0];
        double maxIndex = 0;
        for(int i = 1; i < numClass; ++i){
            if(pResult[i] > max){
                max = pResult[1];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    
}
