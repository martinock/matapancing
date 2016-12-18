/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package secondweka;

import java.io.Serializable;
import static java.lang.Math.*;
import java.util.ArrayList;
import weka.core.Instance;

/**
 *
 * @author asus
 */
class Neuron implements Serializable {

    private double Output;
    private double Error;

    public Neuron() {
        Output = 0;
        Error = 0;
    }

    public void SetOutput(double O) {
        Output = O;
    }

    public double GetOutput() {
        return Output;
    }

    public void SetError(double E) {
        Error = E;
    }

    public double GetError() {
        return Error;
    }

}

class Edge implements Serializable {

    private Neuron N1;
    private Neuron N2;
    private double weight;

    public Edge(Neuron N1, Neuron N2) {
        this.N1 = N1;
        this.N2 = N2;
        weight = random();
    }
        
    public Neuron getN1() {
        return N1;
    }

    public Neuron getN2() {
        return N2;
    }
    public double getWeight() {
        return weight;
    }
    public void setWeight(double x) {
        weight = x;
    }
}

public class MLP  implements Serializable {

    private ArrayList<Edge> ListOfEdge1;
    private ArrayList<Edge> ListOfEdge2;
    public ArrayList<Neuron> InputNeurons;
    public ArrayList<Neuron> HiddenNeurons;
    public ArrayList<Neuron> ResultNeurons;

    public MLP(int input, int hlayer, int hneuron, int output) {
        ListOfEdge1 = new ArrayList<>();
        ListOfEdge2 = new ArrayList<>();
        InputNeurons = new ArrayList<>();
        HiddenNeurons = new ArrayList<>();
        ResultNeurons = new ArrayList<>();


        for (int i = 0; i < input; i++) {
            InputNeurons.add(i, new Neuron());
        }
        for (int i = 0; i < output; i++) {
            ResultNeurons.add(i, new Neuron());
        }
        if (hlayer == 1) {
            for (int i = 0; i < hneuron; i++) {
                HiddenNeurons.add(i, new Neuron());
                for (int k = 0; k < InputNeurons.size(); k++) {
                    ListOfEdge1.add(new Edge(InputNeurons.get(k), HiddenNeurons.get(i)));
                }
                for (int k = 0; k < ResultNeurons.size(); k++) {
                    ListOfEdge2.add(new Edge(HiddenNeurons.get(i), ResultNeurons.get(k)));
                }
            }
        } else {
            for (int k = 0; k < ResultNeurons.size(); k++) {
                for (int l = 0; l < InputNeurons.size(); l++) {
                    ListOfEdge1.add(new Edge(InputNeurons.get(l), ResultNeurons.get(k)));
                }
            }
        }
    }
    
    public double getWeight(Neuron A, Neuron B){
        for(int i = 0; i < ListOfEdge1.size(); i++){
            if ((A == ListOfEdge1.get(i).getN1() && B == ListOfEdge1.get(i).getN2())||(B == ListOfEdge1.get(i).getN1() && A == ListOfEdge1.get(i).getN2())){
                return ListOfEdge1.get(i).getWeight();
            }
        }
        for(int i = 0; i < ListOfEdge2.size(); i++){
            if ((A == ListOfEdge2.get(i).getN1() && B == ListOfEdge2.get(i).getN2())||(B == ListOfEdge2.get(i).getN1() && A == ListOfEdge2.get(i).getN2())){
                return ListOfEdge2.get(i).getWeight();
            }
        }
        return 0;
    }

    public double getSigmoid(double a){
        double d = 1 + exp(a*(-1));
        return 1/d;
    }
    
    public void inputVal(ArrayList<Double> x){
        for(int i = 0; i < InputNeurons.size(); ++i){
            InputNeurons.get(i).SetOutput(x.get(i));
        }
        /*for(int i = 0; i < HiddenNeurons.size(); ++i){
            HiddenNeurons.get(i).SetOutput(0);
            HiddenNeurons.get(i).SetError(0);
        }
        for(int i = 0; i < ResultNeurons.size(); ++i){
            ResultNeurons.get(i).SetOutput(0);
            ResultNeurons.get(i).SetError(0);         
        }*/       
        for(int i = 0; i < ListOfEdge1.size(); ++i){
            ListOfEdge1.get(i).getN2().SetOutput(ListOfEdge1.get(i).getN2().GetOutput() + ListOfEdge1.get(i).getWeight()*ListOfEdge1.get(i).getN1().GetOutput());            
        }
        for(int i = 0; i < HiddenNeurons.size(); ++i){
            HiddenNeurons.get(i).SetOutput(getSigmoid(HiddenNeurons.get(i).GetOutput()));            
        }
        for(int i = 0; i < ListOfEdge2.size(); ++i){
            ListOfEdge2.get(i).getN2().SetOutput(ListOfEdge2.get(i).getN2().GetOutput() + ListOfEdge2.get(i).getWeight()*ListOfEdge2.get(i).getN1().GetOutput());
        }
        for(int i = 0; i < ResultNeurons.size(); ++i){
            ResultNeurons.get(i).SetOutput(getSigmoid(ResultNeurons.get(i).GetOutput()));
        }
    }
    
    public double getRes(){
        return ResultNeurons.get(0).GetOutput();
    }
    
    public ArrayList<Double> getResArr(){
        ArrayList<Double> a = new ArrayList<>();
        for (int i = 0; i < ResultNeurons.size(); i++){
            //System.out.println(ResultNeurons.get(i).GetOutput());
            a.add(ResultNeurons.get(i).GetOutput());
        }
        return a;
    }
    
    public void BackProp(ArrayList<Double> target){
        for (int i = 0; i < ResultNeurons.size(); i++){
            double x = ResultNeurons.get(i).GetOutput() * (1 - ResultNeurons.get(i).GetOutput()) * (target.get(i) - ResultNeurons.get(i).GetOutput());
            ResultNeurons.get(i).SetError(x);
        }
        for (int i = 0; i < ListOfEdge2.size(); i++){
            ListOfEdge2.get(i).getN1().SetError( ListOfEdge2.get(i).getN1().GetError() + ListOfEdge2.get(i).getWeight() * ListOfEdge2.get(i).getN2().GetError() );
        }
        for (int i = 0; i < HiddenNeurons.size(); i++){
            HiddenNeurons.get(i).SetError(HiddenNeurons.get(i).GetError() * HiddenNeurons.get(i).GetOutput() * (1 - HiddenNeurons.get(i).GetOutput()));
        }
    }
    
    public void UpdateWeight(Double lr){
        for (int i = 0; i < ListOfEdge1.size(); i++){
            ListOfEdge1.get(i).setWeight(ListOfEdge1.get(i).getWeight() + lr * (ListOfEdge1.get(i).getN2().GetError() * ListOfEdge1.get(i).getN1().GetOutput()));            
        }
        for (int i = 0; i < ListOfEdge2.size(); i++){
            ListOfEdge2.get(i).setWeight(ListOfEdge2.get(i).getWeight() + lr * (ListOfEdge2.get(i).getN2().GetError() * ListOfEdge2.get(i).getN1().GetOutput()));
        }
    }
    
    public ArrayList<Edge> getList1() {
        return ListOfEdge1;
    }
    
    public ArrayList<Edge> getList2() {
        return ListOfEdge2;
    }
}
