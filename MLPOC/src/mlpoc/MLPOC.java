/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mlpoc;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

/**
 *
 * @author Jagan
 */
public class MLPOC {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        try {
            // TODO code application logic here
            BufferedReader br;
            br=new BufferedReader(new FileReader("D:/Extra/B.E Project/agrodeploy/webapp/Data/ClusterAutotrain12.arff"));
            Instances training_data = new Instances(br);
            br.close();
            training_data.setClassIndex(training_data.numAttributes() - 1);
            br=new BufferedReader(new FileReader("D:/Extra/B.E Project/agrodeploy/webapp/Data/TestFinal.arff"));
            Instances testing_data = new Instances(br);
            br.close();
            testing_data.setClassIndex(testing_data.numAttributes() - 1);
            String summary = training_data.toSummaryString();
            int number_samples = training_data.numInstances();
            int number_attributes_per_sample = training_data.numAttributes();
            System.out.println("Number of attributes in model = " + number_attributes_per_sample);
            System.out.println("Number of samples = " + number_samples);
            System.out.println("Summary: " + summary);
            System.out.println();
            
            J48 j48 = new J48();
            FilteredClassifier fc = new FilteredClassifier();
            fc.setClassifier(j48);
            fc.buildClassifier(training_data);
            System.out.println("Testing instances: "+testing_data.numInstances());
            for (int i = 0; i < testing_data.numInstances(); i++) 
            {
                double pred = fc.classifyInstance(testing_data.instance(i));
                String s1=testing_data.classAttribute().value((int) pred);
                System.out.println(testing_data.instance(i)+" Predicted value: " + s1);
            }
            Evaluation crossValidate = crossValidate("D:/Extra/B.E Project/agrodeploy/webapp/Data/ClusterAutotrain12.arff");    
            
            DataSource source = new DataSource("D:/Extra/B.E Project/agrodeploy/webapp/Data/ClusterAutotrain12.arff");
            Instances data = source.getDataSet();
            System.out.println(data.numInstances()); 
            data.setClassIndex(data.numAttributes() - 1);

            // 1. meta-classifier
            useClassifier(data);

            // 2. filter
            useFilter(data);
        } catch (Exception ex) {
            Logger.getLogger(MLPOC.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * uses the meta-classifier
     */
    protected static void useClassifier(Instances data) throws Exception {
      System.out.println("\n1. Meta-classfier");
      AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
      CfsSubsetEval eval = new CfsSubsetEval();
      GreedyStepwise search = new GreedyStepwise();
      search.setSearchBackwards(true);
      J48 base = new J48();
      classifier.setClassifier(base);
      classifier.setEvaluator(eval);
      classifier.setSearch(search);
      Evaluation evaluation = new Evaluation(data);
      evaluation.crossValidateModel(classifier, data, 10, new Random(1));
      System.out.println(evaluation.toSummaryString());
    }

    /**
     * uses the filter
     */
    protected static void useFilter(Instances data) throws Exception {
      System.out.println("\n2. Filter");
      weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
      CfsSubsetEval eval = new CfsSubsetEval();
      GreedyStepwise search = new GreedyStepwise();
      search.setSearchBackwards(true);
      filter.setEvaluator(eval);
      filter.setSearch(search);
      filter.setInputFormat(data);
      Instances newData = Filter.useFilter(data, filter);
      System.out.println(newData);
    }

    public static Evaluation crossValidate(String filename){
       Evaluation eval=null;
        try {
            BufferedReader br=new BufferedReader(
                              new FileReader(filename));
            // loads data and set class index
            Instances data = new Instances(br);
            br.close();
            /*File csv=new File(filename);
            CSVLoader loader = new CSVLoader();
            loader.setSource(csv);
            Instances data = loader.getDataSet();*/
            data.setClassIndex(data.numAttributes() - 1);

            // classifier
            String[] tmpOptions;
            String classname="weka.classifiers.trees.J48 -C 0.25";
            tmpOptions     = classname.split(" ");
            classname      = "weka.classifiers.trees.J48";
            tmpOptions[0]  = "";
            Classifier cls = (Classifier) Utils.forName(Classifier.class, classname, tmpOptions);

            // other options
            int seed  = 2;
            int folds = 10;

            // randomize data
            Random rand = new Random(seed);
            Instances randData = new Instances(data);
            randData.randomize(rand);
            if (randData.classAttribute().isNominal())
              randData.stratify(folds);

            // perform cross-validation
           eval = new Evaluation(randData);
            for (int n = 0; n < folds; n++) {
              Instances train = randData.trainCV(folds, n);
              Instances test = randData.testCV(folds, n);
              // the above code is used by the StratifiedRemoveFolds filter, the
              // code below by the Explorer/Experimenter:
              // Instances train = randData.trainCV(folds, n, rand);

              // build and evaluate classifier
              Classifier clsCopy = Classifier.makeCopy(cls);
              clsCopy.buildClassifier(train);
              eval.evaluateModel(clsCopy, test);
            }

            // output evaluation
            System.out.println();
            System.out.println("=== Setup ===");
            System.out.println("Classifier: " + cls.getClass().getName() + " " + Utils.joinOptions(cls.getOptions()));
            System.out.println("Dataset: " + data.relationName());
            System.out.println("Folds: " + folds);
            System.out.println("Seed: " + seed);
            System.out.println();
            System.out.println(eval.toSummaryString("Summary for testing", true));
            System.out.println("Correctly Classified Instances: "+eval.correct());
            System.out.println("Percentage of Correctly Classified Instances: "+eval.pctCorrect());
            System.out.println("InCorrectly Classified Instances: "+eval.incorrect());
            System.out.println("Percentage of InCorrectly Classified Instances: "+eval.pctIncorrect());
            
        } catch (Exception ex) {
            System.err.println(ex.getMessage());
        }
        return eval;
  }
}
