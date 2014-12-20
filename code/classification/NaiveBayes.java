package classification;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;


public class NaiveBayes {

    private static int max_attributes;
    private static ArrayList<Integer> training_labels;
    private static ArrayList<HashMap<Integer,Integer>> training_data;
    private static ArrayList<Integer> test_labels;
    private static ArrayList<HashMap<Integer,Integer>> test_data;
    private static HashMap<Integer, HashMap<Integer,HashMap<Integer,Double>>> classifier;
    private static HashMap<Integer,Integer> training_labels_frequency;
    private static ArrayList<Integer> predicted_labels;
    private static HashMap<Integer,ArrayList<Integer>> attributes_unique_values;

    private static void findMaxAttributes(String training_file, String test_file)
    {
        attributes_unique_values = new HashMap<Integer, ArrayList<Integer>>();
        try {

            //finding max in training file
            BufferedReader br = new BufferedReader(new FileReader(training_file));
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.equals("")) {

                    String[] temp = line.split(" ");
                    for (int i = 1; i < temp.length; i++) {
                        int attr = Integer.parseInt(temp[i].split(":")[0]);
                        int val = Integer.parseInt(temp[i].split(":")[1]);
                        if (attr > max_attributes)
                            max_attributes = attr;

                        if(!attributes_unique_values.containsKey(attr))
                        {
                            ArrayList<Integer> values = new ArrayList<Integer>();
                            values.add(val);
                            attributes_unique_values.put(attr,values);
                        }
                        else
                        {
                            if(!attributes_unique_values.get(attr).contains(val))
                            {
                                attributes_unique_values.get(attr).add(val);
                            }
                        }


                    }
                }
            }

            //finding max in test file
            br = new BufferedReader(new FileReader(test_file));
            while ((line = br.readLine()) != null) {
                if (!line.equals("")) {

                    String[] temp = line.split(" ");
                    for (int i = 1; i < temp.length; i++) {
                        int attr = Integer.parseInt(temp[i].split(":")[0]);
                        int val = Integer.parseInt(temp[i].split(":")[1]);
                        if (attr > max_attributes)
                            max_attributes = attr;

                        if(!attributes_unique_values.containsKey(attr))
                        {
                            ArrayList<Integer> values = new ArrayList<Integer>();
                            values.add(val);
                            values.add(0);
                            attributes_unique_values.put(attr,values);
                        }
                        else
                        {
                            if(!attributes_unique_values.get(attr).contains(val))
                            {
                                attributes_unique_values.get(attr).add(val);
                            }
                        }
                    }
                }
            }

        }
        catch(Exception ex)
        {
            System.out.println(ex);
        }
    }

    private static void load_data(String training_file, String test_file)
    {
        training_labels = new ArrayList<Integer>();
        training_data = new ArrayList<HashMap<Integer, Integer>>();
        test_labels = new ArrayList<Integer>();
        test_data = new ArrayList<HashMap<Integer, Integer>>();

        try {

            //Load training data
            BufferedReader br = new BufferedReader(new FileReader(training_file));
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.equals("")) {

                    String[] temp = line.split(" ");

                    if(temp[0].equals("+1"))
                        temp[0] = "1";

                    int label = Integer.parseInt(temp[0]);

                    //Storing training class labels
                    training_labels.add(label);

                    HashMap<Integer, Integer> attribute_values = new HashMap<Integer, Integer>();
                    for (int i = 1; i < temp.length; i++) {
                        int attr = Integer.parseInt(temp[i].split(":")[0]);
                        int val = Integer.parseInt(temp[i].split(":")[1]);
                        attribute_values.put(attr,val);
                    }

                    for(int i=1; i<=max_attributes; i++)
                    {
                        if(!attribute_values.containsKey(i))
                        {
                            attribute_values.put(i,0);
                        }
                    }
                    training_data.add(attribute_values);
                }
            }

            //Load test data
            br = new BufferedReader(new FileReader(test_file));
            while ((line = br.readLine()) != null) {
                if (!line.equals("")) {

                    String[] temp = line.split(" ");

                    if(temp[0].equals("+1"))
                        temp[0] = "1";

                    int label = Integer.parseInt(temp[0]);

                    //Storing test class labels
                    test_labels.add(label);

                    HashMap<Integer, Integer> attribute_values = new HashMap<Integer, Integer>();
                    for (int i = 1; i < temp.length; i++) {
                        int attr = Integer.parseInt(temp[i].split(":")[0]);
                        int val = Integer.parseInt(temp[i].split(":")[1]);
                        attribute_values.put(attr,val);
                    }

                    for(int i=1; i<=max_attributes; i++)
                    {
                        if(!attribute_values.containsKey(i))
                        {
                            attribute_values.put(i,0);
                            if(!attributes_unique_values.get(i).contains(0))
                                attributes_unique_values.get(i).add(0);
                        }
                    }

                    test_data.add(attribute_values);
                }
            }

        }
        catch(Exception ex)
        {
            System.out.println(ex);
        }
    }

    private static void buildClassifier()
    {
        classifier = new HashMap<Integer,HashMap<Integer, HashMap<Integer, Double>>>();
        training_labels_frequency = new HashMap<Integer, Integer>();
        for(int i=0; i < training_data.size(); i++) {
            //Count the frequency of occurrence of each class
            int label = training_labels.get(i);
            if (training_labels_frequency.containsKey(label)) {
                int val = training_labels_frequency.get(label);
                training_labels_frequency.put(label, ++val);
            } else {
                training_labels_frequency.put(label, 1);
            }

            for (Integer attr : training_data.get(i).keySet()) {
                int val = training_data.get(i).get(attr);
                if (!classifier.containsKey(attr)) {
                    HashMap<Integer, HashMap<Integer, Double>> attr_values = new HashMap<Integer, HashMap<Integer, Double>>();
                    HashMap<Integer, Double> values_class_frequency = new HashMap<Integer, Double>();
                    values_class_frequency.put(label, 1.0);
                    attr_values.put(val, values_class_frequency);
                    classifier.put(attr, attr_values);
                } else {
                    if (!classifier.get(attr).containsKey(val)) {
                        HashMap<Integer, Double> values_class_frequency = new HashMap<Integer, Double>();
                        //values_class_frequency.put(label,1.0);
                        values_class_frequency.put(label, 1.0);
                        classifier.get(attr).put(val, values_class_frequency);
                    } else {
                        if (!classifier.get(attr).get(val).containsKey(label)) {
                            //attribute_class_frequency.get(attr).get(val).put(label,1.0);
                            classifier.get(attr).get(val).put(label, 1.0);
                        } else {
                            double count = classifier.get(attr).get(val).get(label);
                            classifier.get(attr).get(val).put(label, ++count);
                        }
                    }
                }


            }
        }

        //adding labels which might not have occurred for a particular attribute-value pair
        for(Integer attribute : classifier.keySet())
        {
            for(Integer value : classifier.get(attribute).keySet())
            {
                for(Integer class_label : training_labels_frequency.keySet())
                {
                    if(!classifier.get(attribute).get(value).containsKey(class_label)) {
                        classifier.get(attribute).get(value).put(class_label, 0.0);
                    }
                }
            }
        }
    }

    public static void generate_rule_measures(ArrayList<Integer> labels, ArrayList<Integer> predicted)
    {
        int true_positive = 0;
        int true_negative = 0;
        int false_positive = 0;
        int false_negative = 0;
        for(int i = 0; i < labels.size(); i++)
        {
            int actual_label = labels.get(i);
            int predicted_label = predicted_labels.get(i);
            if(actual_label == 1 && predicted_label == 1)
                true_positive++;
            else if(actual_label == 1 && predicted_label == -1 )
                false_negative++;
            else if(actual_label == -1 && predicted_label == 1)
                false_positive++;
            else if(actual_label == -1 && predicted_label == -1)
                true_negative++;
        }

        System.out.println(true_positive + " " + false_negative + " " + false_positive + " " + true_negative);

    }

    private static void predict(ArrayList<HashMap<Integer,Integer>> data, boolean applyLaplace)
    {
        predicted_labels = new ArrayList<Integer>();

        for(int i=0; i < data.size(); i++)
        {
            double max_probability = 0.0;
            int predicted_label = 0;

            for (Integer class_label : training_labels_frequency.keySet())
            {
                double conditional_probability = 1.0;
                int class_frequency = training_labels_frequency.get(class_label);
                for (Integer attr : data.get(i).keySet())
                {
                    int value = data.get(i).get(attr);

                    double attr_value_class_frequency;
                    if(classifier.get(attr).containsKey(value))
                    {
                        attr_value_class_frequency = classifier.get(attr).get(value).get(class_label);
                        if(applyLaplace)
                        {
                            attr_value_class_frequency++;
                        }
                    }
                    else
                    {
                        if(applyLaplace) {
                            attr_value_class_frequency = 1.0;
                        }
                        else
                            attr_value_class_frequency = 0.0;
                    }

                    if(!applyLaplace)
                    {
                        conditional_probability = conditional_probability * (attr_value_class_frequency / (class_frequency * 1.0));
                    }
                    else {
                        int freq = class_frequency + attributes_unique_values.get(attr).size();
                        conditional_probability = conditional_probability * (attr_value_class_frequency / (freq * 1.0));
                    }
                }

                double current_probability = conditional_probability * (training_labels_frequency.get(class_label) / (training_data.size() * 1.0));
                if (max_probability < current_probability) {
                    max_probability = current_probability;
                    predicted_label = class_label;
                }
            }

            predicted_labels.add(predicted_label);
        }

    }

    private static void naive_bayes(String training_file, String test_file)
    {
        //Step 1: Find Maximum Attributes
        findMaxAttributes(training_file,test_file);

        //Step 2: Load the training and test data into memory
        load_data(training_file,test_file);

        //Step3: Build Naive Bayes Classifier
        buildClassifier();

        //Step4: Predict on training data Using the Classifier
        predict(training_data,true);

        //Step 5: Generate Rule measures for prediction on training data
        generate_rule_measures(training_labels,predicted_labels);

        //Step6: Predict on test data Using the Classifier
        predict(test_data,true);

        //Step 7: Generate Rule measures for prediction on test data
        generate_rule_measures(test_labels,predicted_labels);
    }

    public static void main(String[] args)
    {
        String training_file = args[0];
        String test_file = args[1];

        naive_bayes(training_file,test_file);

    }
}

