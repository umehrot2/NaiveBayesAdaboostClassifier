package classification;

import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by uditmehrotra on 13/11/14.
 */
public class NBAdaBoost
{

    private static ArrayList<HashMap<Integer,Integer>> training_data;
    private static HashMap<Integer, Double> tuple_weights;
    private static ArrayList<HashMap<Integer,Integer>> sample_data;
    private static ArrayList<Integer> sample_data_labels;
    private static ArrayList<HashMap<Integer,Integer>> test_data;
    private static ArrayList<Integer> training_labels;
    private static ArrayList<Integer> test_labels;
    private static int max_attributes;
    private static HashMap<Integer, HashMap<Integer,HashMap<Integer,Double>>> classifier;
    private static HashMap<Integer,Integer> sample_labels_frequency;
    private static ArrayList<Integer> predicted_labels;
    private static HashMap<Integer,ArrayList<Integer>> sample_attribute_unique_values;

    private static void findMaxAttributes(String training_file, String test_file)
    {
        try {

            //finding max attribute in training file
            BufferedReader br = new BufferedReader(new FileReader(training_file));
            String line;
            while ((line = br.readLine()) != null) {
                if (!line.equals("")) {

                    String[] temp = line.split(" ");
                    for (int i = 1; i < temp.length; i++) {
                        int attr = Integer.parseInt(temp[i].split(":")[0]);
                        if (attr > max_attributes)
                            max_attributes = attr;

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
                        if (attr > max_attributes)
                            max_attributes = attr;

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


    private static void get_sample()
    {
        sample_data = new ArrayList<HashMap<Integer, Integer>>();
        sample_data_labels = new ArrayList<Integer>();

        ArrayList<Double> weights = new ArrayList<Double>();
        for(int i=0; i < training_data.size(); i++)
        {
            if(i == 0)
                weights.add(tuple_weights.get(i));
            else
                weights.add(weights.get(i-1) + tuple_weights.get(i));
        }

        for(int i=0; i < training_data.size(); i++)
        {
            double val = Math.random();
            int tuple_index = 0;
            for(int j = 0; j < weights.size(); j++)
            {
                double w = weights.get(j);
                if(val < w)
                {
                    if(j == 0)
                        tuple_index = 0;
                    else
                        tuple_index = j;
                    break;
                }
            }

            HashMap<Integer,Integer> temp = new HashMap<Integer,Integer>();
            for(Integer attr : training_data.get(tuple_index).keySet())
            {
                temp.put(attr,training_data.get(tuple_index).get(attr));
            }
            sample_data.add(temp);
            int label = training_labels.get(tuple_index);
            sample_data_labels.add(label);

        }


    }

    //This function generates initial weights for all the training records
    private static void generate_initial_weights()
    {
        tuple_weights = new HashMap<Integer, Double>();
        for(int i = 0; i < training_data.size(); i++)
        {
            tuple_weights.put(i, 1 / (training_data.size() * 1.0));
        }
    }

    private static void buildClassifier()
    {
        classifier = new HashMap<Integer,HashMap<Integer, HashMap<Integer, Double>>>();
        sample_labels_frequency = new HashMap<Integer, Integer>();
        for(int i=0; i < sample_data.size(); i++) {

            //Count the frequency of occurrence of each class
            int label = sample_data_labels.get(i);
            if (sample_labels_frequency.containsKey(label)) {
                int val = sample_labels_frequency.get(label);
                sample_labels_frequency.put(label, ++val);
            } else {
                sample_labels_frequency.put(label, 1);
            }

            for (Integer attr : sample_data.get(i).keySet()) {
                int val = sample_data.get(i).get(attr);
                if (!classifier.containsKey(attr))
                {
                    HashMap<Integer, HashMap<Integer, Double>> attr_values = new HashMap<Integer, HashMap<Integer, Double>>();
                    HashMap<Integer, Double> values_class_frequency = new HashMap<Integer, Double>();
                    values_class_frequency.put(label, 1.0);
                    attr_values.put(val, values_class_frequency);
                    classifier.put(attr, attr_values);
                }
                else
                {
                    if (!classifier.get(attr).containsKey(val))
                    {
                        HashMap<Integer, Double> values_class_frequency = new HashMap<Integer, Double>();
                        values_class_frequency.put(label, 1.0);
                        classifier.get(attr).put(val, values_class_frequency);
                    } else
                    {
                        if (!classifier.get(attr).get(val).containsKey(label))
                        {
                            classifier.get(attr).get(val).put(label, 1.0);
                        }
                        else
                        {
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
                for(Integer class_label : sample_labels_frequency.keySet())
                {
                    if(!classifier.get(attribute).get(value).containsKey(class_label))
                    {
                        classifier.get(attribute).get(value).put(class_label, 0.0);
                    }
                }
            }
        }
    }

    private static void predict(ArrayList<HashMap<Integer,Integer>> data, boolean applyLaplace)
    {
        predicted_labels = new ArrayList<Integer>();

        for(int i=0; i < data.size(); i++)
        {
            double max_probability = 0.0;
            int predicted_label = 0;

            for (Integer class_label : sample_labels_frequency.keySet())
            {
                double conditional_probability = 1.0;

                int class_frequency = sample_labels_frequency.get(class_label);

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
                        if(applyLaplace)
                        {
                            attr_value_class_frequency = 1.0;
                        }
                        else
                        {
                            attr_value_class_frequency = 0.0;
                        }
                    }

                    if(!applyLaplace)
                    {
                        conditional_probability = conditional_probability * (attr_value_class_frequency / (class_frequency * 1.0));
                    }
                    else {
                        int freq = class_frequency + sample_attribute_unique_values.get(attr).size();
                        conditional_probability = conditional_probability * (attr_value_class_frequency / (freq * 1.0));
                    }

                }

                double current_probability = conditional_probability * (sample_labels_frequency.get(class_label) / (sample_data.size() * 1.0));
                if (max_probability < current_probability) {
                    max_probability = current_probability;
                    predicted_label = class_label;
                }
            }

            predicted_labels.add(predicted_label);
        }

    }

    public static void generate_rule_measures(ArrayList<Integer> labels)
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

        DecimalFormat df = new DecimalFormat("#.####");

        System.out.println(true_positive + " " + false_negative + " " + false_positive + " " + true_negative);

    }

    public static double get_classifier_error(ArrayList<Integer> labels)
    {
        double error = 0;
        for(int i = 0; i < labels.size(); i++) {
            int actual_label = labels.get(i);
            int predicted_label = predicted_labels.get(i);
            if (actual_label != predicted_label) {
                //Error
                double weight_tuple = tuple_weights.get(i);
                error += weight_tuple * 1.0;
            }
        }

        return error;
    }

    public static void compute_weights(ArrayList<Integer> labels, double error)
    {

        for(int i = 0; i < labels.size(); i++) {
            int actual_label = labels.get(i);
            int predicted_label = predicted_labels.get(i);
            if(actual_label == predicted_label)
            {
                double weight_tuple = tuple_weights.get(i);
                weight_tuple = weight_tuple * (error / (1 - error));
                tuple_weights.put(i,weight_tuple);
            }
        }

        //Normalize the weights
        double sum = 0;
        for(int i = 0; i < tuple_weights.size(); i++)
            sum = sum + tuple_weights.get(i);

        for(int i = 0; i < tuple_weights.size(); i++)
        {
            double normalized_weight = tuple_weights.get(i)/ ( sum * 1.0);

            //Adding the weight to training data
            tuple_weights.put(i, normalized_weight);
        }


    }

    private static void final_prediction(ArrayList<ArrayList<Integer>> classifier_predicted_labels, ArrayList<Double> classifier_errors)
    {
        int size = classifier_predicted_labels.get(0).size();
        predicted_labels = new ArrayList<Integer>();
        for(int i = 0; i < size; i++)
        {
            HashMap<Integer, Double> label_classifiers_weight = new HashMap<Integer, Double>();
            for(int j =0; j < classifier_predicted_labels.size(); j++)
            {
                int predicted_label = classifier_predicted_labels.get(j).get(i);
                double error = classifier_errors.get(j);
                double weight = Math.log((1 - error) / (error * 1.0));
                if(label_classifiers_weight.containsKey(predicted_label))
                {
                    double previous_weight = label_classifiers_weight.get(predicted_label);
                    previous_weight = previous_weight + weight;
                    label_classifiers_weight.put(predicted_label, previous_weight);
                }
                else
                {
                    label_classifiers_weight.put(predicted_label, weight);
                }
            }

            //find label which has maximum weight
            int predicted = 0;
            double weight = 0;
            for(Integer label : label_classifiers_weight.keySet())
            {
                if(label_classifiers_weight.get(label) > weight)
                {
                    weight = label_classifiers_weight.get(label);
                    predicted = label;
                }
            }

            predicted_labels.add(predicted);

        }
    }

    private static void get_attribute_unique_values(ArrayList<HashMap<Integer,Integer>> data1, ArrayList<HashMap<Integer,Integer>> data2)
    {
        sample_attribute_unique_values = new HashMap<Integer, ArrayList<Integer>>();

        for(int i = 0; i < data1.size(); i++)
        {
            for(Integer attr : data1.get(i).keySet())
            {
                if(!sample_attribute_unique_values.containsKey(attr))
                {
                    ArrayList<Integer> values = new ArrayList<Integer>();
                    values.add(data1.get(i).get(attr));
                    sample_attribute_unique_values.put(attr, values);
                }
                else
                {
                    int value = data1.get(i).get(attr);
                    if(!sample_attribute_unique_values.get(attr).contains(value))
                        sample_attribute_unique_values.get(attr).add(value);
                }
            }
        }

        for(int i = 0; i < data2.size(); i++)
        {
            for(Integer attr : data2.get(i).keySet())
            {
                if(!sample_attribute_unique_values.containsKey(attr))
                {
                    ArrayList<Integer> values = new ArrayList<Integer>();
                    values.add(data2.get(i).get(attr));
                    sample_attribute_unique_values.put(attr, values);
                }
                else
                {
                    int value = data2.get(i).get(attr);
                    if(!sample_attribute_unique_values.get(attr).contains(value))
                        sample_attribute_unique_values.get(attr).add(value);
                }
            }
        }
    }

    private static void naive_adaboost(String training_file, String test_file)
    {
        ArrayList<Double> classifier_errors = new ArrayList<Double>();
        ArrayList<ArrayList<Integer>> classifier_predicted_labels = new ArrayList<ArrayList<Integer>>();
        ArrayList<ArrayList<Integer>> classifier_training_predicted_labels = new ArrayList<ArrayList<Integer>>();

        //Find Maximum Attributes
        findMaxAttributes(training_file,test_file);

        //Load the training and test data into memory
        load_data(training_file,test_file);

        //Generating Initial weights - Training data
        generate_initial_weights();

        int k = 0;
        int fail_safe = 0;
        while(k != 8)
        {
            //Get Random Sample Data
            get_sample();

            //Step 5: Build classifier on sample
            buildClassifier();

            //Step 6: Predict on Training data
            get_attribute_unique_values(sample_data,training_data);

            predict(training_data, true);

            ArrayList<Integer> temp = new ArrayList<Integer>();
            temp.addAll(predicted_labels);

            //Step 8: get error incurred by classifier
            double error = get_classifier_error(training_labels);

            if(error > 0.5) {
                if(fail_safe > 5)
                {
                    fail_safe = 0;
                    generate_initial_weights();
                }
                else
                {
                    fail_safe++;
                }
                continue;
            }

            k++;

            //Step 9: compute weights and normalize
            compute_weights(training_labels, error);


            //Step 11: store classifier errors
            classifier_errors.add(error);


            //predict on training data
            classifier_training_predicted_labels.add(temp);


            get_attribute_unique_values(sample_data,test_data);

            //Step 12: predict on test data
            predict(test_data, true);


            //Step 14: store the predicted labels
            temp = new ArrayList<Integer>();
            temp.addAll(predicted_labels);
            classifier_predicted_labels.add(temp);
        }

        //Get Final Predictions on Training Data from all classifiers
        final_prediction(classifier_training_predicted_labels,classifier_errors);

        //Generate Rule Measures - Training Data
        generate_rule_measures(training_labels);

        //Get Final Predictions on Test Data from all classifiers
        final_prediction(classifier_predicted_labels,classifier_errors);

        //Generate Rule Measures - Test Data
        generate_rule_measures(test_labels);
    }


    public static void main(String[] args)
    {
        String training_file = args[0];
        String test_file = args[1];

        naive_adaboost(training_file, test_file);

    }
}

