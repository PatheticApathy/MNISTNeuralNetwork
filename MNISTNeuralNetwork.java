import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

public class MNISTNeuralNetwork {

    public static double e = 2.71828;
    public static double learningrate = 3.0;
    public static int minibatchsize = 10;
    public static int epoch = 30;
    public static int inputsize = 784;
    public static int outputsize = 10;
    public static int hiddensize = 15;
    public static int trainingsize = 60000;
    public static int testingsize = 10000;

    public static double[][] weights1; // weights between input and hidden layer
    public static double[][] weights2; // weights between hidden and output layer
    public static double[] biases1;    // biases for hidden layer
    public static double[] biases2;    // biases for output layer
    public static double[] hidden;
    public static double[] output;
    public static double[] deltaOutput;
    public static double[] deltaHidden;

    public static void main(String[] args) throws IOException {
        
        Scanner scanner = new Scanner(System.in);
        int number;

        System.out.print("Enter a number: \n");
        System.out.print("[1] Train Network \n");
        System.out.print("[2] Load a pre-trained network \n");
        System.out.print("[0] Exit \n");

        MNISTNeuralNetwork nn = new MNISTNeuralNetwork(inputsize, hiddensize, outputsize, learningrate, minibatchsize, epoch);

        // loop to continuously ask for input until '0' is entered
        while (true) {
            number = scanner.nextInt();

            if (number == 1) {
                nn.train("mnist_train.csv", trainingsize);

                while (true) {

                    System.out.print("Enter a number: \n");
                    System.out.print("[1] Train Network \n");
                    System.out.print("[2] Load a pre-trained network \n");
                    System.out.print("[3] Display network accuracy on TRAINING data \n");
                    System.out.print("[4] Display network accuracy on TESTING data \n");
                    System.out.print("[5] Run network on Testing data showing images and labels \n");
                    System.out.print("[6] Display the misclassified TESTING images \n");
                    System.out.print("[7] Save the network state to file \n");
                    System.out.print("[0] Exit \n");

                    number = scanner.nextInt();
                    
                    if (number == 1) {
                        nn = new MNISTNeuralNetwork(inputsize, hiddensize, outputsize, learningrate, minibatchsize, epoch);
                        nn.train("mnist_train.csv", trainingsize);
                    }
                    if (number == 2) {
                        nn.loadWeightsFromFile("weightset.csv");
                    }
                    if (number == 3) {
                        nn.accuracy("mnist_train.csv", trainingsize);
                    }
                    if (number == 4) {
                        nn.accuracy("mnist_test.csv", testingsize);
                    }
                    if (number == 5) {
                        nn.runOnTestDataAndDisplay("mnist_test.csv", 10000);
                    }
                    if (number == 6) {
                        nn.displayMisclassified("mnist_test.csv", testingsize);
                    }
                    if (number == 7) {
                        nn.saveWeightsToFile("weightset.csv");
                    }
                    if (number == 0) {
                        break;
                    }
                }
            }
            if (number == 2) {
                nn.loadWeightsFromFile("weightset.csv");

                while (true) {

                    System.out.print("Enter a number: \n");
                    System.out.print("[1] Train Network \n");
                    System.out.print("[2] Load a pre-trained network \n");
                    System.out.print("[3] Display network accuracy on TRAINING data \n");
                    System.out.print("[4] Display network accuracy on TESTING data \n");
                    System.out.print("[5] Run network on TESTING data showing images and labels \n");
                    System.out.print("[6] Display the misclassified TESTING images \n");
                    System.out.print("[7] Save the network state to file \n");
                    System.out.print("[0] Exit \n");

                    number = scanner.nextInt();
                    
                    if (number == 1) {
                        nn = new MNISTNeuralNetwork(inputsize, hiddensize, outputsize, learningrate, minibatchsize, epoch);
                        nn.train("mnist_train.csv", trainingsize);
                    }
                    if (number == 2) {
                        nn.loadWeightsFromFile("weightset.csv");
                    }
                    if (number == 3) {
                        nn.accuracy("mnist_train.csv", trainingsize);
                    }
                    if (number == 4) {
                        nn.accuracy("mnist_test.csv", testingsize);
                    }
                    if (number == 5) {
                        nn.runOnTestDataAndDisplay("mnist_test.csv", 10000);
                    }
                    if (number == 6) {
                        nn.displayMisclassified("mnist_test.csv", testingsize);
                    }
                    if (number == 7) {
                        nn.saveWeightsToFile("weightset.csv");
                    }
                    if (number == 0) {
                        break;
                    }
                }
            }
            if (number == 0) {
                break;
            }
        }
        scanner.close();
    }

    public MNISTNeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate, int miniBatchSize, int epochs) {

        // Initialize weights and biases randomly between -1 and 1
        Random rand = new Random();

        // weights between input and hidden
        weights1 = new double[hiddenSize][inputSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights1[i][j] = 2 * rand.nextDouble() - 1; // range -1 to 1
            }
        }

        // weights between hidden and output
        weights2 = new double[outputSize][hiddenSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weights2[i][j] = 2 * rand.nextDouble() - 1;
            }
        }

        // biases for hidden layer
        biases1 = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            biases1[i] = 2 * rand.nextDouble() - 1;
        }

        // biases for output layer
        biases2 = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            biases2[i] = 2 * rand.nextDouble() - 1;
        }
    }

    public double[] feedforward(double[] input) {
        // hidden layer activation
        hidden = new double[biases1.length];
        for (int i = 0; i < biases1.length; i++) {
            hidden[i] = biases1[i];
            for (int j = 0; j < input.length; j++) {
                hidden[i] += weights1[i][j] * input[j];
            }
            hidden[i] = 1.0 / (1.0 + Math.pow(e,-hidden[i]));
        }

        // output layer activation
        output = new double[biases2.length];
        for (int i = 0; i < biases2.length; i++) {
            output[i] = biases2[i];
            for (int j = 0; j < hidden.length; j++) {
                output[i] += weights2[i][j] * hidden[j];
            }
            output[i] = 1.0 / (1.0 + Math.pow(e, -output[i]));
        }

        return output;
    }

    public void backpropagation(double[] input, double[] target) {

        // output layer error
        deltaOutput = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            deltaOutput[i] = (output[i] - target[i]) * output[i] * (1 - output[i]);
        }

        // hidden layer error
        deltaHidden = new double[hidden.length];
        for (int i = 0; i < hidden.length; i++) {
            double error = 0.0;
            for (int j = 0; j < deltaOutput.length; j++) {
                error += deltaOutput[j] * weights2[j][i];
            }
            deltaHidden[i] = error * hidden[i] * (1 - hidden[i]);
        }
    }

    public void updateweightsandbiases(double[] input) {
        // update weights and biases for output layer
        for (int i = 0; i < biases2.length; i++) {
            biases2[i] -= learningrate/minibatchsize * deltaOutput[i];
            for (int j = 0; j < hidden.length; j++) {
                weights2[i][j] -= learningrate/minibatchsize * deltaOutput[i] * hidden[j];
            }
        }

        // update weights and biases for hidden layer
        for (int i = 0; i < biases1.length; i++) {
            biases1[i] -= learningrate/minibatchsize * deltaHidden[i];
            for (int j = 0; j < input.length; j++) {
                weights1[i][j] -= learningrate/minibatchsize * deltaHidden[i] * input[j];
            }
        }
    }
    
    private int[] shuffleIndices(int size) {
        Random rand = new Random();
        int[] indices = new int[size];
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        for (int i = 0; i < size; i++) {
            int j = rand.nextInt(size);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        return indices;
    }

    // train the network using mini-batches
    public void train(String trainingDataPath, int trainingSize) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(trainingDataPath));
        String[] lines = new String[trainingSize];
        String line;
        int index = 0;
        
        // load all lines from the CSV file
        while ((line = br.readLine()) != null && index < trainingSize) {
            lines[index] = line;
            index++;
        }

        // randomize the order of the input data
        int[] indices = shuffleIndices(trainingSize);

        // loop over the number of epochs
        for (int epochCount = 0; epochCount < epoch; epochCount++) {
            // loop over the training data in batches
            for (int batchStart = 0; batchStart < trainingSize; batchStart += minibatchsize) {
                // loop through each sample in the current minibatch
                for (int i = 0; i < minibatchsize && (batchStart + i) < trainingSize; i++) {
                    String[] values = lines[indices[batchStart + i]].split(",");
                    double[] input = new double[784];
                    double[] target = new double[10];

                    // make pixel values between 0 and 1
                    for (int j = 0; j < 784; j++) {
                        input[j] = Double.parseDouble(values[j + 1]) / 255.0;
                    }

                    // one-hot encode the label
                    int label = Integer.parseInt(values[0]);
                    target[label] = 1.0;

                    feedforward(input);
                    backpropagation(input, target);
                    updateweightsandbiases(input);
                }
            }
            System.out.println("Epoch " + (epochCount + 1) + " completed.");
            accuracy(trainingDataPath, trainingSize);
        }
        br.close();
    }

    public void accuracy(String dataPath, int dataSize) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(dataPath));
        String line;

        int[] correctCounts = new int[10];
        int[] totalCounts = new int[10];
        int totalCorrect = 0;
        int totalTested = 0;

        while ((line = br.readLine()) != null && totalTested < dataSize) {
            String[] values = line.split(",");
            double[] input = new double[784];
            int label = Integer.parseInt(values[0]);

            // make inputs between 0 and 1
            for (int i = 0; i < 784; i++) {
                input[i] = Double.parseDouble(values[i + 1]) / 255.0;
            }

            // feedforward to get network output
            double[] output = feedforward(input);

            // find the outputed label
            int outputedLabel = 0;
            double maxOutput = output[0];
            for (int i = 1; i < output.length; i++) {
                if (output[i] > maxOutput) {
                    outputedLabel = i;
                    maxOutput = output[i];
                }
            }

            // update counts
            totalCounts[label]++;
            if (outputedLabel == label) {
                correctCounts[label]++;
                totalCorrect++;
            }
            totalTested++;
        }

        // print statistics
        for (int i = 0; i < 10; i++) {
            System.out.print(i + " = " + correctCounts[i] + "/" + totalCounts[i] + "  ");
            if (i == 5) {
                System.out.println();
            }
        }
        double accuracy = (double) totalCorrect / totalTested * 100;
        System.out.print("Accuracy = " + totalCorrect + "/" + totalTested + " = " + accuracy + "%\n" );

        br.close();
    }

    // function to save the current weight set to a file
    public void saveWeightsToFile(String filename) throws IOException {
        PrintWriter writer = new PrintWriter(new FileWriter(filename));

        // save weights1
        writer.println("weights1");
        for (int i = 0; i < weights1.length; i++) {
            for (int j = 0; j < weights1[i].length; j++) {
                writer.print(weights1[i][j] + ",");
            }
            writer.println();
        }

        // save biases1
        writer.println("biases1");
        for (int i = 0; i < biases1.length; i++) {
            writer.println(biases1[i]);
        }

        // save weights2
        writer.println("weights2");
        for (int i = 0; i < weights2.length; i++) {
            for (int j = 0; j < weights2[i].length; j++) {
                writer.print(weights2[i][j] + ",");
            }
            writer.println();
        }

        // save biases2
        writer.println("biases2");
        for (int i = 0; i < biases2.length; i++) {
            writer.println(biases2[i]);
        }

        writer.close();
    }

    // function to load weights and biases from a file
    public void loadWeightsFromFile(String filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        String section = "";

        int row = 0;

        while ((line = reader.readLine()) != null) {
            if (line.equals("weights1")) {
                section = "weights1";
                row = 0;
            } else if (line.equals("biases1")) {
                section = "biases1";
                row = 0;
            } else if (line.equals("weights2")) {
                section = "weights2";
                row = 0;
            } else if (line.equals("biases2")) {
                section = "biases2";
                row = 0;
            } else {
                String[] values = line.split(",");
                switch (section) {
                    case "weights1":
                        for (int i = 0; i < values.length; i++) {
                            weights1[row][i] = Double.parseDouble(values[i]);
                        }
                        row++;
                        break;
                    case "biases1":
                        biases1[row] = Double.parseDouble(values[0]);
                        row++;
                        break;
                    case "weights2":
                        for (int i = 0; i < values.length; i++) {
                            weights2[row][i] = Double.parseDouble(values[i]);
                        }
                        row++;
                        break;
                    case "biases2":
                        biases2[row] = Double.parseDouble(values[0]);
                        row++;
                        break;
                }
            }
        }
        reader.close();
    }

    // function to display the image in the console
    private void displayImage(double[] pixels, int correctLabel, int outputedLabel, boolean isCorrect) {
        // display the correct and outputed classification
        System.out.print("Correct label: " + correctLabel + "  ");
        System.out.print("outputed label: " + outputedLabel + "  ");
        System.out.print("Was the network correct? " + (isCorrect ? "Yes" : "No"));
        System.out.println();
        
        // display X for pixel values above 0 and space for 0
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                if (pixels[i * 28 + j] > 0) {  
                    System.out.print("X");
                } else {
                    System.out.print(" ");
                }
            }
            System.out.println();
        }
    }

    private int getUserInput(Scanner scanner) {
        System.out.println("Enter 1 to continue. All other values return to main menu");
        return scanner.nextInt();
    }

    // run the network on the testing data, displaying images and labels
    public void runOnTestDataAndDisplay(String dataPath, int dataSize) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(dataPath));
        String line;
        Scanner scanner = new Scanner(System.in);

        int currentIndex = 0;
        double[] inputs = new double[784];
        int correctLabels = 0;

        while (currentIndex < dataSize) {
            line = br.readLine();
            String[] values = line.split(",");
            correctLabels = Integer.parseInt(values[0]);

            // make inputs between 0 and 1
            for (int i = 0; i < 784; i++) {
                inputs[i] = Double.parseDouble(values[i + 1]) / 255.0;
            }

            // feedforward to get network prediction
            double[] output = feedforward(inputs);

            // find the outputed label
            int outputedLabel = 0;
            double maxOutput = output[0];
            for (int i = 1; i < output.length; i++) {
                if (output[i] > maxOutput) {
                    outputedLabel = i;
                    maxOutput = output[i];
                }
            }

            // check if the output is correct
            boolean isCorrect = (outputedLabel == correctLabels);

            // display the image with the classification information
            displayImage(inputs, correctLabels, outputedLabel, isCorrect);

            // handle user input to navigate
            int userInput = getUserInput(scanner);
            if (userInput == 1) {
                currentIndex = (currentIndex + 1) % dataSize;
            } else {
                break;
            }
        }
        br.close();
    }

    // display the misclassified images
    public void displayMisclassified(String dataPath, int dataSize) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(dataPath));
        String line;
        Scanner scanner = new Scanner(System.in);

        int totalTested = 0;
        double[] input = new double[784];
        int correctLabel = 0;

        while ((line = br.readLine()) != null && totalTested < dataSize) {
            String[] values = line.split(",");
            correctLabel = Integer.parseInt(values[0]);

            // make inputs between 0 and 1
            for (int i = 0; i < 784; i++) {
                input[i] = Double.parseDouble(values[i + 1]) / 255.0;
            }

            // feedforward to get network prediction
            double[] output = feedforward(input);

            // find the outputed label
            int outputedLabel = 0;
            double maxOutput = output[0];
            for (int i = 1; i < output.length; i++) {
                if (output[i] > maxOutput) {
                    outputedLabel = i;
                    maxOutput = output[i];
                }
            }

            // display the misclassified images
            if (outputedLabel != correctLabel) {
                displayImage(input, correctLabel, outputedLabel, false);

                // handle user input to navigate
                int userInput = getUserInput(scanner);
                if (userInput == 1) {
                    totalTested++;
                } else {
                    break;
                }
            }
        }
        br.close();
    }
}