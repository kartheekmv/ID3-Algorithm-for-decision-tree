import java.io.*;
import java.util.*;


class node 
{
	public static int counter;
	public node(){
		super();
		counter++;
	}
	node parent;
	node leftNode;
	node rightNode;
	int label = -1;
	boolean isLeaf = false;
	int target_Attribute = -1;
	int left_Indices[];
	int right_Indices[];
}

public class DecissionTree 
{
	private static int count = 0;
	
	private static double Log2Calculation(double fraction) 
	{
		return Math.log10(fraction) / Math.log10(2);
	}
	
	public static void main(String[] args) 
	{
		if (args.length != 4) 
		{
			System.out.println("Not enough command line arguments!!");
			return;
		}
		int h=0;
		node root=null;
	//	System.out.println(node.counter);
		
		int[] training_features = feautres_size(args[0]);
		int[] validation_features = feautres_size(args[1]);
		int[] test_feautures = feautres_size(args[2]);
		
		int[][] values = new int[training_features[1]][training_features[0]];
		int[][] values2 = new int[validation_features[1]][validation_features[0]];
		int[][] values3 = new int[test_feautures[1]][test_feautures[0]];
		
		String[] feature_Names = new String[training_features[0]];
		
		int[] done = new int[training_features[0]];
		
		int[] index_List = new int[values.length];
		
		Read_values(args[0], values, feature_Names, done, index_List,training_features[0]);
		
		root = construct_DecissionTree(null, values, done, training_features[0] - 1, index_List, null,h);
		
		int no_Nodes = node.counter;
		
		//	System.out.println(no_Nodes);
			
		System.out.println("*********** TREE BEFORE PRUNING ****************");
		print_Tree(root, 0, feature_Names);
		
		int pruneNodes = (int) (Double.parseDouble(args[3])*node.counter);
		
		//		System.out.println(pruneNodes);
		
		node pruneTree = PruningAlgorithm(args[1], pruneNodes, root, values, training_features[0] - 1);
		
	    System.out.println("\n**************Pre-pruned Accuracy:*************");
		
		System.out.println("\nNumber of training instances: "+(training_features[1]-1));
		System.out.println("Number of training Attributes: "+(training_features[0]-1));
		System.out.println("Total number of nodes in the tree: = "+no_Nodes);
		System.out.println("Total number of Leaf nodes in tree: = "+ LeafNodes(root));
		System.out.println("Accuracy of the model on training data set " + 100*(accuracy_calculation(args[0], root))+" %");
		
		System.out.println("\nNumber of validation instances: "+(validation_features[1]-1));
		System.out.println("Number of validation Attributes: "+(validation_features[0]-1));
		System.out.println("Accuracy of the model on the validation data set " + 100*(accuracy_calculation(args[1], root))+" %");
		
		
		System.out.println("\nNumber of testing instances: "+  (test_feautures[1]-1));
		System.out.println("Number of testing Attributes: "+ (test_feautures[0]-1));
		System.out.println("Accuracy of the model on testing data set " + 100*(accuracy_calculation(args[2], root))+" %");
		
		
		System.out.println("\n\n***********TREE AFTER PRUNING ****************");
		print_Tree(pruneTree, 0, feature_Names);
		
		
	    int prunedAlready = no_Nodes - pruneNodes ;
		
		//System.out.println(prunedAlready);
		 
		System.out.println("\n**************Post-pruned Accuracy:*************");
		
		
		System.out.println("\nNumber of training instances: "+(training_features[1]-1));
		System.out.println("Number of training Attributes: "+(training_features[0]-1));
		System.out.println("Total number of nodes in tree: = "+prunedAlready);
		System.out.println("Total number of Leaf nodes in tree: = "+ LeafNodes(pruneTree));
		System.out.println("Accuracy of the model on the training data set after pruning " + 100*(accuracy_calculation(args[0], pruneTree))+" %");
		
		System.out.println("\nNumber of validation instances: "+(validation_features[1]-1));
		System.out.println("Number of validation Attributes: "+(validation_features[0]-1));
		System.out.println("Accuracy of the model on the validation data set after pruning " + 100*(accuracy_calculation(args[1], pruneTree))+" %");
		
		System.out.println("\nNumber of testing instances: "+  (test_feautures[1]-1));
		System.out.println("Number of testing Attributes: "+(test_feautures[0]-1));
		System.out.println("Accuracy of the model on testing data set after pruning " + (100*accuracy_calculation(args[2], pruneTree))+" %");
		


	}

	
	/*Decission Tree ID3 algorithm */
	private static node findBestAttributeToConstructTree(node root, int[][] values, int[] done, int features,
			int[] index_List) 
	{
		int i = 0;
		int j = 0;
		int k = 0;
		double maxInformationGain = 0;
		int maxLeftIndex[] = null;
		int maxRightIndex[] = null;
		int maxIndex = -1;
		for (; i < features; i++) 
		{
			if (done[i] == 0) 
			{
				double negatives = 0;
				double positives = 0;
				double leftNode = 0;
				double rightNode = 0;
				double leftEntropy = 0;
				double rightEntropy = 0;
				int[] leftIndex = new int[values.length];
				int[] rightIndex = new int[values.length];
				double entropy = 0;
				double right_Positives = 0;
				double informationGain = 0;
				double right_Negatives = 0, left_Positives = 0, left_Negatives = 0;
				for (k = 0; k < index_List.length; k++) 
				{
					if (values[index_List[k]][features] == 1) 
					{
						positives++;
					} else 
					{
						negatives++;
					}
					if (values[index_List[k]][i] == 1) 
					{
						rightIndex[(int) rightNode++] = index_List[k];
						if (values[index_List[k]][features] == 1) 
						{
							right_Positives++;
						} else 
						{
							right_Negatives++;
						}

					} else {
						leftIndex[(int) leftNode++] = index_List[k];
						if (values[index_List[k]][features] == 1) 
						{
							left_Positives++;
						} else 
						{
							left_Negatives++;
						}

					}

				}

				entropy = (-1 * Log2Calculation(positives / index_List.length) * ((positives / index_List.length)))
						+ (-1 * Log2Calculation(negatives / index_List.length) * (negatives / index_List.length));
				leftEntropy = (-1 * Log2Calculation(left_Positives / (left_Positives + left_Negatives))
						* (left_Positives / (left_Positives + left_Negatives)))
						+ (-1 * Log2Calculation(left_Negatives / (left_Positives + left_Negatives))
								* (left_Negatives / (left_Positives + left_Negatives)));
				rightEntropy = (-1 * Log2Calculation(right_Positives / (right_Positives + right_Negatives))
						* (right_Positives / (right_Positives + right_Negatives)))
						+ (-1 * Log2Calculation(right_Negatives / (right_Positives + right_Negatives))
								* (right_Negatives / (right_Positives + right_Negatives)));
				if (Double.compare(Double.NaN, entropy) == 0) 
				{
					entropy = 0;
				}
				if (Double.compare(Double.NaN, leftEntropy) == 0) 
				{
					leftEntropy = 0;
				}
				if (Double.compare(Double.NaN, rightEntropy) == 0) 
				{
					rightEntropy = 0;
				}

				informationGain = entropy
						- ((leftNode / (leftNode + rightNode) * leftEntropy) + (rightNode / (leftNode + rightNode) * rightEntropy));
				if (informationGain >= maxInformationGain) 
				{
					maxInformationGain = informationGain;
					maxIndex = i;
					int leftTempArray[] = new int[(int) leftNode];
					for (int index = 0; index < leftNode; index++) 
					{
						leftTempArray[index] = leftIndex[index];
					}
					int rightTempArray[] = new int[(int) rightNode];
					for (int index = 0; index < rightNode; index++) 
					{
						rightTempArray[index] = rightIndex[index];
					}
					maxLeftIndex = leftTempArray;
					maxRightIndex = rightTempArray;

				}
			}
		}
		root.target_Attribute = maxIndex;
		root.left_Indices = maxLeftIndex;
		root.right_Indices = maxRightIndex;
		return root;
	}

	
	 /*Checking whether all the examples have Target_Attribute as 1*/
	public static boolean All_Positive(int[] index_List, int[][] instances, int features) 
	{
		boolean state = true;
		for (int i : index_List) 
		{
			if (instances[i][features] == 0)
				state = false;
		}
		return state;

	}

	 /*Checking whether all the examples have Target_Attribute as 0 */
	public static boolean All_Negative(int[] index_List, int[][] instances, int features) 
	{
		boolean state = true;
		for (int i : index_List) 
		{
			if (instances[i][features] == 1)
				state = false;
		}
		return state;

	}

	
	/* Finding max value of Target_Attribute among the examples available at the given splitting attribute */
	public static int max_value(node root, int[][] instances, int features) 
	{
		int One = 0;
		int Zero = 0;
		if (root.parent == null) 
		{
			int i = 0;
			for (i = 0; i < instances.length; i++) 
			{
				if (instances[i][features] == 1) 
				{
					One++;
				} 
				else 
				{
					Zero++;
				}
			}
		} 
		else 
		{
			for (int i : root.parent.left_Indices) 
			{
				if (instances[i][features] == 1) 
				{
					One++;
				} 
				else 
				{
					Zero++;
				}
			}

			for (int i : root.parent.right_Indices) 
			{
				if (instances[i][features] == 1) 
				{
					One++;
				} 
				else 
				{
					Zero++;
				}
			}
		}
		if(Zero > One)
			return 0;
		else
			return 1;

	}

	
	/* To check if all the Attributes are done or not */
	public static boolean Done_allAttributes(int[] done) 
	{
		boolean Done = true;
		for (int i : done) 
		{
			if (i == 0)
				Done = false;
		}
		return Done;
	}

	
	public static node construct_DecissionTree(node root, int[][] values, int[] done, int features, int[] index_List, node parent, int h) 
	{
		if (root == null) 
		{
			root = new node();
			if (index_List == null || index_List.length == 0) 
			{
				root.label = max_value(root, values, features);
				root.isLeaf = true;
				return root;
			}
			if (All_Positive(index_List, values, features)) 
			{
				root.label = 1;
				root.isLeaf = true;
				return root;
			}
			if (All_Negative(index_List, values, features)) 
			{
				root.label = 0;
				root.isLeaf = true;
				return root;
			}
			if (features == 1 || Done_allAttributes(done)) 
			{
				root.label = max_value(root, values, features);
				root.isLeaf = true;
				return root;
			}
		}
		root = findBestAttributeToConstructTree(root, values, done, features, index_List);
		root.parent = parent; 
		if (root.target_Attribute != -1)
			done[root.target_Attribute] = 1;
		int leftIsDone[] = new int[done.length];
		int rightIsDone[] = new int[done.length];
		for (int j = 0; j < done.length; j++) {
			leftIsDone[j] = done[j];
			rightIsDone[j] = done[j]; 

		}

		root.leftNode = construct_DecissionTree(root.leftNode, values, leftIsDone,features, root.left_Indices, root,h);
		root.rightNode = construct_DecissionTree(root.rightNode, values, rightIsDone,features,root.right_Indices, root,h);
		return root;
	}

	
	/* creating copy for the given tree and return the tree*/
	public static node createCopy(node root) 
	{
		if (root == null)
			return root;

		node temp = new node();
		temp.label = root.label;
		temp.isLeaf = root.isLeaf;
		temp.left_Indices = root.left_Indices;
		temp.right_Indices = root.right_Indices;
		temp.target_Attribute = root.target_Attribute;
		temp.parent = root.parent;
		temp.leftNode = createCopy(root.leftNode); 
		temp.rightNode = createCopy(root.rightNode); 
		return temp;
	}
	
	
	/*Pruning algorithm on the constructed Tree.*/
		
	public static node PruningAlgorithm(String pathName,int K, node root, int[][] values, int features) 
	{
		node postPrunedTree = new node();
		int i = 0;
		postPrunedTree = root;
		double maxAccuracy = Accuracy_validationSet(pathName, root);
		node newRoot = createCopy(root);
		Random randomNumbers = new Random();
		int L = NotLeafNodes(newRoot);
		
		int calculate_NotLeadNodes = NotLeafNodes(newRoot);

		node nodeArray[] = new node[calculate_NotLeadNodes+1];
		count =0;
		buildArray(newRoot, nodeArray);
		for (i = 0; i < K; i++) 
		{
			int p = 4 + randomNumbers.nextInt(L-4+1);
			if (calculate_NotLeadNodes == 0)
				break;	
			if(p<=L)
			{
				nodeArray[p] = create_Leaf(nodeArray[p], values, features);
			}
			else
			{
				System.out.println("Number of nodes to prune are greater than the nodes present!");
				System.exit(0);
			}
		}
		
		double accuracy =  Accuracy_validationSet(pathName, newRoot);
		if (accuracy > maxAccuracy) {
				postPrunedTree = newRoot;
				maxAccuracy = accuracy;
		}
		return postPrunedTree;
	}


	
	private static double Accuracy_validationSet(String file, node newRoot) {
		int[][] validationSet = ValidationSet(file);
		double count = 0;
		for (int i = 1; i < validationSet.length; i++) {
			count += isClassificationCorrect(validationSet[i], newRoot);
		}
		return count / validationSet.length;
	}

	/*
	 * To check clasification is correct or not
	 * as per the Constructed Tree.
	 */
	private static int isClassificationCorrect(int[] setValues, node newRoot) {
		int index = newRoot.target_Attribute;
		//System.out.println(index);
		int correctlyClassified = 0;
		node testingNode = newRoot;
		while (testingNode.label == -1&&index!=-1) {
			if (setValues[index] == 1) {
				testingNode = testingNode.rightNode;
			} else {
				testingNode = testingNode.leftNode;
			}
			if (testingNode.label == 1 || testingNode.label == 0) {
				if (setValues[setValues.length - 1] == testingNode.label) {
					correctlyClassified = 1;
					break;
				} else {
					break;
				}
			}
			index = testingNode.target_Attribute;
		}
		return correctlyClassified;
	}


	private static int[][] ValidationSet(String pathName) {
		int[] featuresAndLength = feautres_size(pathName);
		String csvFile = pathName;
		int[][] validationSet = new int[featuresAndLength[1]][featuresAndLength[0]];
		BufferedReader bufferedReader = null;
		String line = "";
		String cvsSplitBy = ",";
		try {
			bufferedReader = new BufferedReader(new FileReader(csvFile));
			int i = 0;
			int count = 0;
			while ((line = bufferedReader.readLine()) != null) {
				String[] lineParameters = line.split(cvsSplitBy);
				int j = 0;
				if (count == 0) {
					count++;
					continue;
				} else {
					for (String lineParameter : lineParameters) {
						validationSet[i][j++] = Integer.parseInt(lineParameter);
					}
				}
				i++;
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (bufferedReader != null) {
				try {
					bufferedReader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return validationSet;
	}

	
	/* To return the maxValue of the Target_Attribute among 
	the the examples present at the node of decision tree.*/
	private static int Max_Value_At_GivenNode(node root, int[][] values, int features) {
		int noOfOnes = 0;
		int noOfZeroes = 0;
		if (root.left_Indices != null) {
			for (int i : root.left_Indices) {
				if (values[i][features] == 1) {
					noOfOnes++;
				} else {
					noOfZeroes++;
				}
			}
		}

		if (root.right_Indices != null) {
			for (int i : root.right_Indices) {
				if (values[i][features] == 1) {
					noOfOnes++;
				} else {
					noOfZeroes++;
				}
			}
		}
		return noOfZeroes > noOfOnes ? 0 : 1;
	}

	private static node create_Leaf(node node, int[][] values, int features) {
		node.isLeaf = true;
		node.label = Max_Value_At_GivenNode(node, values, features);
		node.leftNode = null;
		node.rightNode = null;
		return node;
	}

	
	private static void buildArray(node root, node[] nodeArray) {
		if (root == null || root.isLeaf) {
			return;
		}
		count++;
		nodeArray[count] = root;
		if (root.leftNode != null) {
			buildArray(root.leftNode, nodeArray);
		}
		if (root.rightNode != null) {
			buildArray(root.rightNode, nodeArray);
		}
	}

	
	/* counting the number of non leaf nodes  */
	private static int NotLeafNodes(node root) {
		if (root == null || root.isLeaf)
			return 0;
		else
			return (1 + NotLeafNodes(root.leftNode) + NotLeafNodes(root.rightNode));
	}
	
	/* counting the number of leaf nodes. */
	private static int LeafNodes(node root) {
		if (root == null)
			return 0;
		else if (root.isLeaf==true)
			return 1;
		else
			return (LeafNodes(root.leftNode) + LeafNodes(root.rightNode));
	}
	
	private static int[] feautres_size(String file) 
	{
		BufferedReader br = null;
		int c = 0;
		int[] features_instances = new int[2];
		String line = "";
		try 
		{

			br = new BufferedReader(new FileReader(file));
			while ((line = br.readLine()) != null) 
			{
				if (c == 0) 
				{
					String[] features = line.split(",");
					features_instances[0] = features.length;
				}
				c++;
			}

		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		} 
		finally 
		{
			if (br != null) 
			{
				try 
				{
					br.close();
				} 
				catch (Exception e) 
				{
					e.printStackTrace();
				}
			}
		}
		features_instances[1] = c;
		return features_instances;
	}

	
	private static void Read_values(String file, int[][] instances, String[] feature_Names, int[] done,int[] index_List, int features) 
	{
		BufferedReader br = null;
		String line = "";
		for (int k = 0; k < features; k++) {
			done[k] = 0;
		}
		int k = 0;
		for (k = 0; k < instances.length; k++) {
			index_List[k] = k;
		}
		try {

			br = new BufferedReader(new FileReader(file));
			int c = 0;
			while ((line = br.readLine()) != null) 
			{
				String[] fields = line.split(",");
				int i = 0;
				if (c == 0) 
				{
					for (String field : fields) 
					{
						feature_Names[i++] = field;
					}
				}

				else 
				{

					for (String field : fields) 
					{
						instances[c][i++] = Integer.parseInt(field);
					}
				}
				c++;
			}
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
		finally 
		{
			if (br != null) 
			{
				try 
				{
					br.close();
				} 
				catch (Exception e) 
				{
					e.printStackTrace();
				}
			}
		}
	}

	
	private static void print_Tree(node root, int printLines, String[] feature_Names) 
	{
		int i = printLines;
		if (root.isLeaf) 
		{
			System.out.println(" " + root.label);
			return;
		}
		for (int j = 0; j < i; j++) 
		{
			System.out.print("| ");
		}
		if (root.leftNode != null && root.leftNode.isLeaf && root.target_Attribute !=-1)
			System.out.print(feature_Names[root.target_Attribute] + "= 0 :");
		else
			if(root.target_Attribute !=-1)
			System.out.println(feature_Names[root.target_Attribute] + "= 0 :");

		printLines++;
		print_Tree(root.leftNode, printLines, feature_Names);
		for (int j = 0; j < i; j++) 
		{
			System.out.print("| ");
		}
		if (root.rightNode != null && root.rightNode.isLeaf&& root.target_Attribute !=-1)
			System.out.print(feature_Names[root.target_Attribute] + "= 1 :");
		else
			if(root.target_Attribute !=-1)
			System.out.println(feature_Names[root.target_Attribute] + "= 1 :");
		print_Tree(root.rightNode, printLines, feature_Names);
	}

	
	private static double accuracy_calculation(String file, node root) 
	{
		double accuracy = 0;
		int[][] testdata = read_testingdata(file);
		for (int i = 0; i < testdata.length; i++) {
			accuracy += isClassificationCorrect(testdata[i], root);
		}
		return accuracy / testdata.length;

	}

	
	private static int[][] read_testingdata(String file) 
	{
		int[] features_instances = feautres_size(file);
		int[][] validation_set = new int[features_instances[1]][features_instances[0]];
		BufferedReader br = null;
		String line = "";
		try {

			br = new BufferedReader(new FileReader(file));
			int i = 0;
			int c = 0;
			while ((line = br.readLine()) != null) 
			{
				String[] fields = line.split(",");
				int j = 0;
				if (c == 0) 
				{
					c++;
					continue;
				}

				else 
				{

					for (String field : fields) 
					{
						validation_set[i][j++] = Integer.parseInt(field);
					}
				}
				i++;
			}
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		} 
		finally 
		{
			if (br != null) 
			{
				try 
				{
					br.close();
				} 
				catch (Exception e) 
				{
					e.printStackTrace();
				}
			}
		}
		return validation_set;
	}
	
}	
