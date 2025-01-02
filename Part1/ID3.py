import numpy as np
# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}

# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label

class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

        # further variables and functions can be added...


    def calculate_entropy__(self, dataset, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """
        entropy_value = 0.0

        """
        Entropy calculations
        """
        total_length = len(dataset)
        unique_labels = np.unique(labels)
        num_of_each_label = np.zeros(len(unique_labels))
        for i, label in enumerate(unique_labels):
            num_of_each_label[i] = np.count_nonzero(labels == label)
        # entropy = -sum((count / total) * math.log2(count / total)
        p_i = num_of_each_label / total_length
        entropy_value = -(np.sum((p_i) * np.log2(p_i)))

        return entropy_value

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        average_entropy = 0.0
        """
            Average entropy calculations
        """
        # Average Entropy = Information
        # I will take attribute as a number [0, 15]
        attribute_values = np.unique(dataset[:, attribute])
        dataset_length = len(dataset)
        for attribute_value in attribute_values:
            sample_indexes_with_attribute_value = np.where(dataset[:, attribute] == attribute_value)
            samples_with_attribute_value = dataset[sample_indexes_with_attribute_value]
            labels_with_attribute_value = labels[sample_indexes_with_attribute_value]
            entropy = self.calculate_entropy__(samples_with_attribute_value, labels_with_attribute_value)
            average_entropy += (len(samples_with_attribute_value) / dataset_length) * entropy
        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        information_gain = 0.0
        """
            Information gain calculations
        """
        # Entropy - Average Entropy
        entropy = self.calculate_entropy__(dataset, labels)
        average_entropy = self.calculate_average_entropy__(dataset, labels, attribute)
        information_gain = entropy-average_entropy
        return information_gain

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        intrinsic_info = 0.0
        """
            Intrinsic information calculations for a given attribute
        """
        attribute_values = np.unique(dataset[:, attribute])
        dataset_length = len(dataset)
        for attribute_value in attribute_values:
            sample_indexes_with_attribute_value = np.where(dataset[:, attribute] == attribute_value)
            div = len(sample_indexes_with_attribute_value) / dataset_length
            ii = -div * np.log2(div)
            intrinsic_info += ii
        return intrinsic_info

    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        gain_ratio = 0.0
        """
            Your implementation
        """
        # Information Gain /Intrinsic Information
        ig = self.calculate_information_gain__(dataset, labels, attribute)
        ii = self.calculate_intrinsic_information__(dataset, labels, attribute)
        gain_ratio = ig / ii
        return gain_ratio


    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current  tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """
        """
            Your implementation
        """
        if len(np.unique(labels)) == 1:
            return TreeLeafNode(dataset, labels)
        
        # If there are no more attributes to split, return a leaf node
        # if len(used_attributes) == len(self.features):
        #    return TreeLeafNode(dataset, labels)

    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array 
        :return: predicted label of x
        
        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """
        predicted_label = None
        """
            Your implementation
        """

        return predicted_label

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")