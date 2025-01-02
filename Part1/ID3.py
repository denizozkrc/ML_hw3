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

        if len(np.unique(labels)) == 1:  # pure node
            return TreeLeafNode(dataset, labels)
        elif len(used_attributes) == len(self.features):  # used all attributes, still not pure
            return TreeLeafNode(dataset, labels)
        
        selected_attribute_criterion_value = 0
        selected_attribute = -1 #don't need to make it -1 but just in case

        if self.criterion == "information gain":
            for attribute in range(len(self.features)):
                if attribute not in used_attributes:
                    gain = self.calculate_information_gain__(dataset, labels, attribute)
                    if gain > selected_attribute_criterion_value:
                        selected_attribute_criterion_value = gain
                        selected_attribute = attribute
        elif self.criterion == "gain ratio":
            for attribute in range(len(self.features)):
                if attribute not in used_attributes:
                    gain = self.calculate_gain_ratio__(dataset, labels, attribute)
                    if gain > selected_attribute_criterion_value:
                        selected_attribute_criterion_value = gain
                        selected_attribute = attribute

        if (selected_attribute == -1):
            print("ERROR: selected attribute is -1")
            return None

        node = TreeNode(selected_attribute)
        used_attributes.append(selected_attribute)

        # For each value of attribute
        attribute_values = np.unique(dataset[:, selected_attribute])
        for attribute_value in attribute_values:
            sample_indexes_with_attribute_value = np.where(dataset[:, selected_attribute] == attribute_value)
            samples_with_attribute_value = dataset[sample_indexes_with_attribute_value]
            labels_with_attribute_value = labels[sample_indexes_with_attribute_value]
            if len(samples_with_attribute_value) == 0:  # atribute value not seen in this node, do nothing or empty node?
                continue
            else:
                node.subtrees[str(attribute_value)] = self.ID3__(samples_with_attribute_value, labels_with_attribute_value, used_attributes)
        return node

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
        node = self.root
        while not isinstance(node, TreeLeafNode):
            attribute = node.attribute
            attribute_value = x[attribute]
            if str(attribute_value) in node.subtrees: #search in dictionary
                node = node.subtrees[str(attribute_value)]
            else:
                print(f"ERROR: attribute value {attribute_value}  of attribute {attribute} not found in node")
                return None

        # came to the leaf
        unique_labels = np.unique(node.labels)
        unique_label_counts = np.zeros(len(unique_labels))
        labels = node.labels
        for label in unique_labels:
            unique_label_counts[label] = np.count_nonzero(labels == label)
        max_counted_label_index = np.argmax(unique_label_counts)
        predicted_label = unique_labels[max_counted_label_index]

        return predicted_label

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")
