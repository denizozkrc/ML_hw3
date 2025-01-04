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
        self.most_important_features = []

        # further variables and functions can be added...
    def max_labels(self, current_node: TreeNode):  # retruns the most probable label
        num_of_labels = np.zeros(len(self.labels))
        for attr_val in current_node.subtrees:
            if isinstance(current_node.subtrees[attr_val], TreeNode):
                num_of_labels += self.max_labels(current_node.subtrees[attr_val])
            else:  # leaf node
                for i, label in enumerate(self.labels):
                    num_of_labels[i] += np.sum(current_node.subtrees[attr_val].labels == label)
        return self.labels[np.argmax(num_of_labels)]

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
            num_of_each_label[i] = np.sum(labels == label)
        p_i = num_of_each_label / total_length
        entropy_value = -(np.sum((p_i) * np.log(p_i)))

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
        average_entropy_from_attribute_value = np.zeros(len(attribute_values))
        for i, attribute_value in enumerate(attribute_values):
            sample_indexes_with_attribute_value = np.where(dataset[:, attribute] == attribute_value)
            samples_with_attribute_value = dataset[sample_indexes_with_attribute_value]
            labels_with_attribute_value = labels[sample_indexes_with_attribute_value]
            entropy = self.calculate_entropy__(samples_with_attribute_value, labels_with_attribute_value)
            average_entropy_from_attribute_value[i] = (len(samples_with_attribute_value) / dataset_length) * entropy
        average_entropy = np.sum(average_entropy_from_attribute_value)
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
        ii_from_attribute_value = np.zeros(len(attribute_values))
        for i, attribute_value in enumerate(attribute_values):
            sample_indexes_with_attribute_value = np.where(dataset[:, attribute] == attribute_value)
            div = len(sample_indexes_with_attribute_value) / dataset_length
            ii_from_attribute_value[i] = -div * np.log(div)
        intrinsic_info = np.sum(ii_from_attribute_value)
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
        selected_attribute = -1

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
        
        if (len(self.most_important_features) < 5):
            self.most_important_features.append(selected_attribute)

        node = TreeNode(selected_attribute)
        used_attributes.append(selected_attribute)

        # For each value of attribute
        attribute_values = np.unique(dataset[:, selected_attribute])
        for attribute_value in attribute_values:
            sample_indexes_with_attribute_value = np.where(dataset[:, selected_attribute] == attribute_value)
            samples_with_attribute_value = dataset[sample_indexes_with_attribute_value]
            labels_with_attribute_value = labels[sample_indexes_with_attribute_value]
            if len(samples_with_attribute_value) == 0:  # atribute value not seen in this node, do nothing
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
        root = self.root
        while isinstance(root, TreeNode):
            root_attribute = root.attribute
            attribute_value_of_x = x[root_attribute]
            if str(attribute_value_of_x) in root.subtrees:
                root = root.subtrees[str(attribute_value_of_x)]
            else:  # attribute value was not in the training data
                return self.max_labels(root)  # return the most common label in the subtree

        # came to the leaf
        unique_labels = np.unique(root.labels)
        labels = root.labels
        max_counted_label_index = -1
        max_label_count = -1
        for i, label in enumerate(unique_labels):
            count = np.sum(labels == label)
            if count > max_label_count:
                max_label_count = count
                max_counted_label_index = i
        predicted_label = unique_labels[max_counted_label_index]
        return predicted_label

    def train(self):
        self.most_important_features = []
        self.root = self.ID3__(np.array(self.dataset), np.array(self.labels), [])
        print("Training completed")
        # print("Most important three features: ", self.features[self.most_important_features])
        print(f"For criterion {self.criterion}:")
        for i, feature in enumerate(self.most_important_features):
            print(f"    most important feature {i}: {self.features[feature]}")
