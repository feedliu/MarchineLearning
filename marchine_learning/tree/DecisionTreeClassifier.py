#encoding=utf-8
'''
implement tree algorithm
'''

from math import log
import matplotlib.pyplot as plt

class DecisionTreeClassifier:
    '''
    implement decision tree classifier
    '''

    def __init__(self):
        pass

    def create_dataset(self):
        '''
        create the dataset
        '''
        dataset = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        return dataset, labels

    def calculate_info_entropy(self, data):
        '''
        claculate the info entropy of the data
        Args:
            data: the last column of data is label
        '''
        m = len(data)
        label_counts = {}
        for x in data:
            label = x[-1]
            if label not in label_counts.keys():
                label_counts[label] = 0
            label_counts[label] += 1
        entropy = 0.0
        for key in label_counts:
            prob = float(label_counts[key]) / m
            entropy -= prob * log(prob, 2)
        return entropy

    def split_dataset(self, data, axis, value):
        '''
        split the data by the special feature and the special feature value
        Args:
            data: the dataset to be splited
            axis: the feature by which the data will be splited
            value: the feature value the returned data equal to in the special feature
        '''
        ret_data = []
        for x in data:
            if x[axis] == value:
                reduced_feat_vec = x[:axis]
                reduced_feat_vec.extend(x[axis + 1 :])
                ret_data.append(reduced_feat_vec)
        return ret_data

    def choose_best_feature_to_split(self, data):
        '''
        choose the best feature which has the maxmize info gain
        '''
        #get the number of the features
        num_features = len(data[0]) - 1
        base_entropy = self.calculate_info_entropy(data)
        best_info_gain = 0.0
        best_feature = -1
        #compute the info gain for each feature
        for i in xrange(num_features):
            feature_list = [example[i] for example in data]
            unique_values = set(feature_list)
            new_entropy = 0.0
            for value in unique_values:
                sub_data = self.split_dataset(data, i, value)
                prob = len(sub_data) / float(len(data))
                new_entropy += prob * self.calculate_info_entropy(sub_data)
            info_gain = base_entropy - new_entropy
            if(info_gain > best_info_gain):
                best_info_gain = info_gain
                best_feature = i
        return best_feature

    def marjority_count(self, label_list):
        '''
        get the most label by the label list
        '''
        class_count = {}
        for vote in label_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
            class_count[vote] += 1
        #sort the dist, return a list whose item is a tuple containing key and vlue.
        sorted_class_count = sorted(class_count.items(), key=lambda k : k[1], reverse=True)
        return sorted_class_count[0][0]

    def create_tree(self, dataset, labels):
        class_list = [example[-1] for example in dataset]
        #if the dataset has only one class, return it.
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        #if all features has been splited, return the class which has the max
        if len(dataset[0]) == 1:
            return self.marjority_count(class_list)
        best_feature = self.choose_best_feature_to_split(dataset)
        best_feature_label = labels[best_feature]
        tree = {best_feature_label:{}}
        del (labels[best_feature])
        feature_values = [example[best_feature] for example in dataset]
        unique_values = set(feature_values)
        for value in unique_values:
            sub_labels = labels[:]
            tree[best_feature_label][value] = self.create_tree(self.split_dataset(dataset, best_feature, value), sub_labels)
        return tree

    def tree_plotter(self, tree):
        '''
        plot entry
        '''
        self.decision_node = dict(boxstyle='sawtooth', fc='0.8')
        self.leaf_node = dict(boxstyle='round4', fc='0.8')
        self.arrow_args = dict(arrowstyle='<-')
        self.__create_plot__(tree)

    def __plot_node__(self, node_txt, center_point, parent_point, node_type):
        '''
        plot node
        '''
        self.plot.annotate(node_txt, \
                                xy=parent_point, \
                                xycoords='axes fraction', \
                                xytext=center_point, \
                                textcoords='axes fraction', \
                                va='center', \
                                ha='center', \
                                bbox=node_type, \
                                arrowprops=self.arrow_args)

    def __create_plot__(self, tree):
        '''
        init plot
        '''
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.plot = plt.subplot(111, frameon=False, **axprops)
        self.totalw = float(self.get_leaf_number(tree))
        self.totalD = float(self.get_tree_depth(tree))
        self.x_off = -0.5 / self.totalw
        self.y_off = 1.0
        self.plot_tree(tree, (0.5, 1.0), '')
        plt.show()

    def get_leaf_number(self, tree):
        '''
        get leaf number
        '''
        num_leafs = 0
        first_str = tree.keys()[0]
        second_dict = tree[first_str]
        for key in second_dict.keys():
            if type(second_dict[key]).__name__ == 'dict':
                num_leafs += self.get_leaf_number(second_dict[key])
            else:
                num_leafs += 1
        return num_leafs

    def get_tree_depth(self, tree):
        '''
        get tree depth
        '''
        max_depth = 0
        first_str = tree.keys()[0]
        second_dict = tree[first_str]
        for key in second_dict.keys():
            if type(second_dict[key]).__name__ == 'dict':
                this_depth = 1 + self.get_tree_depth(second_dict[key])
            else:
                this_depth = 1
            if this_depth > max_depth : 
                max_depth = this_depth
        return max_depth

    def plot_mid_text(self, cntr_point, parent_point, txt_string):
        '''
        plot mid text
        '''
        x_mid = (parent_point[0] - cntr_point[0]) / 2.0 + cntr_point[0]
        y_mid = (parent_point[1] - cntr_point[1]) / 2.0 + cntr_point[1]
        self.plot.text(x_mid, y_mid, txt_string)

    def plot_tree(self, tree, parent_point, node_txt):
        '''
        plot tree
        '''
        num_leafs = self.get_leaf_number(tree)
        depth = self.get_tree_depth(tree)
        first_str = tree.keys()[0]
        cntr_point = (self.x_off + (1.0 + float(num_leafs)) / 2.0 / self.totalw, self.y_off)
        self.plot_mid_text(cntr_point, parent_point, node_txt)
        self.__plot_node__(first_str, cntr_point, parent_point, self.decision_node)
        second_dict = tree[first_str]
        self.y_off = self.y_off - 1.0 / self.totalD
        for key in second_dict.keys():
            if type(second_dict[key]).__name__ == 'dict':
                self.plot_tree(second_dict[key], cntr_point, str(key))
            else:
                self.x_off = self.x_off + 1.0 / self.totalw
                self.__plot_node__(second_dict[key], (self.x_off, self.y_off), cntr_point, self.leaf_node)
                self.plot_mid_text((self.x_off, self.y_off), cntr_point, str(key))
        self.y_off = self.y_off + 1.0 / self.totalD

    def classify(self, tree, feat_labels, test_data):
        '''
        predict the label of the test data
        '''
        first_str = tree.keys()[0]
        second_dict = tree[first_str]
        feat_index = feat_labels.index(first_str)
        for key in second_dict.keys():
            if test_data[feat_index] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    label = self.classify(second_dict[key], feat_labels, test_data)
                else:
                    label = second_dict[key]
        return label

def main():
    tree = DecisionTreeClassifier()
    data, labels = tree.create_dataset()
    myTree = tree.create_tree(data, labels)
    print myTree
    tree.tree_plotter(myTree)

if __name__ == '__main__':
    main()