import numpy as np
import pandas as pd
from math import log, e
import copy

from sklearn import datasets

class TreeFN:
    def __init__(self, fn):
        self.fn = fn
        
    def __call__(self, *args):
        return self.fn(*args)

# Cost functors
class CostFN:
  # Classes defined here calculate the cost of a decision tree node
  # Classes contain a lambda function that must take two inputs
  # y = true labels, w = predicted label
  # y should be a numpy array
  # w can be a single label or a numpy array or labels
    
    class Misclassification(TreeFN):
        def __init__(self):
            super().__init__(lambda y, w: np.count_nonzero(y != w)/len(y)) #Count of misclassified labels divided by number of labels
    class MSE(TreeFN):
        def __init__(self):
            super().__init__(lambda y, w: np.sum((y-w)*(y-w))/len(y)) #Square difference in labels divided by number of labels
    class Entropy(TreeFN):
        def __init__(self):
            def entropy(y, w):
                value,counts = np.unique(y, return_counts=True) #Get unique labels and associated frequencies
                norm_counts = counts / counts.sum() #Probability for each label (P(y ==c))
                return -(norm_counts * np.log(norm_counts)).sum() #entropy formula
            super().__init__(entropy)
    class Gini(TreeFN):
        def __init__(self):
            def gini(y, w):
                value,counts = np.unique(y, return_counts=True) #Get unique labels and associated frequencies
                norm_counts = counts / counts.sum() #Probability for each label (P(y ==c))
                return(sum(norm_counts*(1-norm_counts))) #p(y=c)*(1-p(y=c))
            super().__init__(gini)

# Fit functors
class FitFN:
  # Classes defined here determine how a node is labeled
  # Classes contain a lambda function that must take 1 input
  # labels is a numpy array of labels
        
    class Classification(TreeFN):
        def __init__(self):
            def mode(labels):
              values, counts = np.unique(labels, return_counts = True)
              return values[np.argmax(counts)]
            super().__init__(mode) #Return most frequent label
    class Regression(TreeFN):
        def __init__(self):
            super().__init__(lambda labels: np.mean(labels)) #Return mean of labels

# Stop functors
class StopFN:
  # Classes defined here determine when tree splitting stops
  # Classes contain a lambda function that must take 1 input
  # node is the node in the decision tree that is being evaluated for splitting
    
    class NoStop(TreeFN):
        def __init__(self):
            super().__init__(lambda node: False) #Never return stop condition
    class Depth(TreeFN):
        def __init__(self, depth):
            super().__init__(lambda node: node.depth > depth) #Return stop condition once depth has exceeded max depth
    class MinNodeCost(TreeFN):
        def __init__(self, node_cost):
            super().__init__(lambda node: node.node_cost < node_cost) #Return stop condition once node cost is sufficiently small
    class MinImprovement(TreeFN):
        def __init__(self, min_improvement):
            super().__init__(lambda node: node.parent.node_cost - node.node_cost < min_improvement) #Return stop condition once improvements become sufficiently small

class DecisionTreeNode:

    def __init__(self, x, y, parent, depth, condition, label, left, right, label_fn, cost_fn):
        self.x = x  # Training features
        self.y = y  # Training labels
        self.parent = parent #Parent node
        self.depth = depth  # Node depth in tree
        self.condition = condition  # Condition used to split node
        self.label = label_fn(
            self.y) if label is None else label  # Label of node
        self.left = left  # left child of node
        self.right = right  # right child of node
        self.label_fn = label_fn  # function used to determine the node label
        self.cost_fn = cost_fn  # function used to determine the split cost
        self.split_cost = self.cost_fn(self.y, self.label) #np.finfo(np.float32).max  # Split cost of node
        self.is_split = False  # Node is not initially split
        self.node_cost = self.cost_fn(self.y, self.label)  # Cost of node

    def try_split_by_feature_val(self, feature, val):
        # Returns True if node splits, False otherwise

        # handles exception if split results in left or right being empty
        try:
            # Get left split
            left_x = self.x[self.x[:, feature] <= val]
            left_y = self.y[self.x[:, feature] <= val]
            left_label = self.label_fn(left_y)  # Calulate label for left split

            # Get right split
            right_x = self.x[self.x[:, feature] > val]
            right_y = self.y[self.x[:, feature] > val]
            # Calculate label for right split
            right_label = self.label_fn(right_y)
        except:
            return False
        # Calculate cost of splitting
        left_cost = self.cost_fn(left_y, left_label)
        right_cost = self.cost_fn(right_y, right_label)
        split_cost = (len(left_y) * left_cost + len(right_y) * right_cost) / (len(left_y) + len(right_y))

        # Update node split if split is favorable
        if split_cost < self.split_cost:
            # Update split cost
            self.split_cost = split_cost

            # Create left node
            left = DecisionTreeNode(left_x, left_y, #Features and labels
                                    self, #Parent
                                    self.depth + 1, #Node depth
                                    None, #Split Condition
                                    left_label, #Node label
                                    None, None, #Children nodes
                                    self.label_fn, self.cost_fn) #Label and cost functions
            # Create right node
            right = DecisionTreeNode(right_x, right_y, #Features and labels
                                     self, #Parent
                                     self.depth + 1, #Node depth
                                     None, #Split condition
                                     right_label, #Node label
                                     None, None, #Chilren nodes
                                     self.label_fn, self.cost_fn) #Lable and cost functions

            # Set children nodes for parent node
            self.left = left
            self.right = right
            self.condition = lambda data: data[feature] <= val
            self.is_split = True

            return True

        return False

    def is_empty(self):
        return len(self.y) == 0

class DecisionTree:

    def __init__(self, fit_fn=FitFN.Classification(), cost_fn=CostFN.Entropy(), stop_fn=StopFN.NoStop()):
        self.fit_fn = fit_fn
        self.cost_fn = cost_fn
        self.stop_fn = stop_fn

    def fit(self, x, y):
        # store data
        self.x = x
        self.y = y

        # Create root node of decision tree
        self.root = DecisionTreeNode(
            x, y, None, 0, None, None, None, None, self.fit_fn, self.cost_fn)

        return self._fit_tree(self.root)

    def _fit_tree(self, node):
        # Check if stop criterion is met
        if self.stop_fn(node):
            return node

        # Split node if it is favorable to do so
        if self._greedy_test(node):
            #node.left = self._fit_tree(node.left)
            #node.right = self._fit_tree(node.right)
            self._fit_tree(node.left)
            self._fit_tree(node.right)
        return node

    def _greedy_test(self, node):
        # iterate over features
        for feature in range(node.x.shape[1]):
            #Split on median of feature
            #node.try_split_by_feature_val(feature, np.median(node.x[:,feature]))
            # test splitting at each unqie value for feature  
            for val in np.unique(node.x[:, feature]):
                # try to split node
                node.try_split_by_feature_val(feature, val)

        return node.is_split
 
    def predict(self, x):
        #returns the predicted labels for input data
        y = []
        
        #Ensure correct shape of data
        if len(x.shape) == 1:
            x = x.reshape((1,-1))
        for data in x:
            #traverse through tree until leaf node is encountered
            node = self.root
            while node.condition is not None:
                if node.condition(data): #if condition is true then left else right
                    node = node.left
                else:
                    node = node.right
            #Append label of leaf node
            y.append(node.label)
        
        return np.array(y)
    
    def evaluate_acc(self, y_pred, y_true):
        true_pos = len(np.argwhere((y_pred ==1) & (y_true == 1)))
        true_neg = len(np.argwhere((y_pred ==0) & (y_true == 0)))
        
        false_pos = len(np.argwhere((y_pred ==1) & (y_true == 0)))
        false_neg = len(np.argwhere((y_pred ==0) & (y_true == 1)))
        
        return (true_pos + true_neg)/(true_pos + true_neg + false_pos + false_neg)
    
    def leaf_nodes(self):
        #Get all leaf nodes of tree
        return self._leaf_nodes(self.root, [])
    
    def _leaf_nodes(self, node, nodes):
        #leaf nodes helper function
        #Binary tree traversal
        #Explore left all the way, then when no left option available explore right
        if node == None:
            return
        self._leaf_nodes(node.left, nodes)
        if (node.left == None and node.right == None):
            nodes.append(node)
        self._leaf_nodes(node.right, nodes)
        return nodes
    
    def tree_cost(self):
        return sum([node.node_cost for node in self.leaf_nodes()]) #Cost of all leaf nodes
    
    def gen_pruned_trees(self):
        tree = copy.deepcopy(self) #Create a deep copy so that self is not reduced to a tree of size 1
        trees = [copy.deepcopy(tree)] #Insert a deepcopy of the tree so that future mutations of tree are not reflected
        
        #Continue pruning while the root has children
        while tree.root.right is not None and tree.root.left is not None:
            #Get parents of leaf nodes
            leaf_parents = np.array([node.parent for node in tree.leaf_nodes()]) 
            #Find index of node with least impact if children are removed
            ind = np.argmin([node.node_cost - node.right.node_cost - node.left.node_cost for node in leaf_parents]) 
            
            #remove children from parent and set condition to None
            #Converts internal node (parent) into a leaf node
            leaf_parents[ind].left = None 
            leaf_parents[ind].right = None
            leaf_parents[ind].condition = None
            trees.append(copy.deepcopy(tree)) #store a deepcopy of the tree
        
        self.pruned_trees = trees #Store thr list of pruned trees
        
        return trees
    
    def prune_trees_on_validation_set(self, validation_x, validation_y):
        trees = self.gen_pruned_trees()
        cost = []
        
        #Iterate over the trees in reverse order
        #cost[0] corresponds to trees[-1], cost[1] -> trees[-1-1], costs[2] -> tree[-1-2]
        #cost[n] -> trees[-1-n]
        for t in reversed(trees):
            preds = t.predict(validation_x) #get predicted values for validation data
            cost.append(CostFN.MSE()(validation_y, preds)) #calculate mean squared arror for the validation data
            
        ind = np.argmin(cost) #get index corresponding to first occurance of min cost
        return trees[-1 -ind] #return the tree with the smallest mean squared error on the validation set
        

    def print_region(self):
        self._print_region(self.root)

    def _print_region(self, node):
        if node == None:
            return

        self._print_region(node.left)
        if (node.left == None and node.right == None):
            labels = ' '.join([str(y) if y == node.label else '\033[91m' + str(y) + '\033[0m' for y in node.y])
            
            print("prediction: {}, condition: {}, cost: {} \nlabels: {}".format(
                node.label, node.condition, node.node_cost, labels))
        self._print_region(node.right)

#Build Accuracy Tests
cost_fns = [CostFN.Entropy(), CostFN.MSE(), CostFN.Misclassification()]
cost_fn_names = ["Entropy", "MSE", "Missclassification"]
stop_fns = []
stop_fn_names = []
for depth in range(0,10):
    stop_fns.append(StopFN.Depth(depth))
    stop_fn_names.append("Max Depth: {}".format(depth))

for cost in range (0,11):
    stop_fns.append(StopFN.MinNodeCost(cost/10))
    stop_fn_names.append("Min Node Cost: {}".format(cost/10))

data = pd.read_csv(r"C:\Users\Jake\Downloads\hepatitis.data", header = None)
data = data[~(data == '?').any(axis=1)]

data =data.values
data = np.flip(data)
#Random shuffle rows
np.random.shuffle(data)

#Split data into train(60%), validation (40%) and test (20%) set
train, validate, test =np.split(data, [int(.6*data.shape[0]), int(.8*data.shape[0])])

#Run accuracy Tests
results = []
for (cost_fn, cost_name) in zip(cost_fns, cost_fn_names):
    for (stop_fn, stop_name) in zip(stop_fns, stop_fn_names):
        print(cost_name, stop_name)
        #Construct Tree
        tree = DecisionTree(cost_fn = cost_fn, stop_fn = stop_fn)
        #Fit decision tree
        tree.fit(train[:,0:train.shape[1]-1], train[:,train.shape[1]-1])
        
        #Pre pruned accuracy
        predictions = tree.predict(test[:,0:test.shape[1]-1])
        pre_pruned_regions = len(tree.leaf_nodes())
        pre_pruned_acc = tree.evaluate_acc(predictions, test[:,test.shape[1]-1], 2, 1)
        
        #Pruned accuracy
        #Prune to best tree on validation set
        pruned = tree.prune_trees_on_validation_set(validate[:,0:train.shape[1]-1], validate[:,train.shape[1]-1])
        predictions = pruned.predict(test[:,0:test.shape[1]-1])
        pruned_regions = len(pruned.leaf_nodes())
        pruned_acc = pruned.evaluate_acc(predictions, test[:,test.shape[1]-1], 2, 1)
        
        results.append([cost_name, stop_name, pre_pruned_regions, pre_pruned_acc, pruned_regions, pruned_acc])
   
df_hep = pd.DataFrame(data = results, columns = ["CostFN", "StopFN", "Pre Pruned Regions", "Pre Pruned Accuracy", "Pruned Regions", "Pruned Accuracy"])


#get data
data = pd.read_csv(r"C:\Users\Jake\Downloads\messidor_features.arff", header = None)
data = data[~(data == '?').any(axis=1)]

data =data.values
print(data)
data = data[data[:,0] == 1]
print(data)
#Random shuffle rows
np.random.shuffle(data)

#Split data into train(60%), validation (40%) and test (20%) set
train, validate, test =np.split(data, [int(.6*data.shape[0]), int(.8*data.shape[0])])

#Run accuracy Tests
results = []
for (cost_fn, cost_name) in zip(cost_fns, cost_fn_names):
    for (stop_fn, stop_name) in zip(stop_fns, stop_fn_names):
        print(cost_name, stop_name)
        #Construct Tree
        tree = DecisionTree(cost_fn = cost_fn, stop_fn = stop_fn)
        #Fit decision tree
        tree.fit(train[:,0:train.shape[1]-1], train[:,train.shape[1]-1])
        
        #Pre pruned accuracy
        predictions = tree.predict(test[:,0:test.shape[1]-1])
        pre_pruned_regions = len(tree.leaf_nodes())
        pre_pruned_acc = tree.evaluate_acc(predictions, test[:,test.shape[1]-1], 1, 0)
        
        #Pruned accuracy
        #Prune to best tree on validation set
        pruned = tree.prune_trees_on_validation_set(validate[:,0:train.shape[1]-1], validate[:,train.shape[1]-1])
        predictions = pruned.predict(test[:,0:test.shape[1]-1])
        pruned_regions = len(pruned.leaf_nodes())
        pruned_acc = pruned.evaluate_acc(predictions, test[:,test.shape[1]-1], 1,0)
        
        results.append([cost_name, stop_name, pre_pruned_regions, pre_pruned_acc, pruned_regions, pruned_acc])
   
df_mes = pd.DataFrame(data = results, columns = ["CostFN", "StopFN", "Pre Pruned Regions", "Pre Pruned Accuracy", "Pruned Regions", "Pruned Accuracy"])

