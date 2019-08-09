README for dataset TRIANGLES


=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

There are OPTIONAL files if the respective information is available:

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DS_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i

=== Description === 

TRIANGLES is one of the synthetic datasets of randomly generated graphs (Knyazev et al., 2019).
The task is to count triangles (3-node cliques) in a graph. 
The dataset provides unnormalized attention values for each node in a graph in the 
TRIANGLES_node_attributes.txt file, which denote to how many triangles a particular node belongs.
The first 30000 graphs should be used for training, the next 5000 graphs for validation 
and the last 10000 graphs for testing. 10000 test graphs consist of 2 different subsets with 5000 graphs in each. 
The first test subset is similar to the training and validation sets and 
the second test subset has larger graphs.


=== References ===

@article{knyazev2019understanding,
  title = {Understanding Attention and Generalization in Graph Neural Networks},
  author = {Knyazev, Boris and Taylor, Graham W and Amer, Mohamed R},
  journal = {arXiv preprint arXiv:1905.02850},
  year = {2019}
}
