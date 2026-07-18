import numpy as np
import networkx as nx
from skimage.segmentation import slic
from skimage.graph import rag_mean_color

def build_graph_from_image(image_data, num_segments= 100):
    """
    Takes the image_data
    and constructs a graph where each node represents a superpixel and edges represent adjacency between superpixels.
    """

    #1 Segment the image into superpixels
    labels = slic(image_data, n_segments = num_segments, compactness = 10, start_label = 1)

    #2 Build the Region Adjacency Graph (RAG)
    rag = rag_mean_color(image_data, labels)
    nx_graph = nx.Graph(rag)

    return nx_graph, labels


