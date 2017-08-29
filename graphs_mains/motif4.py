import os
from graphs_mains import featuresList
from graphs_mains.features_calculator import featuresCalculator
from learning.TagsLoader import TagsLoader
from graphs_mains import main_manager as mm

currentDirectory = str(os.getcwd())

if __name__ == "__main__":
    wdir = os.getcwd()
    file_in = str(wdir) + r'/../data/directed/check/input/graph.txt'
    output_dir = str(wdir) + r'/../data/directed/check/features'

    calculator = featuresCalculator()
    # features_list = featuresList.featuresList(directed=True, analysisType='nodes').getFeatures()
    result = calculator.calculateFeatures(['motif4'], file_in, output_dir, directed=True, analysisType='nodes')

    # classification_cora_result = ['cora_tags']  # , 'Nucleus', 'Membrane', 'Vesicles', 'Ribosomes', 'Extracellular']
    # ml_algos = ['adaBoost', 'RF', 'L-SVM', 'RBF-SVM']
    # directory_tags_path = str(wdir) + r'/../data/directed/cora/tags/'
    # result_path = str(wdir) + r'/../data/directed/cora/results/'
    # tagsLoader = TagsLoader(directory_tags_path, classification_cora_result)
    # tagsLoader.Load()

    gnx = result[0]
    map_fetures = result[1]
    number_of_learning_for_mean = 10.0


