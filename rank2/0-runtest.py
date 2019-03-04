import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
list_files("../")




%run 1-0-data_preprocess.ipynb
%run 1-1-EdgePrepare.ipynb
%run 1-2-Graph_embeddings.ipynb
%run 2-1-FeatureExact.ipynb
%run 2-2-extract_features.ipynb
%run 3-1-lgb_embedding.ipynb
%run 3-2-lightgbm_model.ipynb
%run 4-1-blend_model.ipynb
%run 4-2-rule_adjust.ipynb