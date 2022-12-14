import json

from model_learning import *

if __name__ == '__main__':
    SimpleShape1 = load_annotations("annotations/SimpleShape1.csv")
    SimpleShape2 = load_annotations("annotations/SimpleShape2.csv")
    with open("annotations/SpatialSense", 'r') as f:
        SpatialSense = json.load(f)
    X1, Y1 = compute_extendedRLM_on_SimpleShape("images/SimpleShapes1", SimpleShape1, (68, 1, 84, 255), 3, 2)
    X2, Y2 = compute_extendedRLM_on_SimpleShape("images/SimpleShapes2", SimpleShape2, (68, 1, 84, 255), 3, 2)
    train_model(X1, Y1, True)

