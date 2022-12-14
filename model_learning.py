import csv
import glob

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier

from image import image_processing


def train_model(X=None, Y=None, print_scores=False):
    """
    Train a MLP Classifier from X data on Y classes.
    :param X: Data to learn from ; usually RLM values of both objects and the
    :param Y: Classes where the data of X[i] corresponds to the class Y[i].
    :param print_scores: boolean, used to allow or not the printing of accuracy found during training.
    :return: the trained model.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) # 0.2
    clf = MLPClassifier(hidden_layer_sizes=(4, 448), solver='adam', max_iter=1000)  # 224
    clf.fit(X_train, Y_train)
    if print_scores:
        scores = cross_val_score(clf, X, Y, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    return clf


def compute_extendedRLM_on_SimpleShape(folder, annotations, background, step, force):
    """
    Compute the histograms (RLMs and forces) of all images of the SimpleShape dataset (either S1 or S2).

    :param folder: folder where images are stocked.
    :param annotations: path of the file containing the annotations of a SimpleShape dataset.
    :param background: Background color of the images.
    :param step: Step of the angle to use for the computation of the histograms.
    :param force: type of force to use (0, or 2) for the computation of the f-histogram.
    :return: Two lists: (Histograms and classes).
    """
    X_data = []
    Y_data = []
    for row in annotations:
        print(f"img-{row['obj1']}-{row['obj2']}-{row['nb']}.png")
        if row['nb'] != "?":
            img_name = f"{folder}/img-{row['obj1']}-{row['obj2']}-{row['nb']}.png"
            rlm1, rlm2, forces = image_processing(img_name, background, step, force)
            X_data.append(rlm1 + rlm2 + forces)
            Y_data.append(row["rel"])
    return X_data, Y_data


def load_annotations(csvfile):
    """
    Load an annotated csv file (differs from time_management.read_csv(csvfile) -> no values is changed here, and there
     is no 'result' field).

    :param csvfile: csv file to read data.
    :return: list of the rows of the csv file. Rows are defined as dict.
    """
    rows = []
    with open(csvfile, 'r') as file:
        reader = csv.DictReader(file, delimiter=',', quotechar="|")
        for row in reader:
            rows.append(row)
    return rows

