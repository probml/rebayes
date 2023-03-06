import os

from demos.showdown.classification import classification_train as benchmark

if __name__ == "__main__":
    output_path = os.environ.get("REBAYES_OUTPUT")