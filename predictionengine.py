
from gooey import Gooey, GooeyParser
import logging
import webbrowser
import os
from imageai.Prediction import ImagePrediction

# disables irrelevant warnings
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Gooey decorator used to create GUI for program based upon command line interface
# parameters specify program name, button appearances, and directory of image icon location to be used with the GUI
@Gooey(program_name="RecognizeMe!", show_restart_button=False, show_success_modal=False, image_dir="/Users/chadmcguire/Local/Python Final Project/images")
def main():

    # GooeyParser used in place of ArgumentParser in order to specify widgets
    parser = GooeyParser(description="Image Recognition Search for Wikipedia")

    # facilitates file selection for later image processing
    parser.add_argument(
        "Filename", help="Select the the image file to process", widget="FileChooser")
    arg = parser.parse_args()

    print("Your image is processing...")

    # ImageAI library setup, sets specifications for image prediction and processing
    execution_path = os.getcwd()
    prediction = ImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(
        execution_path + "/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    prediction.loadModel()

    # call of predictImage function, returns object predictions and probabilities
    predictions, percentage_probabilities = prediction.predictImage(
        arg.Filename, result_count=1)

    # output testing of prediction and probabilities, used for debugging
    # for index in range(len(predictions)):
    #     print(predictions[index], " : ", percentage_probabilities[index])

    # if object probability is greater than 20%, proceed with output
    if percentage_probabilities[0] > 20:

        # removes underscores in string with spaces for cleaner output
        prediction_object = predictions[0].replace('_', ' ')

        print("Success!\nA " + prediction_object +
              " has been detected.\nPlease see your default web browser to view the Wikipedia page.\n")

        # calls fetch_webpage in order to load correct Wikipedia page
        fetch_webpage(predictions[0])

    # if object probability is less than 20%, print below and do NOT load webpage
    else:
        print("Sorry!\nWe couldn't find an object in your image with high probability.\n")

    return


# fetch Wikipedia page for object passed in as extension
def fetch_webpage(extension):

    # construct web address
    url = "https://en.wikipedia.org/wiki/" + extension

    # load webpage in default web browser
    webbrowser.open_new_tab(url)

    return


if __name__ == "__main__":
    main()
