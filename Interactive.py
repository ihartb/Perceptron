from ProcessData import ProcessData
from Perceptron import Perceptron

class Interactive:
    def __init__(self):
        self.faceTest = ProcessData().loadFaceImages('facedatatest.txt')
        self.faceTestLabel = ProcessData().makeFaceLabels('facedatatestlabels.txt')
        self.faceTrain = ProcessData().loadFaceImages('facedatatrain.txt')
        self.faceTrainLabel = ProcessData().makeFaceLabels('facedatatrainlabels.txt')

        self.digitTest = ProcessData().loadDigitImages('testimages.txt')
        self.digitTestLabel = ProcessData().makeDigitLabels('testlabels.txt')
        self.digitTrain = ProcessData().loadDigitImages('trainingimages.txt')
        self.digitTrainLabel = ProcessData().makeDigitLabels('traininglabels.txt')

        self.epoch = 3
        self.percent = 50
        self.pcptF = Perceptron([70 * 60, 1], 1)
        self.pcptD = Perceptron([28 * 28, 10], 1)

    def run(self):
        user_input = None
        print()
        print("The percent of training data used to train the algorithm is set to 50%")
        print("The number of epochs to train the algorithm is set to 3")
        print("Select an action by typing the lowercase letter corresponding to the action you wish to take")
        print()

        print("a: Show possible actions\n"
              "p: Change percentage of training data\n"
              "e: Change number of epochs\n"
              "f: Test on faces\n"
              "d: Test on digits\n"
              "q: Quit")

        while user_input != "q":
            print()
            user_input = input("Choose an action from above : ")

            if user_input == "a":
                print()
                print("a: Show possible actions\n"
                      "p: Change percentage of training data\n"
                      "e: Change number of epochs\n"
                      "f: Test on faces\n"
                      "d: Test on digits\n"
                      "q: Quit\n")

            if user_input == "p":
                print()
                self.percent = int ( input("Choose a percentage: ") )
                print()

            if user_input == "e":
                print()
                self.epoch = int ( input("Choose number of epochs: ") )
                print()

            if user_input == "f":
                print()
                print("Testing Faces")
                print("Percent of Training Data: "+str(self.percent)+"%")
                print("Number of epochs: "+str(self.epoch))
                self.pcptF.demo(self.faceTrain, self.faceTrainLabel, self.percent, self.epoch,self.faceTest, self.faceTestLabel)


            if user_input == "d":
                print()
                print("Testing Digits")
                print("Percent of Training Data: "+str(self.percent)+"%")
                print("Number of epochs: "+str(self.epoch))
                self.pcptD.demo(self.digitTrain, self.digitTrainLabel, self.percent, self.epoch, self.digitTest, self.digitTestLabel)


Interactive().run()


