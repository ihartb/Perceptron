# Perceptron
Perceptron is a widely used algorithm that teaches computers to label a set of images. 
This project implements a perceptron algorithm from scratch and then uses it to classify partially processed images of faces and digits. 

## Using the code 

### Relevant Files
#### 1. Images
This folder contains the raw partially processed images and labels for faces and numbers and processed images and labels. 
#### 2. ProcessData.py
This code processes the raw images and labels from the Images file. Additionally, it pickles (serializes) the processed images and labels to be instantly used during runtime without having to process them ever again. These processed images and labels are saved under ProcessedData in Images. 
#### 3. Perceptron.py
Perceptron.py does is the core of this project as it is the learning algorithm itself. It not only trains the algorithm to classify the training images based on training labels, but also tests the algorithm's "knowledge" on a separate set of test images and test labels. 
#### 4. Interactive.py
This code takes user inputs through the terminal to run the project. Users can:
- Change the percentage of total training data to use when training the algorithm (defaulted to 50%)
- Change the number of epochs (number of times to run the training algorithm on the same training data before running it on testing data) (defaulted to 3 epochs)
- Test algorithm on face data
- Test algorithm on digit data 

Both testing algorithms print: 
  - Percentage of training data used for training
  - Number of epochs
  - Time the algorithm took to run
  - Accuracy of algorithms predictions on the test data

### Installation and Running
- Download as zip
- Unzip file
- Open file in your preferred source code editor that supports Python 
- Run Interactive.py
- Follow the commands in the terminal window to see results 

### Dependencies
Numpy (should already be included in the zip) 
If you receive an error which states that numpy module cannot be found, download numpy by typing the following in the terminal using: 
```
pip install numpy
```

## Theory
### Feature Extraction in ProcessData.py
The raw partially processed images are represented using 3 characters: ' ', '+', '#'.
##### Digit Example
```
                            
             ++###+         
             ######+        
            +######+        
            ##+++##+        
           +#+  +##+        
           +##++###+        
           +#######+        
           +#######+        
            +##+###         
              ++##+         
              +##+          
              ###+          
            +###+           
            +##+            
           +##+             
          +##+              
         +##+               
         ##+                
        +#+                 
        +#+                 
                            
```
##### Face Example
```
                                                            
 ####                                                       
     ###                      #                             
        ####                ## #                            
            ######         #    #                           
                  #########      ######                     
 #                                     ##                   
  #                #         ##          #                  
  #               #         #  #          #                 
  #               #         #   #          #                
 #                 #  ######    #          #                
 #                  ##          #           #               
 #                 #            #            #              
 #             ####                           #             
                                              #             
             #                                #             
            #      ###############            #             
           #      #               ####       #              
           #     #                    #     #               
          #     #                      #    #    #          
          #     #                       #        #          
          #    #                         #       #          
          #    #                         #        #         
         #     #                          #       #         
         #    #                           #       #         
         #    #                           #       #         
 #       #    #                           #      #          
 #       #   #                   ######   #      #          
 #       #   #   #    ##      ###          #     #          
 #        #  #    #     ###  #             #     #          
 #        #  #     ##       #       #      #     #          
 #        #  #    #  #      #     ##       #    #           
  #       #  #       #      #    #         #    #           
  #       #   #   #  #      #    #         #    #           
  #       #   #  # ##       #     #####   #    #            
  #           #             #   #         #   ##            
  #           #        #    #   #         #  #  #           
  #           #        #    #   #        #   #  #        ## 
  #            #      #     #   #        #  #   #       #   
               #      #     #   #        #  #  #       #    
               #      #    #   #        #   #  #       #    
               #      #    #   #        #  #   #       #    
             # #      #    #  ##        #  #  #        #    
 #          #  #      #     ##  #       #     #        #    
 #          #        #                  #    #         #    
 #           #       #                  #   #           #   
 #            #      #   ########       # ##            #   
 #             #     #           #      #               #   
 #              #     #         #      #        #       #   
 #              #      ##     ##      #         #       #   
 #              #        #####       #    #     #       #   
 #               #                  #     #    #       #    
 #               #                 #      #    #       #    
 #                ####           ##        #   #       #    
 #                    ##      ###          #   #       #    
                  #     ######         #   #           #    
                  #                   #     #          #    
                 ###                  #   #  #         #    
                #                     #   #  #          #   
               #     #               #     #  #         #   
               #     #              #          #        #   
              #       #  ######    #        #   #           
             #         ##      ####        #     ##         
             #   #                         #       ###      
           ##    #                        #           ##    
         ##      #                        #             #   
        #         #                      #               ## 
      ##          #                      #                  
    ##             #                    #                   
                                                            
```
For both the faces and digits, I scanned each line of the respective .txt file, and matched each character present with a numerical value:
‘ ‘   =  0
‘+’  = .5
‘#’  =  1
The numbers from 0 to 1 represent the darkness of the image at that point. 0 means it is completely white, 1 means it is completely black. 
I simply took the numerical value of each character in the file and added it to a numpy vector. For the digits, each digit was represented in the .txt file using 28x28 characters, therefore our feature was a 784x1 vector (28x28=784) filled with 0,.5, and 1 at the appropriate location. Similarly, for the faces, which were represented using 70x60 characters, we had a 4200x1 vector.

### How the computer learns in Perceptron.py
Perceptron uses linear regression techniques to properly classify images. We train the computer to make a line that takes an x value, and passes it through a linear function to output an y value. This y value may be passed through an activation function to map all predictions either 0 or 1, or between 0 and 1, etc. Classic perceptron takes any y >= 0, makes it a 1, and otherwise a 0. We used a sigmoid activation function that passed y through the sigmoid function: 
$ f = \frac {1}{1+e^{-y}} $
The x value is the extracted feature, and the f value is the computer’s prediction of the label. The learning algorithm continuously changes this line’s placement and its slope in a multidimensional coordinate system until the error between the prediction and label is minimized. 

For digits, we have ten equations for classifying each digit.
y0 = b0 +w0,0x0+w0,1x1+...+w0,783x783
y1 = b1 +w1,0x0+w1,1x1+...+w1,783x783
…
y9 = b9 +w9,0x0+w9,1x1+...+w9,783x783
wi,j=weight for digit i associated with the value at j in the feature vector x
This method lends itself very obviously to linear algebra operations. 



