# Mystery-Art-Object-Recognition
A CNN(convolutional neural network) model to classify images of artworks of ancient cities and to recognize the style of mystery artworks. Final project for MTH 353 Seminar in Deep Learning &amp; ARH 212 Ancient Cities and Sanctuaries.

## Brief Abstract
Convolutional Neural Network (CNN) is a subtype of Neural Networks that is mainly used for applications in image recognition. For our final project, we plan to train a CNN that recognizes artworks from ancient Near Eastern, Egyptian, Minoan, Mycenaean, Greek, Etruscan, Roman that were introduced in the ARH 212 course. The training data of this CNN would be pictures of artworks from those cultures, and we will split the pictures into small fractions so that we can both have more data for training and focus on details of the pictures. The label for the training data will be the culture origin of the artworks in the photo. With this trained neural network, we will choose a photo of a mystery object from one of the categories and ask the neural network to predict what culture or time period the mystery object is from. This will be a classification problem.

## Data
Data used for this project comes from WikiArt, the MET collection, and several directly from Google search engine. WikiArt images were used for prototype testing, which is the "Final-Project-Prototype" file. It yields good accuracy, but the dataset is poorly balanced with poor resolution.

The final training dataset uses images from the MET collection, and has the categories as follows: Near East[^1] 150 images, Egyptian 250 images, Greek 200 images, and Roman 50 images. Images only have one category in the MET collection, and has at least 72*72 resolution. To balance the data distribution, I also tested 1)Near East 150 images, Egyptian 300 images, Greek 250 images[^2], and 2)Egyptian 250 images, Greek+Roman 250+50=300 images[^3]. 1 yields an 80% accuracy for Greek; and 2 yields a 67% overall accuracy rate.

[You can find the link to download the images I used in this link.]([https://www.google.com](https://drive.google.com/drive/folders/1dqcusMaq_19r6rkIIgJHlU9Mt2Pexwxt?usp=drive_link)https://drive.google.com/drive/folders/1dqcusMaq_19r6rkIIgJHlU9Mt2Pexwxt?usp=drive_link)

[^1]: The Near East category is a discrete set of artworks in the following categories: Phrygian, Assyrian, Iran, Achaemenid, Urartian, Israelite, Scythian, Babylonian, Parthian, Cypriot, Hillite, and Xiong Nu.
[^2]: This combination is to offset the really small Roman category.
[^3]: This combination is to have a binary classification with equal number of artworks. However, as indicated by the final result, Greek has a similar level of resemblance to both Egyptian and Roman, meaning that adding Roman to Egyptian might makes the model more difficult to recognize, even thought the two category has equal number of data. 

For testing, or the "Mystery Art Object Reconition", I tested out with the images in "Mixed_Categories", which are images that have multiple styles as labels or don't have a settled style from the MET collection.

You can see a snippet of the image data[^4] here: 

<img src="https://github.com/Yuhan-Wang-yw/Mystery-Art-Object-Recognizer/assets/102437257/10aa8471-0d85-4ea0-85b0-dd60e1aa0015" alt="Image Data Snippet" width="450"/>
<br />
<br />
Some Image augenmentation is applied to prevent overfitting as follows: 
<img src="https://github.com/Yuhan-Wang-yw/Mystery-Art-Object-Recognizer/assets/102437257/49abbf64-650e-4122-a797-d2abb23731ad" alt="Image Augmentation Preview" width="450"/>

[^4]:You can see that images are rescaled to the same size (256*256) without preserving its original ratio, because after testing, not preserving its ratio makes the model to perform slightly better. My guess is that it's more important for the model to know its outer shape, instead of focusing on the detailed pattern and color.

## Model
The models I used are CNNs, one Keras and one Tensorflow. (Keras is model 1, Tensorflow is model 2) Essentially they are the same thing, but the Keras one is more nuanced with multiple layers, while the Tensorflow one is really a high-level, basic CNN, with 3 hidden layers.

I trained both model with the original dataset(4 categories)-- the Keras model yields better result on the overall accuracy. However, the Keras model takes around 2.5 hours to train itself, so Tensorflow basic model is used for the different category experimenting and the final testing. For the number of epochs, I just go with typical 25. The validation result somehow does not improve as well, and the model's performance is not as stable. Therefore, 25 is a good amount to reach a good accuracy while not being too long to train.
<img width="400" alt="Accuracy & Loss of Model 2 training" src="https://github.com/Yuhan-Wang-yw/Mystery-Art-Object-Recognizer/assets/102437257/c9fc654b-545b-483f-b544-c65eb2ede468" >

## Evaluation & Results
Here is a table showing the overall results.
|       Model       |                Data               |  Accuracy  |
| ----------------- |:---------------------------------:| :---------:|
| Model 2, epoch 10 |              Wiki Art             |    0.77    |
| Model 1, epoch 15 |          MET Collection           |    0.64    |
| Model 2, epoch 25 |          MET Collection           |    0.53    |
| Model 2, epoch 25 |  MET: Egyptian, Near East, Greek  |    0.57    |
| Model 2, epoch 25 |  MET: Egyptian, Greek+Roman       |    0.63    |

The model overall reached an accuracy around 60%, which is not as ideal. Because the training takes much time, and the artwroks from ancient cultures indeed have many similar features, it is hard to know whether this is a good stopping point, or there are some potential places to work on to improve the model. The model's performance at this stage seems to really be dependent on the data available for one category.

Results of 2 models on the original dataset:
<img width="750" alt="2 Model on Original Dataset, Accuracy Bar Plot" src="https://github.com/Yuhan-Wang-yw/Mystery-Art-Object-Recognizer/assets/102437257/c5a7cf95-0334-480c-9420-d1fdcceb4d21" >
<img width="750" alt="2 Model on Original Dataset, Confusion Matrix" src="https://github.com/Yuhan-Wang-yw/Mystery-Art-Object-Recognizer/assets/102437257/e9cbab5d-68db-489e-949e-71ab9cb7ef2a">

Then coming to different categorizations, it definitely helps to improve the model's performance. I think it will be more interesting to dig deeper into the model and find out what features of the images lead to its prediction at this moment.

Results of Tensorflow model on 3 different categorizations:
<img width="750" alt="Tensorflow Model on 3 categorizations" src="https://github.com/Yuhan-Wang-yw/Mystery-Art-Object-Recognizer/assets/102437257/ec07a264-99f6-4b1f-8549-6c64d32dbb22">

Then coming to the prediction, the model definitely is more confident to (or prefer to) predicting Greek and Egyptian, as it never predicts the artwork to be Roman even though it is trained on 4 categories.
<img src="https://github.com/Yuhan-Wang-yw/Mystery-Art-Object-Recognizer/assets/102437257/9b9e6b40-2a0e-4c5e-a7d3-adac73499838" alt="Model Prediction on Mystery Object Confusion Matrix" width="450"/>

Greek and Egyptian are the two that are predicted the most, while Roman is never predicted as the highest probable culture origin. To officially implement the model in real art recognition, more work is needed to decipher how the model predicts, which will more effectively provide insights about what features link to what prediction.

## Final Thoughts
This project concludes my study of art of ancient cities and of machine learning in general. The prolonged data collection phase allowed me to go over a huge number of artworks in different cultures, and the overall digging of data also provides insights about the number of artworks we have in record for different cultures– Egyptian being the most abundant. Moreover, the model is an innovative foundation for future application of ML to analyze artworks of ancient cities. It is transferable to similar research and will be more informative if fed with abundant high-quality data. Lastly, the models’ performance indicates the fact that artworks of ancient cities are highly related and share similarities in many aspects– material, pattern, subject matter, and so on. Greek being the culture in between Egyptian and Roman has equal resemblance to both styles in the model’s eyes, but the fact that the model performs well in predicting Greek also points out some unique features of Greek artworks in the model’s perception. Also, when concluding the model’s prediction, I only looked at the class with the highest probability, but it will be interesting to examine the probability distribution and analyze the artwork as a collection of styles. Overall, the project intersects art history with computer science and effectively utilizes modern technology to scientifically analyze artworks in their contexts, and I am excited to continue similar approaches for creative exploration of art through the computer’s perception.

### Mathematics in the project
Neural network embodies many important math concepts and functions as its foundation. Linear algebra, multivariate calculus and basic notions of statistics will all be used in the training of a CNN. Linear algebra is used for example, when each neuron calculates its input with its weights, it performs a dot product of two matrices or vectors. Multivariate calculus is used in the evaluation and training of the CNN, for example, when performing backpropagation, we can apply gradient descent to the function to find a local minimum that minimizes the loss of a CNN. Basic statistics, like probability, is the very basis of the CNN because the CNN outputs a probability of how likely one artwork is from one culture.

##### Resources:
DuBois, J. (n.d.). Using Convolutional Neural Networks to Classify Art Genre . John Carroll University Carroll Collected. https://collected.jcu.edu/cgi/viewcontent.cgi?article=1147&context=honorspapers

Zoe Falomir, Lledó Museros, Ismael Sanz, Luis Gonzalez-Abril, Categorizing paintings in art styles based on qualitative color descriptors, quantitative global features and machine learning (QArt-Learn), Expert Systems with Applications, Volume 97, 2018, Pages 83-94, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2017.11.056.

Egypt, Samos, and the archaic style in greek sculpture - sage journals. (n.d.-b). https://journals.sagepub.com/doi/10.1177/030751338106700108 
