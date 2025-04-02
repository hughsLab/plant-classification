# plant-classification

Plant classification is in conjunction with the Genus-and-Species-Android-app repository. 
The plant data comes from the University of Glasgow. There are 10,001 classes of different species and genus of plants. 
Most classes have only a few to several hundred data points, making training very, very hard. 
There are several solutions forward, such as adding to the class sets which are lacking using methods such as augmentation and web crawling to gather more data, etc. 
This has worked to a certain degree, while dipping into uncertainty in solution output.
My current method is to break the data down into many trained models and somehow feed that into the application. This is yet to be explored!

Below are training results based on the models provided.

![image](https://github.com/user-attachments/assets/c2d0bd63-2e3e-4d6c-bdfc-89836a2e8e77)
Training validation of 10 parent chunks with 1K classes in each chunk for 60 epochs.
This includes a Dropout layer to reduce overfitting and Data augmentation. Overfitting generally occurs when there are a small number of training examples, which is what we have here!!

Yolo model identifying genus and species correctly.
![val_batch0_labels](https://github.com/user-attachments/assets/681e0cf3-5ed0-4b73-bbfa-c8f661914ac8)
