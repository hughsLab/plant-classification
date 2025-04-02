# plant-classification
plant classification is in conjunction with Genus-and-Species-Android-app repo. 
The plant data comes form the universtiy of Glasgo, there are 10, 001 classes of different species and genus of plants. 
Most classes have only a few to several hundred data points, making training very very hard. 
There are several solutions forward, adding to the class sets which are lacking using methods such as augmentation, webcrawl to pluck more data ect. 
This has worked to a certain degree, whilst dipping into uncerntantiy in solution output.
My current method is break the data down into many trained models and somehow feed that into the application, this is yet to be explored!

Below are training results based on the models provided.

![image](https://github.com/user-attachments/assets/c2d0bd63-2e3e-4d6c-bdfc-89836a2e8e77)
Training validation of 10 parent chunks with 1K classes in each chunk for 60 epochs.
This includes Dropout layer to reduce overfitting and Data augmentation for Overfitting generally occurs when there are a small number of training examples. Which is what we have here!!

50 epochs 
![image](https://github.com/user-attachments/assets/7e16a150-4de5-4248-9201-0414cc5844f6)


