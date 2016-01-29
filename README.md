# VaguenessDetection
Deep Learning based Tool for Multi-lingual Vagueness Detection

For any details, contact Bargav Jayaraman (bargavjayaraman@gmail.com)

Note: NR-GoldSet is present in 'annotated dataset/raw/' folder. It consists of 3 datasets each containing 500 sentences manually annotated for vagueness. The 500 sentences of each language (English/Spanish/Portuguese) are translations of each other across the three langauges.

Requirements:
- Download and install Python 2.x or above
- Download and install Theano from 'http://deeplearning.net/software/theano/' (elaborate instructions for installing both python and theano are given on this website)

Instructions for using the Vagueness Detection Tool:
- For detecting vagueness in Spanish/Portuguese sentences, run the script 'VaguenessDetector.py' and provide the Spanish/Portuguese sentences (For starters, sample test sentences are provided in the script as input for vagueness detection)
- For training the Tool from the scratch in Spanish/Portuguese, run the script 'VaguenessDetectorTrain.py' (the script also evaluates the performance of the Tool on annotated gold-set of 500 English/Spanish/Portuguese sentences)
- For training the Tool in any other European language, run the script 'VaguenessDetectorTrain.py' and replace the Spanish/Portuguese training data with the required European language (parallel data can be found in Europarl dataset 'http://www.statmt.org/europarl/'). It is, however, the responsibility of the User to preprocess the dataset and align the parallel English-Target language sentences. Feel free to contact the author for further help.
- For any other non-European language, a considerable size of parallel corpus should be created containing English-Target langauge sentence pairs.

Due to size limitation, some training dataset files are missing from the repo. Please fetch the below files as directed:
-> Missing file: 'english_train_data.txt' from 'spanish' folder
Directions: - Fetch the 'eng.zip' file from 'https://drive.google.com/open?id=0Byz0OQf_YuHvaXFzcU1Sb1g2bnc'
            - Unzip the file and save as 'english_train_data.txt' in 'spanish' folder

-> Missing file: 'english_train_data_labels.txt' from 'spanish' folder
Directions: - Fetch the 'eng_labels.zip' file from 'https://drive.google.com/open?id=0Byz0OQf_YuHvYS13eGhtdm1aOE0'
            - Unzip the file and save as 'english_train_data_labels.txt' in 'spanish' folder

-> Missing file: 'spanish_train_data.txt' from 'spanish' folder
Directions: - Fetch the 'sp.zip' file from 'https://drive.google.com/open?id=0Byz0OQf_YuHvUmtWeGxHSG5CakE'
            - Unzip the file and save as 'spanish_train_data.txt' in 'spanish' folder

-> Missing file: 'english_train_data.txt' from 'portuguese' folder
Directions: - Fetch the 'eng2.zip' file from 'https://drive.google.com/open?id=0Byz0OQf_YuHvMENGN3EtZ3BlWDQ'
            - Unzip the file and save as 'english_train_data.txt' in 'portuguese' folder

-> Missing file: 'english_train_data_labels.txt' from 'portuguese' folder
Directions: - Fetch the 'eng_labels2.zip' file from 'https://drive.google.com/open?id=0Byz0OQf_YuHvVzJDbnNTbUl4MjA'
            - Unzip the file and save as 'english_train_data_labels.txt' in 'portuguese' folder

-> Missing file: 'portuguese_train_data.txt' from 'portuguese' folder
Directions: - Fetch the 'pt.zip' file from 'https://drive.google.com/open?id=0Byz0OQf_YuHvVTdlaW1mS0hHVUk'
            - Unzip the file and save as 'portuguese_train_data.txt' in 'portuguese' folder
