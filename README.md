# Master Thesis: Speaker Recognition Based on Gaussian Mixture Models and Neural Networks

Python code to perform text-independent Speaker Identification using Gaussian Mixture Models 
and Neural Networks. We use data from the Librispeech ASR corpus dataset but other datasets can be used.

### Prerequisites

Python 3
Librosa library
Sci-kit learn library
Librispeech ASR corupus dataset available at: http://www.openslr.org/12  

 You need to download this two files from the website:  
    1- train-clean-100.tar.gz [6.3G]   (training set of 100 hours "clean" speech )  
    2- train-clean-360.tar.gz [23G]   (training set of 360 hours "clean" speech )  
    
    
### Installing

1- You need to install the prerequisites and to download the dataset.  

2- Create a new directory called "audio" inside the data folder.  

3- Place the "train-clean-100" and "train-clean-360" un-zipped folders in this directory.  

4- Your file structure has to be the following:  


.
|-data  
|---|-audio  
|-------|-train-clean-100  
|-------|-train-clean-360  


## Running the tests

To train and test the different systems, simply open the desired .py file:  

For example: GMM.py  

Now, set the desired number of speakers, changing the value of the "n_speakers" variable:  

For example: n_speakers = 2.  

Then, run the .py file.  


## Authors

* **Fernando Fragío Sánchez**

## Acknowledgments

* Thanks to my supervisor Jesús Cid-Sueiro.
