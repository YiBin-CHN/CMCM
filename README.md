# Non-Autoregressive-Cross-Modal-Coherence-Modelling
Codes for our paper "Non-Autoregressive Cross-Modal Coherence Modelling" in ACM MM 2022
## Prerequisites
Pytorch v1.6.0 
Python 3
nltk 3.6.7
gensim 4.1.2
torchvision 0.7.0

Our implementation adopts 300d GloVe for word embedding. 
The glove.42B.300d.txt can be downloaded from https://nlp.stanford.edu/data/glove.42B.300d.zip

## Data and Preprocessing
### Dataset
The SIND dataset can be downloaded from the Visual Storytelling website https://visionandlanguage.net/VIST/dataset.html

The preprocessed dataset can be downloaded from https://www.filecad.com/o8kB/data_ordering.zip

The raw TACoS Multi-Level Corpus can be downloaded from https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/tacos-multi-level-corpus/ 

Image frames and corresponding sentences for video can be seen from https://www.filecad.com/fQa8/TACoS.zip 

### Preprocessing
All the images are resized to 256x256 by resize.py. 

```
python resize.py --image_dir [image_dir] --output_dir [output_dir]
```

The build_vocab.py script can be used to build vocabulary.  The obtained pkl file can be placed in the voc directory.

## Training & Test
Run maintbi.py and mainibt.py script to train and test the model for sentence and image ordering. 

```
python main.py
```
