# DeepADN-PLUS
This project builds a deep learning pipeline based DeepADN-PLUS model to detect the calls of adult male white headed langurs in audio files recorded by automatic recording units, and ultimately find the audio containing the target call among many recordings.

![Model](image/pipeline.png)

# Initialization

## Setup

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/yhzdsk/XXX.git
cd xxx
```

## Dependencies

Our implementation is based on [PyTorch](https://pytorch.org). We recommend using `conda` to create the environment and install dependencies.Select the appropriate `cudatoolkit` version according to your system:

```
conda create --name XXX python=3.8
conda activate XXX
conda install conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Prepare the data

Generate a data list for the next step, with `audio_path` as the audio file path. Users need to store the audio dataset in the `dataset/audio` directory in advance, with each folder containing a category of audio data. Each audio data should be at least 3 seconds long, such as` dataset/audio/bird calls/···`. `Audio `is the location where the data list is stored, and the format of the generated data category is` Audio Path \ t Audio Corresponding Category Label`. The audio path and label are separated by a tab `\ t`. Readers can also modify the following functions according to their own way of storing data.

Taking `audio.zip` as an example, this is a dataset of wild white headed langur calls created by our research team, which includes two categories: positive calls from adult male white headed langurs and negative sounds without white headed langur calls. The following is a function for generating a data list for this dataset. If readers want to use this dataset, please download and extract it to the `dataset` directory, and change the code for generating the data list to the following code.

