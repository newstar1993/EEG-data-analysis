# EEG-data-analysis
Analysis of EEG Dataset

Reference paper : https://katlabatfsu.weebly.com/uploads/9/0/8/2/90824315/cavanagh_meyer___hajcak_2017.pdf

Preprocessing EEG dataset
in chronological order

1.Re-referencing: while recording, all channels are referenced to Cz, with this first step, now the mastoid electrodes are used as comparators/reference now

2.Filters: Zero phase shift Butterworth filters. 

Global filter settings:

Low cutoff: 0.1 Hz, time constant 15.91549, order 4

High cutoff: 30 Hz, order 4 

3.Segmentation -500 ms before response until 1000 ms after response

4.Ocular correction, regression based (“Gratton&Coles” -method)

Artifact rejection: in channels with artifacts (as defined below), affected segments will be marked and then excluded from further analysis
criteria: 

Check Gradient:  Maximal allowed voltage step: 50 µV/ms

Mark as Bad: 	 Before Event: 200 ms	 After Event: 200 ms

Check Difference (Max-Min):  Maximal allowed difference of values in intervals: 300 µV

Interval Length: 100 ms

5.Average based on the Segments, leaving out the artifact-affected epochs on certain channels, one average each for errors and correct responses

6.Baseline correction: -200 to 0 (this is a linear transformation, where all averages are aligned so that the mean activity of this period equals zero)
Then, the averaged data are usually exported as mean values during a certain period, e.g. ms 0-100 after the responses 


 A figure from a review was made, first over several trials of one person (up right), then also averaging across multiple people (down left). The ERN is usually maximal at FCz, but also clearly apparent on adjacent electrodes, like Cz, Fz, FC1, FC2, etc. Electrodes are named according.
 
![mean_eeg](https://user-images.githubusercontent.com/33709389/52156669-9bd5c880-2657-11e9-8ae1-0f464ba63bee.png)

Goal: employ machine learning methods with this data to predict whether a person is GAD patient or not.

Data preprocessing:  Get the result of FCZ response and extract the mean of correct and incorrect response. use data Augmentation and removal of outliers.

Measurements: Do 10 fold cross validation on patients level for 40 times. Obtain the average test accuracy.


Final results on results.xlsx at this repository.
Major codes on main.py

unable to upload the original data which is in mat file. The file in preprocessing is used to extract features from original data.
