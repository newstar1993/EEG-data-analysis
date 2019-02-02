# EEG-data-analysis
Analysis of EEG Dataset
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
 
 ![]images(mean_eeg.png)
