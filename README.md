<h2>Sampling assignment</h2><br>
<b>Sampling</b> - It is the practice of analyzing a subset of all data in order to uncover the meaningful information in the larger data set.<br>
In this assignment we study about various sampling techniques and their methods to calculate the sample size, how well different models provide their results and based on that we generate an accuracy table.<br>
<br>
Now let us discuss the methodology.
<h2>Methodology</h2>
<h3>1. Balancing the dataset</h3>
Before creating samples make sure that the dataset is balanced. Here to balance our "Creditcard_data.csv" dataset I use SMOT oversampling technique. You can download the balanced dataset from "Balanced.csv".<br>
<h3>2. Generation of samples</h3>
There are various types of sampling techniques available out of which we use Simple random sampling, Stratified sampling, Systematic sampling, Cluster sampling, Convenience sampling to create samples of our dataset.<br>
<br>
<b>Simple Random Sampling</b> - In this method of sampling, the data is collected at random i.e. each item has an equal probability of being in the sample, which makes the method impartial.<br>
<br>
<b>Systematic Sampling</b> - In this method to form a sample, every nth term or item of the numbered items is selected. For example, if 10 out of 200 people are to be selected for investigation, then these are first arranged in a systematic order. After that one of the first 10 people would be randomly selected. In the same way, every 10th person from the selected item will be taken under the sample. <br>
<br>
<b>Stratified Sampling</b> - Stratified or Mixed Sampling is a method in which the population is divided into different groups, also known as strata with different characteristics, and from those strata some of the items are selected to represent the population.<br>
<br>
<b>Cluster sampling</b> - Cluster sampling is a type of probability sampling in which every and each element of the population is selected equally, we use the subsets of the population as the sampling part rather than the individual elements for sampling.<br>
<br>
<b>Convenience Sampling</b> - It is a method of collecting data in which the investigator selects the items from the population that suits his convenience.

<br>
<h3>3. Applying 5 differnt models</h3>
Now let us apply five different machine learning classification models on our samples to identify the best accuracy.<br>
<br>
Here, I apply,<br>
1. Random Forest<br>
2. Decision tree<br>
3. Extra tree Classifier<br>
4. Logistic Regression <br>
5. SVM <br>
<h3>4. Generating accuracy table</h3><br>

|               | Simple Random | Systematic | Cluster | Stratified | Convenience |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Logistic Regression | 89.583333 | 85.714286 | 84.403670 | 93.684211 | 97.727273 |
| SVM | 88.541667 | 90.909091 | 83.486239 | 94.736842 | 97.727273 |
| Extra tree classifier | 100.000000 | 100.000000 | 99.082569 | 98.947368 | 97.727273 |
| Decision Tree | 97.916667 | 94.805195 | 93.577982 | 94.736842 | 97.727273 |
| Random Forest | 98.958333 | 98.701299 | 96.330275 | 98.947368 | 97.727273 |
<h3>5. Conclusion</h3>
On observing the table it is clearly visible that <b>Extra tree classifier</b> gives us the best accuracy. Also note this that the accuracy also depends on the sample taken each time you run the code.


