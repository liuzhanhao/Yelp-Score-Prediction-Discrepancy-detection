# CS-6220
 Yelp Score Prediction & Fake Review Detection

###### tags: `Yelp` `Classification` `Text Analysis`
------

### File Structure
	|README.md: The readme file
	|
	|.gitignore: The git ignore file
	|
	|/task1: The folder for the 1st task: Sentiment Analysis & Rate Prediction
	|-----
	|	|--classify.py:		Code for the rate prediction & sentiment analysis		
	|	|--yelp_utils.py:	Helper code for the classify.py
	|
	|/task2: The folder for the 2nd task: Automatic Discrepancy Detector [1]
	|-----
	|	|--test_metric.py:  Code for testing metrics for Automatic Discrepancy Detector
	|
	|/task3: The folder for the 3rd task: Machine-generated/Human-written Review Detector |-----
	|	|/yelp_review_generator:	The folder for the yelp review machine generator
	|	|--feature_analysis.py:     Code for feature exploration
	|	|--task3.py:		   Code for the machine-generated/human-written review detector
	|	|--hotel_data.csv:			The Deceptive Opinion Spam Corpus for hotel reviews [2]
	|



### Task1: Sentiment Analysis & Rate Prediction

- Classifier to predict sentiment and rating based on text reviews

- Run: `python classify.py <nb/svm/lr> <True/False> <number in [0, 100]> filename` 

- Dataset: [Yelp Dataset Challenge](https://www.yelp.com/dataset/challenge) [3]


### Task2:

- Use the classification model to detect fake reviews
- 

### Task3:







### Reference

[1\] [Yelp Review Generator](https://github.com/fcchou/yelp_review_generator)

[2] M. Ott, Y. Choi, C. Cardie, and J.T. Hancock. 2011. [Finding Deceptive Opinion Spam by Any Stretch of the Imagination](http://myleott.com/op_spamACL2011.pdf). In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies.

[3] Yelp Dataset, 2014. URL: http://www.yelp.com/dataset_challenge