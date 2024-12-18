# MLP Movie Classification

```plaintext
├── hw1_test.csv
└── hw1_train.csv
└── requirements.txt
├── run.py
└──submission.csv
```

Goal: Create a model that can accurately tag knowledge graph core relations to movie-related utterances. 

Feature engineering and hyperparameter tuning play a crucial role in machine learning algorithms. In constructing a model to predict relations for a given movie-related utterance, feature engineering proved to be an effective method for improving accuracy. The usage of dense vector embeddings, batch data processing, data shuffling, and data resampling increased validation accuracy by 3%. Hyperparameter tuning, such as increasing the learning rate and decreasing regularization, boosted validation accuracy to 96% and test accuracy to 81%.

To run:

```bash
pip install -r requirements.txt
python run.py hw1_train.csv hw1_test.csv submission.csv
```
