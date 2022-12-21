# JNotes
My Jupyter Notebooks 📑🧾

# Content
- [Stackoverflow Users Similarities](https://github.com/mehditeymorian/JNotes/tree/main/stackoverflow-user-similarity)
- [Equation Minimization Using Genetic Algorithm](https://github.com/mehditeymorian/JNotes/tree/main/equationMinimizationGeneticAlgorithm)
- [Tag Recommendation for Stackoverflow Questions](https://github.com/mehditeymorian/JNotes/tree/main/stackoverflow)
- [Polynomial Approximation using Genetic Algorithms](https://github.com/mehditeymorian/JNotes/tree/main/genetics/polynomial-approximation)
- [House Price Prediction Decision Tree](house-price-prediction/predict-house-prices.ipynb)
- [Cancer Classification using Regression](cancer-classification/cancer_regression.ipynb)
- [CIFAR10 Image Classification using CNN](cifar10-image-classification/cnn-image-classifier.ipynb)

## Stackoverflow Users Similarities
Calculating Jaccord, Cosine L1NORM, and Cosine L2NORM similarities for top 5 Stackoverflow users based on their votes on questions. [Full Detail](https://github.com/mehditeymorian/JNotes/tree/main/stackoverflow-user-similarity)

## Equation Minimization Using Genetic Algorithm
Starting from an initial population, chromosomes evolve through generations and converge toward an optimal answer. in each generation crossovers take place where offsprings chromosomes are generated from selected parents. also each chromosome have a posibility to mutate. the mutation is necessary to avoid local minimums. In This example we are trying to minimze a square root equation. [Full Detail](https://github.com/mehditeymorian/JNotes/tree/main/equationMinimizationGeneticAlgorithm)

<img src="https://latex.codecogs.com/png.image?\dpi{150}&space;\bg_white&space;a_{1}x_{1}^{2}&plus;a_{2}x_{2}^{2}&plus;a_{3}x_{3}^{2}&plus;a_{4}x_{4}^{2}&plus;a_{5}x_{5}^{2}&plus;...=&space;fitness" title="\bg_white a_{1}x_{1}^{2}+a_{2}x_{2}^{2}+a_{3}x_{3}^{2}+a_{4}x_{4}^{2}+a_{5}x_{5}^{2}+...= fitness" />

## Tag Recommendation for Stackoverflow Questions
A matrix of tags is created and filled by Confidence(Tag1, Tag2). [Confidence](https://en.wikipedia.org/wiki/Association_rule_learning#Confidence) is a type of association rule for calculating closeness of items in a dataset. Confidence is the percentage of all transactions satisfying X that also satisfy Y. [Full Detail](https://github.com/mehditeymorian/JNotes/tree/main/stackoverflow)

## Polynomial Approximation using Genetic Algorithms
Each gene consists of 2 number, one is coefficient and the other is x's power. A chromosomes has many genes which forms a polynomial. for example `1 2 1 3` represent `X^2 + X^3`. Starting from a population of chromosomes, we calculate the difference between the expected polynomial and current one. Then those with highest difference get eliminated.
the process will repeat until a good answer is reached. [Full Detail](https://github.com/mehditeymorian/JNotes/tree/main/genetics/polynomial-approximation)

![process](https://github.com/mehditeymorian/JNotes/blob/main/genetics/polynomial-approximation/assets/1.gif)
