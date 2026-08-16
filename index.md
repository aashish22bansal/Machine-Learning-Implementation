# Machine Learning Implementation

Machine learning and statistics worked through from first principles — derivations
written out, algorithms implemented, results plotted. Every page on this site is a
runnable Jupyter notebook; nothing here is pseudocode.

## Where to start

**New to the material?** Follow the sidebar top to bottom. It is ordered as a course:
data → statistics → visualization → supervised → unsupervised → ensembles.

**Looking for one algorithm?** Use the search box, or jump straight to a chapter.

## What's inside

| Chapter | Covers |
|---|---|
| **Understanding Data** | Data types, structure, and what to look at first |
| **Data Analysis and Statistics** | Descriptive measures, correlation, confidence intervals, hypothesis testing |
| **Data Visualization** | matplotlib from basics through styles and small multiples, with exercises |
| **Supervised Learning** | Preprocessing, linear and multivariable regression, logistic regression, naive Bayes, confusion matrices |
| **Unsupervised Learning** | k-means, mean shift, clustering quality, Gaussian mixture models, market segmentation |
| **Ensemble Learning** | Decision trees, random forests, feature importance, class imbalance, grid search |

## Running the notebooks

Every page has a download button in the top right — grab the `.ipynb` and run it
locally. To run the whole repository:

```bash
git clone https://github.com/aashish22bansal/Machine-Learning-Implementation.git
cd Machine-Learning-Implementation
pip install -r requirements.txt
jupyter lab
```

Notebooks read their datasets from CSV files sitting next to them, so run them from
their own directory.

## About this site

Built with [Jupyter Book](https://jupyterbook.org) directly from the notebooks in the
[source repository](https://github.com/aashish22bansal/Machine-Learning-Implementation).
Pages render from saved notebook outputs, so what you see is what the code produced when
it was last run.

Corrections and additions are welcome as pull requests.
