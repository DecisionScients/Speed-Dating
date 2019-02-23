![Logo of the project](https://raw.githubusercontent.com/DecisionScients/Speed-Dating/master/logo.png)

# Speed Dating
> Analysis and Predictive Modeling of Mate Selection in Speed Dating

What can speed dating tell us about mate selection and partnership in modern Western society? What are the dyadic phenomena manifest during mate selection? What are the characteristics and the relationships among them that indicate a partnership?


## Installing / Getting started

To install the project, execute:

```shell
packagemanager install awesome-project
awesome-project start
awesome-project "Do something!"  # prints "Nah."
```

Here you should say what actually happens when you execute the code above.

### Initial Configuration

This project requires the following configuration: 

## Developing

Here's a brief intro about what a developer must do in order to start developing
the project further:

```shell
git clone https://github.com/DecisionScients/Speed-Dating
cd Speed-Dating/
conda env create
source activate speed-dating
```

### Building

Instructions to build project after code changes:

```shell
./configure
make
make install
```

Executing the above code results in:

### Deploying / Publishing

In order to deploy this project to a server, execute the following: 

```shell
packagemanager deploy awesome-project -s server.com -u username -p password
```

Executing the above results in:

## Features

This project analysis and machine learning project has the following features:    
* Analysis of mate selection preferences 
* Examination of dyadic phenomena in successful mating
* Predict probability of mate choice in a speed dating context
* Predict probability of a match in a speed dating context
* Online predictive model of mate choices based upon user profile

## Contributing

If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests are warmly welcome.

## Links

- Project homepage: [Speed-Dating]([https://github.com/DecisionScients/Speed-Dating)
- Repository: [Speed-Dating Repo](https://github.com/DecisionScients/Speed-Dating)
- Issue tracker: [Issues](https://github.com/DecisionScients/Speed-Dating/issues)
  - In case of sensitive bugs like security vulnerabilities, please contact
    jjames@decisionscients.com directly instead of using issue tracker. We value your effort
    to improve the security and privacy of this project!
- Related projects:
  - [Online-Dating](https://github.com/DecisionScients/Online-Dating)
  - [Journal of Statistical Education Paper on Using OkCupid Data for Data Science Courses](https://github.com/rudeboybert/JSE_OkCupid)

## Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
    
## Licensing

The code in this project is licensed under 3-clause BSD license.
