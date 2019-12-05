# Production Machine Learning Systems

In real-world production ML systems are large ecosystems of which a  robust model code itself is just a small part. The rest, consist of code that perform critical functions some of which we have already seen in the pipeline in my [other repo](https://github.com/Sylar257/Google-Cloud-Platform-with-Tensorflow). In this project, we will be learning good characteristics that make for good ML system beyond its ability to make *good predictions*. 

## Contents

[***Architecting Production ML System***](https://github.com/Sylar257/GCP-production-ML-systems#architecting): components we need to design and things to consider as system architects

[***Ingesting data for Cloud-based analytics and ML***](https://github.com/Sylar257/GCP-production-ML-systems#Data_ingesting): bringing data to the cloud

[***Designing adaptable ML systems***](https://github.com/Sylar257/GCP-production-ML-systems#adaptable_ml_system): how to mitigate potential changes in real-world that might affect our ML system

[***High performance ML systems***](https://github.com/Sylar257/GCP-production-ML-systems#high_performance_ML_system): choose the right hardware and removing bottlenecks for ML systems

[***Hybrid ML systems***](https://github.com/Sylar257/GCP-production-ML-systems#Hybrid_ML_system): high-level overview of running hybrid systems on the Cloud



## Architecting

In a Cloud-base ML ecosystem, the model code usually only account for **5%** of the total code. This is because in order to keep a ML system running in production, we need a lot more than just *computing the model’s outputs for a given set of inputs.* 

![Other_components_of_ML_system](images/Other_components_of_ML_system.png)

### data ingestion

![data_ingestion](images/data_ingestion.png)

The first component we should consider is the **data ingestion**. 

*   The input data could come from *streaming data ingestion pipeline* from a mobile, device web service, etc. For **streaming data** we use **PubSub**.

*   Another possibility is **structured data** that live in a data warehouse. Thus we might use a service such as **BigQuery**.
*   If we are transforming data from training that we can train on it later, we might use **Cloud Storage**

### data quality

Machine learning models are only *as good as their training data*. Unlike catastrophic bugs which are easy to find, small data bugs could be preternaturally hard to locate even though *they can still significantly degrade model quality over time*.Because bugs can be disastrous and hard to find, we need **data analysis** and **data validation** components. 

![data_analysis_validation](images/data_analysis_validation.png)

Exploratory Data Analysis(**EDA**) is all about understanding the distribution of our data, which is the first step in detecting small data bugs. Then we could perform helpful pre-processing for our models such as **eliminating outliers**, **adjust skewness of data**, and **correcting wrongly registered data from providers**.

Data validation is all about finding out is our data healthy or not:

1.  is the new distribution similar enough to the old one?
2.  Are all expected features present?
3.  Are any unexpected features present?
4.  Does the feature have the expected type?
5.  Does an expected proportion of the examples contain the feature?
6.  Do the examples have the expected number of values for feature?

### data transformation

The data transformation component allows for **feature wrangling.** It can do things like *generate feature to integer mappings.* Critically, whatever mappings that are generated mush be saved and reused at **serving time**. Failure to do this consistently results in a problem called **training-serving skew**. Common tools to use for data transformation are: **Dataflow, Dataproc, and Dataprep**.

![data_transformation](images/data_transformation.png)

We normally crate data transformation useing **dataflow & dataprep**. Its important create dataflow pipelines that are par of the model graph using TF transform and **export the transform function** for use at serving time.

### Trainer

The trainer is responsible of training our model. It should be able to support **data parallelism and model parallelism**. This could be the primary reason that we might choose TF+GCP over PyTorch since the Google has done such a amazing job behind the scence so that we, as end users, could do data and model parallelization with minimum amount of code. In addition, it should also automatically *monitor and log everything*, and support the use of experimentation. Fanally, the trainer should also support **hyperparameter tuning**. This could be naive grid search, random search or even Tree-Parzen-based Bayesian optimizer. 

![Trainer](images/Trainer.png)

There are two produts that aligned with this component in GCP: **ML Engine** which provides the managed service for TensorFlow and **GKE(Kubeflow)** which provides a managed environment for **hybrid ML models** in Kubeflow. **ML Engine** is a managed excution environment for TensorFlow that allows us to instantly scale up to hundreds of workers, and it’s automatically integrated with the three other core components: the tuner, logging and serving. It also have a built-in concept of models inversion, allowing for easy **AB testing** and there is no lock-in so that we can take our train model and run it anywhere.

### Model Evaluation and Validation

The model evaluation and validation components have one responsibility: to ensure that the models are good before moving them into the production environment. The goal is to ensure that users’ experience aren’t degraded. There are two main things that we care about with respect to model quality. **How safe** the model is to serve, and the model’s **prediction quality**.

**Safe model** refers to low chance of crashing or cause errors in the serving system when being loaded or when sent on expected inputs. It also shouldn't use more than the expected amount of resources, like memory.

**Model evaluation** is part of the *iterative process* where teams try and improve their models. However, because it’s expensive to test on **live data**, experiments are generally run off-line first. It’s in this setting where **model evaluation** takes place. Model evaluation is essentially assessing the model with respect to some *business-relevant metric* like **AUC, ROC curves or cost-weighted error**. If the model meets their criteria, then it can be pushed into production for a **live experiment.**

In contrast to the *model evaluation component*, which is human facing, the **model validation** component is not. Instead, we *design a **fixed threshold** that evaluates the model and **alerts engineers** when things go awry*.  For this we often use **TFX** for model analysis which is a part of Google internal production ML system.

![model_evaluation_and_validation](images/model_evaluation_and_validation.png)

### Serving and logging

The serving component should be **low-latency** to respond to users quickly. **Highly efficient** so that many instances can be run simultaneously, and **scale horizontally** in order to be reliable and robust to failures. In contrast to training when we care about scaling with our data, at serving we care about **responding** to variable user demand **maximizing throughput** and **minimizing response latency**. Another important feature is that it should be easy to update to *new versions of the model*. When we get new data or engineered new features, we will want to retrain and push a new version of the model and we would want the system to seamlessly transition to the new model version. More generally, the system should allow us to set up a multi-armed bandit architecture to **verify** which model version are the best.

![Serving](images/Serving.png)

Just as with training, there are two options for serving on GCP. We can either use a fully managed TensorFlow serving service which is **ML Engine** or we can run **TF serving** on **Kubernets engine**. 

![logging](images/logging.png)

The next component is **logging** which is critical for **debugging and comparison**. All logs should be easily accessible and integrated with **Cloud Reliability**.

### Training Design Decision

When training a model, there two *paradigms*, **static training** and **dynamic training**. 

In **static training**, we do what we did in my [other repo](https://github.com/Sylar257/Google-Cloud-Platform-with-Tensorflow): gather the data, partition it, train our model, and then deploy it. In **dynamic training**, we do this repeatedly as data arrive. This leads to the fundamental trade-off between static and dynamic. Static is simpler to build and test, but likely to become stale. Whereas dynamic is harder to build and test, but will adapt to changes. 

Below are some general pros & cons:

![static_dynamic](images/static_dynamic.png)

From another perspective, we can contemplate the trade-off between static and dynamic based on **peakedness** and **cardinality**:

*   **Peakedness** refers to the extent to which the distribution of the prediction workload is concentrated(similar to **kurtosis**). For example, a model that predicts the next word given the current word would be **highly peaked** because a small number of words account for the majority of words used. In contrast, a model that predicted quarterly revenue for all sales verticals, in order to populate a report, will be run on the same verticals every time and with the same frequency for each. So, it would have very low peakedness.
*   **Cardinality** refers to the number of values in the set. In this case, the set is the set of **all possible things we might have to make predictions for**. A model predicting sales revenue given organization division number, would be fairly *low cardinality*. A model predicting lifetime value given a user for an e-commerce platform would be very high cardinality, because the number of users and the number of characteristics of each user, are likely to be quite large.

Taken together, these two criteria create a space. When the cardinality is sufficiently low, we can store the entire **expected prediction workload**. When the cardinality if high because the size of the input space is large, and the workload is not very peaked, we would probably want to use **dynamic training**.

In practice, we often would choose a **hybrid of static and dynamic**, where we statically cache some of the predictions, or responding on demand for the long tail. This works best when the distribution is *sufficiently peaked*.

![hybrid_model](images/hybrid_model.png)

### Deploy trained model on Cloud AI Platform

##### Create a bucket and move our models in

```python
# parameters for creating bucket
REGION=us-central1
BUCKET=$(gcloud config get-value project)
TFVERSION=1.7
# create a new bucket with Cloud Shell
gsutil mb -l ${REGION} gs://${BUCKET}

# copy the files from G-storage to our BUCKET
gsutil -m cp -R gs://cloud-training-demos/babyweight/trained_model gs://${BUCKET}/babyweight
```

##### Deploy trained model

```python
# parameters for AI-platform
MODEL_NAME=babyweight
MODEL_VERSION=ml_on_gcp
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/babyweight/export/exporter/ | tail -1)

# deploy model in the AI-platform
gcloud ai-platform models create ${MODEL_NAME} --regions $REGION
gcloud ai-platform versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version $TFVERSION
```

Now we need two





## Data_Ingesting



## Adaptable_ml_system



## High_performance_ML_system



## Hybrid_ML_system