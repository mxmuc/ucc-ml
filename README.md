# Welcome to our UCC repository!


## 💪 Motivation

We, a group of TUM students, addressed the issue of optimizing a SAP Support Ticket System in the context of a practical course in Machine Learning for Information Systems. The main goal was to increase the processing efficiency of an incoming ticket. This was attempted through conducting a full ML pipeline on a dataset containing information of about 12000 tickets. We tried multiple ML approaches such as various types of regression, random-forest classifiers, linear SVC, auto-encoded clustering and the classification of NLP embeddings. The classification based on NLP embeddings delivered promising accuracy of 81.6% for the support level classification of German and English tickets, while the rest could not live up to our hopes. We relied heavily on Natural Language Processing Transformers like BERT since most of the information on a ticket is condensed in text format. We also took it upon us to discover new insights and were able to aggregate and visualize our results in the form of a Tableau dashboard.



## 👥 Group Members

| 👤                    | 📨                           |
|-----------------------|------------------------------|
| Agnes Pilch           | agnes.pilch@tum.de           |
| Bavly Shenouda        | bavly.shenouda@tum.de        |
| Maximilian Pfleger    | maximilian.pfleger@tum.de    |
| Janik Schnellbach     | janik.schnellbach@tum.de     |


## 💻 Setup

To get started, it is necessary to install the `requirements.txt`.
As our used dataset as well as our created model are relatively large, we did not include them in the repo.
Instead, we'd like to ask you to download them from our [Drive](https://drive.google.com/drive/folders/1Pi9XV9RcRePrITCkfHs8njhgbE0YbsIy?usp=sharing)
Simply put both folders (`data` and `model`) into the root folder.
The overall structure should look like this:

```bash
.
├── requirements.txt
├── data
│   ├── tickets.csv
│   ├── FAQ_questions.txt
│   ├── status_log.csv
│   ├── tickets_postprp.csv
│   ├── tickets_postprp.pkl
│   ├── tickets.csv
│   └── tickets
│        └── ...
│ 
├── model
│   └── clf.bin
│
├── scripts ...
```

## 📖 Short Summary of Scripts
Overall, there are three different scripts folders with notebooks addressing different aspects of our work. We will provide a short summary here; more details can be found in the corresponding scripts.

### scripts_data
Here, we first extract, clean, and prepare the initially provided dataset. In addition, we also show some visualization

### scripts_classification
This folder comprises the most successful part of our work. We built a binary classifier that detects whether a message belongs to a first or second level support.

### scripts_other
Next to our classifier, we also spent a lot of effort on other ideas for potential ML models. Even though these were not quite as successful, we still want to include them here.

## 🎨 Additional Material
On top of our code, we also created an interactive dashboard as well as a GUI for classifying messages. Both can be found in the corresponding folders. In order to run the dashboard, one needs a running [Tableau Desktop](https://www.tableau.com/de-de/products/desktop) application. TUM students can get a free license for it.
