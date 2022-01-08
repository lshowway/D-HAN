# D-HAN
The source code of D-HAN

#### This is the source code of D-HAN: Dynamic News Recommendation with Hierarchical Attention Network. However, only the code of three tested datasets is uploaded.



##### Update: 2022.1.9
> python 3.6 is used

> NOTE: Since several modules are considered in our method, and many ablation studies have been explored, we do not upload the code of each module, if you are interested in any of them, please contact me. 

> Start from the `run_adressa.py` file, it will use `interactions.py` and `utils.py` file to process news data, use `HAN_DNS_time.py` file to access the model, then continue  `run_adressa.py` file to train, evaluate and compute metrics value.
> `Attention.py` file contains the self-attention, and time embedding; `HAN_DNS_time.py` file contains element-level, sentences-level, news-level, HAN model and dynamic negative sampling core code, etc.
 

> We remove redundant files, only the training files of public dataset Adressa is kept. To run the training process, do the following steps:
> 1. Downloading Adressa dataset from [BUAA drive](AnyShare://赵清华_BY1806168/D_HAN/data.zip) or [Google drive](https://drive.google.com/file/d/1ipW1CClXmwUYIvkcZJp3JUvWbRq_7-oz/view?usp=sharing), note that, the dataset is processed from the full dataset of public Adressa dataset, if you need the original processing file, please contact me.
> 2. Run this command `python run_adressa.py`, parameters can be set according to the paper or kept default. Note that the number of negative samples used in the training phase is 3, but when dynamic negative sampling method is adopted, 50 news items are first randomly selected and then DNS sample 3 items from the 50 items.





------
> Since the main contributions include: (1) We propose to simultaneously capture different granular information, i.e., sentence-, element-, document- and sequence-level information for news recommendation. (2) We propose to recommend news dynamically by a time-aware document-level attention layer, which incorporates the absolute and relative time information. (3) We propose to incorporate negative sampling into the training process to facilitate model optimization.

> You can check the code for their corresponding implementation details, it is very easy to understand. Of course, if you have any questions, please create issues in this repo, and I will respond you as soon as possible.