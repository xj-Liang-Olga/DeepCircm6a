# DeepCircm6a
DeepCircm6a is  a tool for circRNA-m6A predicting. Here we build a model based on deep learning algorithm to predict the m6A site on circRNA. The model has good prediction ability after training, the prediction accuracy has reached 0.976, and the prediction performance of the test set is also excellent. You can download this tool directly to predict your data, or use your data to train it before use.
# Dependencies
Python v3.8.8;  
numpy v1.19.4;  
pandas v1.2.4;  
pytorch v1.6.0;  
biopython v1.77;  
sklearn v0.24.2;  
argparse v1.1;  
matplotlib v3.4.3;  
tqdm v4.55.1
# Usage
## 1.Pretreatment  
Before inputting the data into our prediction tool, you need to extract your sequence into a 51bp sequence centered on base A.  
## 2.Prediction of m6A site  
In this part, you can directly use the model we have built to predict the m6A site. The command is as follows:  
  
```python predict.py input_fa (51bp) -model_path (path of checkpoint) -outfile (filename of output)```  
  
## 3.Use other data to train the model  
If you want to train a set of prediction models yourself, you can use the following commands to complete it. The specific parameters can be obtained through ```python train. py - h```.  
  
```python train.py -pos_fa (pos.fa) -neg_fa (neg.fa) -outdir (dir_name)```  
  
If you want to test the performance indicators of the model, you can select another data for the model and use the following command to obtain it.  
  
```python test.py```
