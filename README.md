## 1. Preparing datasets
To prepare the dataset in HDF5 format, run the following command:  

```bash
python prepare_dataset.py
```

You can modify the dataset name in the `configuration.txt` file.  

### Available Datasets  
[DRIVE](http://www.isi.uu.nl/Research/Databases/DRIVE/), [STARE](http://cecas.clemson.edu/~ahoover/stare/), [CHASE_DB](https://blogs.kingston.ac.uk/retinal/chasedb1/), [HRF](https://www5.cs.fau.de/research/data/fundus-images/)

## 2. Run the Full Workflow  
The workflow consists of two main steps:  
1. **Training** an FCN (Fully Convolutional Network) model  
2. **Testing** the trained FCN model  

### Train an FCN Model  
To start training, run the following command:  

```bash
python pytorch_train.py
```  

- Model architecture and training settings are configured in `configuration.txt`.  
- Number of sub-images (`N_subimgs`) for different datasets:  
  - DRIVE & STARE: **20,000**  
  - CHASE_DB1: **21,000**  
  - HRF: **30,000**  
  - Private dataset: **90,000**  
- Training parameters:  
  - **Epochs** (`N-epochs`): 100  
  - **Batch size** (`batch_size`): 35  
  - **Learning rate** (`lr`): 3e-3  

### Test the FCN Model  

To test the trained model, run the following command:  

```bash
python pytorch_predict_fcn.py
```  

- **Stride settings** for testing:  
  - DRIVE, STARE, CHASE_DB1: `stride_height = 5`, `stride_width = 5`  
  - HRF, Private dataset: `stride_height = 10`, `stride_width = 10`
 
### Pretrained Model
Our pretrained model used in paper are in [GoogleDrive](https://drive.google.com/drive/folders/126apXEpe_ZIhmOYQ68N50ZhIonwbcFjP?usp=sharing).


### Quantitative Evaluation  

After completing all processes, run the following command to obtain evaluation results:  

```bash
python evalution.py
```  

## 3. Citation  

## 4. Acknowledgement
I'm very grateful for my co-first author, Chee Hong Lee's(cheehong200292@gmail.com) diligent efforts and contributions.  

