#  Prob-Sparse Attention with Stacked Group Convolution for Wireless Signal-based Human Activity Recognition
This repository offers a deep learning framework tailored for human activity recognition using WiFi signals. It complements our research paper titled "Prob-Sparse Attention with Stacked Group Convolution for Wireless Signal-based Human Activity Recognition". The paper is accepted by 2024 International Conference on Wireless Communications and Signal Processing (i-WCSP'24).

For ease of performance comparison, the code is compatible with two public datasets: NTU-Fi HAR and NTU-Fi-HumanID. To initiate the program, simply execute the "run.py" file. Dataset can be downloaded via [WiFi-CSI-Sensing-Benchmark](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark).

# Paper Abstract
With the advancement of Internet of Things, WiFi signal-based Human Activity Recognition (HAR) has demonstrated great potential in various domains. Existing WiFi-based HAR systems pursue high recognition accuracy, but often struggle in achieving model lightweightness. To address this issue, we jointly consider the spatial-temporal correlations of WiFi channel state information. Our proposed HAR method employs an encoder-only Transformer with ProbSparse attention to extract crucial global features from time-series sensor data. Furthermore, it utilizes a stacked group convolutional structure to further encode local temporal and spatial features, respectively, thereby realizing effective extraction of key spatial and temporal characteristics, as well as fusion of global and local features. Experimental results demonstrate that the proposed model achieves an exceptional mean average precision exceeding 99.9\% for action recognition across two public datasets: NTU-Fi HAR and NTU-Fi Human-ID, outperforming several state-of-the-art models such as ViT, TCN and BiLSTM. Meanwhile, utilizing ProbSparse attention, our model exhibits a significant improvement in training complexity compared to several state-of-the-art models such as ResNet50, vanilla Transformer, ViT and TCN.

# Supplementary Results

Comparison with reference deep learning models (all using the average accuracy of k-fold cross validation):

| Method   | NTU-Fi HAR |       |       |       | NTU-Fi HumanID |       |       |       |  
|----------|------------|-------|-------|-------|----------------|-------|-------|-------|  
|          | Acc (%)    | F1 (%)| Params (M) | FLOPs (M) | Acc (%)    | F1 (%)| Params (M) | FLOPs (M) |  
| ResNet18 | 98.75      | 99.35 | 11.184  | 334.63  | 95.51        | 97.23 | 11.188  | 334.63  |  
| ResNet50 | 99.69      | 99.84 | 23.551  | 572.38  | 97.50        | 98.59 | 23.567  | 572.40  |  
| LiteHAR  | 95.83      | 95.89 | 0.012   | 12316.01| 84.01        | 83.49 | 0.012   | 12316.01|  
| ViT      | 91.56      | 94.45 | 2.989   | 1636.88 | 81.94        | 87.99 | 2.991   | 1636.89 |  
| GRU      | 96.88      | 98.19 | 0.079   | 157.57  | 97.60        | 98.74 | 0.079   | 157.57  |  
| BiLSTM   | 98.12      | 98.85 | 0.209   | 419.84  | 91.84        | 95.08 | 0.210   | 419.84  |  
| TCN      | 93.18      | 93.02 | 1.317   | 2629.09 | 98.98        | 98.97 | 1.317   | 2629.09 |  
| Our Model| 100        | 100   | 3.550   | 553.87  | 99.93        | 99.93 | 3.215   | 476.93  |


# Citation 
[1] Y. Dao, H. Zhang, S. Feng, J. Fang and W. Wang, "ProbSparse Attention With Stacked Group Convolution for Wireless Signal-based Human Activity Recognition," to appear in Proceedings 2024 International Conference on Wireless Communications and Signal Processing (WCSP'24).
