## Trainer </br>

Development of training PyTorch models is not so simple. The training scripts are not generalizable from project to project. For a project, one has to write custom training loops, keep track of loss and metric calculations, to use learning rate schedulers if necessary, and so on. Once it is done, utilizing these scripts for a different project requires a lot of code changes to be done and it is a time consuming task. </br>

This Trainer makes development of Pytorch training extremely easy and fast while making it as generic as possible from project to project. It helps an individual to focus more on data preprocessing and experimenting various model architectures rather than spending more than sufficient amount of time on writing training scripts. The Trainer also provides many interesting features which can be easily used as required. </br>

</br>
</br>

The Trainer features the following: </br>

- Supports **Model Profiling**: Model Size, Num_Parameters, MACs, FLOPs, Inference Latency </br>

- Supports **Model Training Types**: Single-Input-Single-Output, Single-Input-Multiple-Outputs, Multiple-Inputs-Multiple-Outputs </br>

- Supports **Learning Rate Scheduler**: OneCycleLR </br>

- Supports **Metrics**: Single or multiple metric(s) </br>

- Supports **Callbacks**: EarlyStopping, ModelCheckpoint </br>

- Supports **Loggers**: CSVLogger, TensorBoardLogger </br>

- Supports **Training Precisions**: FP32, FP16, BF16, FP16 AMP, BF16 AMP </br>

- Supports **Gradient Accumulation**: Accumulates gradients over several batches </br>

- Supports **Training History**: Epoch, learning rate, loss and/ or metric(s) </br>

- Supports **Training Progress Bar**: Epoch, learning rate, loss and/ or metric(s) </br>

</br>
</br>

Note: </br>

- By default, the Trainer trains the model with FP32 precision and it only displays the progress bar with information of epoch, learning rate, loss and metric(s) if provided any. All other features can be used as required. </br>

- It is recommended and important that the loss and metric functions must return an averaged value or reduction by 'mean'. </br>

- It is suggested to use PyTorch 2.0 and above, as the Trainer is tested for the same. </br>

</br>
</br>

#### Directory and files information:-
- [notebooks](https://github.com/tpjesudhas/cv_component/tree/main/core/Image_Segmentation/SegFormer/notebooks) -> contains the jupyter notebook. 

- [torch_trainer](https://github.com/tpjesudhas/cv_component/tree/main/core/Image_Segmentation/SegFormer/utils) -> contains the profiling and training utilities.

- [utils](https://github.com/tpjesudhas/cv_component/tree/main/core/Image_Segmentation/SegFormer/utils) -> contains the loss and metrics definition.

- [requirements.txt](https://github.com/tpjesudhas/cv_component/tree/main/core/Image_Segmentation/SegFormer/requirements.txt) -> contains all the required libraries to be installed.

</br>
</br>

<hr>

### Please read the following to understand the features provided by the Trainer: </br>

- **Model Profiling:** It is always good to perform the model profiling before training to get the complexity of the model. It computes the Model Size, Num_Parameters, MACs, FLOPs, Inference Latency. It supports profiling for both devices, "cpu" and "cuda". </br>

- **Model Training Types:** </br>

□ **Single-Input-Single-Output:** The Trainer can train a model that accepts single input and produces a single output, a torch.tensor(). </br>

□ **Single-Input-Multiple-Outputs:** The Trainer can train a model that accepts single input and produces multiple outputs, i.e tuple(torch.tensor(), torch.tensor(), ..., torch.tensor()). </br>

□ **Multiple-Inputs-Multiple-Outputs:** The Trainer can train a model that accepts multiple inputs, i.e. torch.tensor(), torch.tensor(), ..., torch.tensor() and produces multiple outputs, i.e tuple(torch.tensor(), torch.tensor(), ..., torch.tensor()).</br>

- **Learning Rate Scheduler:** The Trainer only supports OneCycleLR scheduler. It is a widely used scheduler and unlike StepLR/ MultiStepLR and many other schedulers, it updates the optimizer's learning rate over each batch. It is based on a 2018 paper titled "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates" (https://arxiv.org/abs/1708.07120) </br>

- **Metrics:** The trainer supports single or multiple metric(s) by defining it as a dictionary, where key is the metric name and its value should be the metric function. It is recommended to use [torchmetrics](https://pypi.org/project/torchmetrics/) as it supports computations on "cpu" as well as "cuda". </br>

- **Callbacks:** </br>

□ **EarlyStopping:** It uses a patience (an integer value) that determines the number of times to wait after last time validation loss improved before stopping the training. It only works if validation dataloader is used! </br>

□ **ModelCheckpoint:** It saves the checkpoint(s) (model, optimizer, scheduler) to the disk for each epoch. It also features to only save the best checkpoint. </br>

- **Loggers:** </br>

□ **CSVLogger:** It logs the hyperparameters (batch_size, epochs, optimizer, scheduler), loss and/ or metrics to a csv file. </br>

□ **TensorBoardLogger:** It logs the hyperparameters (batch_size, epochs, optimizer, scheduler), loss and/ or metrics to a tensorboard log file which then can be visualized in TensorBoard. </br>

- **Training Precisions:** </br>

□ **FP32**: This is the default training precision (single-precision) of the Trainer. Range:- 1.17e-38 to 3.40e38 </br>

□ **FP16**: It trains the model in FP16 (half-precision). It helps in reducing memory consumption. But it is recommended to **NOT** use this precision for training as it has small dynamic range and may result in NaN values during training. Range:- 6.10e-5 to 6.55e4 </br>

□ **BF16**: It trains the model in Brain-Floating, BFP16 (half-precision: a format that was developed by Google Brain, an artificial intelligence research group at Google). It helps in reducing memory consumption. It is preferred over FP16 as it has the same dynamic range as FP32, thus it will avoid producing NaN values during training. It is important to note that it is only supported on the Ampere architecture GPUs and the Trainer will raise an Exception if it is compiled with BF16 for CPU, or the GPU that does not support it. Range:- 1.17e-38 to 3.40e38 </br>

□ **FP16 AMP (Automatic Mixed Precision)**: It trains the model in both FP32 and FP16. The reduction in memory consumption may not be significant. It is preferred over just FP16 as it uses both the single and half precisions that will avoid producing NaN values during training. </br>

□ **BF16 AMP (Automatic Mixed Precision)**: It trains the model in both FP32 and BFP16. The reduction in memory consumption may not be significant. </br>

- **Gradient Accumulation:** It is a technique where you can train on bigger batch sizes than your machine would normally be able to fit into memory. This is done by accumulating gradients over several batches, and only stepping the optimizer after a certain number of batches have been performed. It will cost additional training time. </br>

- **Training History:** It records the training history:- epoch, learning rate, loss and/or metric(s). It also provides methods to plot the loss and metric(s). </br>

- **Training Progress Bar:** It shows progress bar for training while displaying epoch, learning rate, loss and/or metric(s) for each iteration as well as for an epoch.

</br>
</br>

## Examples

- Below is an example of profiling and its output taken from [this notebook](https://github.com/tpjesudhas/cv_component/tree/main/core/CV_Utils/Trainer/notebooks/1_Utilizing_Trainer.ipynb) </br>

<img width="512" alt="profiling" src="https://user-images.githubusercontent.com/50489165/224521425-6155462b-10da-4ed5-be35-48e2ef2388a7.png">

</br>
</br>

- Below is an example of training the model using the Trainer, taken from [this notebook](https://github.com/tpjesudhas/cv_component/tree/main/core/CV_Utils/Trainer/notebooks/1_Utilizing_Trainer.ipynb). Also, this notebook provides a means to have the better understanding of the utilization of features the Trainer provides and on how to include it in your pytorch projects. </br>

<img width="720" alt="image" src="https://user-images.githubusercontent.com/50489165/224524370-7530ef27-44a7-4a4f-98d7-4b6ff892c362.png">

<img width="720"  alt="progress_bar" src="https://user-images.githubusercontent.com/50489165/224524386-345b9af8-318a-4a8b-b66e-2f754192bf30.gif">

</br>
</br>

### (Optional) What are FLOPs and MACs?
Complexity of the model can be measured using the model size, Floating Point Operations(FLOPs), Multiply-Accumulate Computations(MACs), the number of model parameters, and the inference latency. These are model KPIs. </br>

Let us understand what are FLOPs and MACs. </br>

- FLOPs:- </br>

We can determine the total amount of calculations the model will need to carry out in order to estimate the inference time for that model. FLoating point OPeration, or FLOP, actually is an operation involving a floating point value, including addition, subtraction, division, and multiplication falls under this category. </br>

- MACs:- </br>

Multiply-Accumulate Computations, or MACs. MAC is an operation that performs two operations - addition and multiplication. A neural network always performs additions and multiplications. (e.g input * weight + bias). We typically assume that 1 MAC = 2 FLOPs. </br>

- (Optional) FLOPS:- </br>

□ The FLOPS, with a capital S; It is not a model KPI. FLoating point OPerations per Second or FLOPS, is a rate that provides information about the quality of the hardware on which the model is supposed to be deployed. The inference will happen more quickly if the more operations per second we can perform. </br>

□ Estimating Inference Time even before training or building a model:- </br>

Consider a neural network that has 'x' Parameters (e.g. weights and biases). </br>

Layer 1 = w1∗p1 + w2∗p2 + ... + wn∗pn = 1000 FLOPs and so on for each layer. </br>

Let's say it requires a total of 1,000,000 (1M) FLOPs to produce an output. </br>

Consider a CPU with a 1 GFLOPS performance. Then the inference time = FLOPs/FLOPS = (1,000,000)/(1,000,000,000) = 1ms. </br>

General rule:- </br>

- The model should have few FLOPs while still being sufficiently complex to be useful. </br>

- The hardware should have a lot of FLOPS. </br>

</br>
</br>


Collated by : Dheeraj Madda [Please contact for any suggestions or comments]
