## Trainer </br>

Development of training PyTorch models is not so simple. The training scripts are not generalizable from project to project. For a project, one has to write custom training loops, keep track of loss and metric calculations, to use learning rate schedulers if necessary, and so on. Once it is done, utilizing these scripts for a different project requires a lot of code changes to be done and it is a time consuming task. </br>

This Trainer makes development of Pytorch training extremely easy and fast while making it as generic as possible from project to project. It helps an individual to focus more on data preprocessing and experimenting various model architectures rather than spending more than sufficient amount of time on writing training scripts. The Trainer also provides many interesting features which can be easily used as required. </br>

</br>
</br>

The Trainer features the following: </br>

- Supports **Model Profiling**: Model Size, Num_Parameters, MACs, FLOPs, Inference Latency </br>

- Supports **Model Types**: Single-Input-Single-Output, Single-Input-Multiple-Outputs, Multiple-Inputs-Multiple-Outputs </br>

- Supports **Learning Rate Scheduler**: OneCycleLR </br>

- Supports **Metrics**: Single or multiple metric(s) </br>

- Supports **Callbacks**: EarlyStopping, ModelCheckpoint </br>

- Supports **Training Precisions**: FP32, FP16 AMP, BF16 AMP </br>

- Supports **Gradient Accumulation**: Accumulates gradients over several batches </br>

- Supports **Training History**: Epoch, learning rate, loss and/ or metric(s) </br>

- Supports **Training Progress Bar**: Epoch, learning rate, loss and/ or metric(s) </br>

</br>
</br>

Note: </br>

- By default, the Trainer trains the model with FP32 precision and it only displays the progress bar with information of epoch, learning rate, loss and metric(s) if provided any. All other features can be used as required. </br>

- It is recommended and important that the loss and metric functions must return an averaged value or reduction by 'mean'. </br>

- It is suggested to use PyTorch 2.0 and above, as the Trainer is tested for the same. </br>

- It is recommended to use [torchmetrics](https://pypi.org/project/torchmetrics/) as it supports computations on "cpu" as well as "cuda". </br>

- To get started with this Trainer, please go through [this](https://github.com/tpjesudhas/cv_component/tree/main/core/CV_Utils/Trainer/notebooks/1_Torch_Trainer_Tutorial.ipynb) notebook. </br>

</br>
</br>

### Examples: </br>

<img width="580" alt="profiling" src="https://github.com/DheerajMadda/test/assets/50489165/62bf0e7f-7d5b-4f47-8e8e-32396d4beedc">

</br>
</br>

<img width="512" alt="trainer" src="https://github.com/DheerajMadda/test/assets/50489165/ac3d4570-a6a0-46d9-a266-a647371537a3">
<img width="720" alt="progress_bar" src="https://github.com/DheerajMadda/test/assets/50489165/af4370fa-d568-4d08-b0d3-1a5dc8094e32">

</br>
</br>
</br>

<img width="850" alt="history" src="https://github.com/DheerajMadda/test/assets/50489165/1743f4b9-9be7-4fa7-a81f-d562ad784534">

</br>
</br>

<img width="720" alt="image" src="https://github.com/DheerajMadda/test/assets/50489165/8ca3e0b0-3ea6-4838-aaaf-785c339e2c86">


</br>
</br>

#### Directory and files information:-
- [notebooks](https://github.com/tpjesudhas/cv_component/tree/main/core/CV_Utils/Trainer/notebooks/) -> contains the jupyter notebook. 

- [torch_trainer](https://github.com/tpjesudhas/cv_component/tree/main/core/CV_Utils/Trainer/torch_trainer) -> contains the profiling and training utilities.

- [utils](https://github.com/tpjesudhas/cv_component/tree/main/core/CV_Utils/Trainer/utils) -> contains the loss and metrics definition.

- [requirements.txt](https://github.com/tpjesudhas/cv_component/tree/main/core/CV_Utils/Trainer/requirements.txt) -> contains all the required libraries to be installed.

</br>
</br>

<hr>

### Please read the following to understand the features provided by the Trainer: </br>

- **Model Profiling:** It is always good to perform the model profiling before training to get the complexity of the model. It computes the Model Size, Num_Parameters, MACs, FLOPs, Inference Latency. It supports profiling for both devices, "cpu" and "cuda". </br>

- **Model Types:** </br>

&emsp;&emsp; □ **Single-Input-Single-Output:** The Trainer can train a model that accepts single input and produces a single output, a torch.tensor(). </br>

&emsp;&emsp; □ **Single-Input-Multiple-Outputs:** The Trainer can train a model that accepts single input and produces multiple outputs, i.e tuple(torch.tensor(), torch.tensor(), ..., torch.tensor()). </br>

&emsp;&emsp; □ **Multiple-Inputs-Multiple-Outputs:** The Trainer can train a model that accepts multiple inputs, i.e. torch.tensor(), torch.tensor(), ..., torch.tensor() and produces multiple outputs, i.e tuple(torch.tensor(), torch.tensor(), ..., torch.tensor()).</br>

- **Learning Rate Scheduler:** The Trainer only supports OneCycleLR scheduler. It is a widely used scheduler and unlike StepLR/ MultiStepLR and many other schedulers, it updates the optimizer's learning rate over each batch. It is based on a 2018 paper titled "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates" (https://arxiv.org/abs/1708.07120) </br>

- **Metrics:** The trainer supports single or multiple metric(s) by defining it as a dictionary, where key is the metric name and its value should be the metric function. </br>

- **Callbacks:** </br>

&emsp;&emsp; □ **EarlyStopping:** It uses a patience (an integer value) that determines the number of times to wait after last time validation loss improved before stopping the training. It only works if validation dataloader is used! </br>

&emsp;&emsp; □ **ModelCheckpoint:** It saves the checkpoint(s) (model, optimizer, scheduler) to the disk for each epoch. It also features to only save the best checkpoint. </br>

- **Training Precisions:** </br>

&emsp;&emsp; □ **FP32**: This is the default training precision (single-precision) of the Trainer. Range:- 1.17e-38 to 3.40e38 </br>

&emsp;&emsp; □ **FP16 AMP (Automatic Mixed Precision)**: It trains the model in both FP32 and FP16. The reduction in memory consumption may not be significant. It is preferred over true FP16 as it uses both the single and half precisions that will avoid producing NaN values during training. </br>

&emsp;&emsp; □ **BF16 AMP (Automatic Mixed Precision)**: Brain-Floating, BFP16 (half-precision: a format that was developed by Google Brain, an artificial intelligence research group at Google). It helps in reducing memory consumption. It has the same dynamic range as FP32. It is important to note that it is only supported on the Ampere architecture GPUs and the Trainer will raise an Exception if it is compiled with BF16 for CPU, or the GPU that does not support it. Range:- 1.17e-38 to 3.40e38. Now, BF16 AMP (Automatic Mixed Precision), it trains the model in both FP32 and BFP16. The reduction in memory consumption may not be significant. </br>

- **Gradient Accumulation:** It is a technique where you can train on bigger batch sizes than your machine would normally be able to fit into memory. This is done by accumulating gradients over several batches, and only stepping the optimizer after a certain number of batches have been performed. It will cost additional training time. </br>

- **Training History:** It records the training history:- epoch, learning rate, loss and/or metric(s). It also provides methods to plot the loss and metric(s). </br>

- **Training Progress Bar:** It shows progress bar for training while displaying epoch, learning rate, loss and/or metric(s) for each iteration as well as for an epoch.

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

&emsp;&emsp; □ The FLOPS, with a capital S; It is not a model KPI. FLoating point OPerations per Second or FLOPS, is a rate that provides information about the quality of the hardware on which the model is supposed to be deployed. The inference will happen more quickly if the more operations per second we can perform. </br>

&emsp;&emsp; □ Estimating Inference Time even before training or building a model:- </br>

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
