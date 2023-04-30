# DieHardNET

Repository for a reliable Deep Neural Network (DNN) model. DieHardNet stands for
**Die** (integrated circuit) **Har**dened Neural **Net**work

[comment]: <> (TODO: Replace by two images from john mcclane one 
classified with DieHardNet and other with an error
 l![Die hard photo]&#40;/diehard.jpg&#41;)

## Directories

The directories are organized as follows:

* hg_noise_injector - A module to inject realistic errors in the training process
* eval_fault_injection_cfg - Configuration files for NVBITFI for fault injection
* pytorch_scripts - PyTorch scripts for training and inference for the used DNNs. For information read
  the [README](/pytorch_scripts/README.md).

## Run the training process

<ol>
<li>With the Hans Gruber error injector</li>

```{r, engine='bash', code_block_name} 
$ ./main.py ....
```

<li>Without the error injector</li>

```{r, engine='bash', code_block_name} 
$ ./main.py ....
```

</ol>

## Run the inference

An example of running the inference with the pretrained hardened model

```{r, engine='bash', code_block_name} 
$ ./main.py ....
```

# To cite this work

```bibtex
@INPROCEEDINGS{diehardnet,
    author = {AUTHORS},
    title = {TITLE},
    booktitle = {PROCEEDINGS},
    series = {EVENT},
    year = {2022},
    isbn = {ISBN},
} 
```
